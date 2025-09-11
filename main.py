import argparse, os #解析命令行参数
import torch
import numpy as np
from omegaconf import OmegaConf  # 加载模型配置文件
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything #设置随机种子
from torch import autocast
from contextlib import nullcontext
import copy
from pathlib import Path
import time
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


from wsi_core.WSIDataset import WSIDataset
from utils.load_img import load_img
from utils.hdf5 import get_sorted_h5_files, get_sorted_wsi_files, read_h5_coords
from wsi_core.create_patches_fp import WSIPatchExtractor
from utils.reconstruct_vips import reconstruct_with_original_background, bigtiff_to_thumbnail, save_pyramidal_tiff

'''
python main.py \
--sty /mnt/hfang/data/VStain/sty \
--wsi_dir /home/hfang/rxy/kidney_wsi \
--output /mnt/hfang/data/VStain/output \
--out_h5 /mnt/hfang/data/VStain/h5 \
--precomputed /mnt/hfang/data/VStain/precomputed_feats \
--stride 512 --patch_size 512 --num_files 1 --is_patch

'''

feat_maps = [] # 全局变量，用于存储特征图

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step): # i < 1 的步被跳过 所以只处理 i=1 到 49 的
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps

# 定义AdaIN函数 自适应实例归一化
def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

# 模型加载函数
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def get_opt():
    parser = argparse.ArgumentParser()
    # 文件路径设置
    parser.add_argument('--sty', default = '/data2/ranxiangyu/vstain/sty')
    parser.add_argument('--wsi_dir', default = '/data2/ranxiangyu/vstain/wsi')
    # parser.add_argument('--h5_dir', default = '/data2/ranxiangyu/vstain/h5/patches')
    parser.add_argument('--output', type=str, default='/data2/ranxiangyu/vstain/output')
    parser.add_argument('--out_h5', help='Output directory', required=True)
    parser.add_argument('--precomputed', type=str, default='/data2/ranxiangyu/vstain/precomputed_feats', help='保存预训练权重')
    # 切片和hdf5设置
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches')
    parser.add_argument('--stride', type=int, default=512, help='步长')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num_files', type=int, default=4, help='需要处理的文件数量')
    parser.add_argument("--is_patch", action="store_true", help="是否需要进行切片处理")
    # 图像设置
    parser.add_argument('--patch_level', default = 0, type=int)
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    # DDIM设置
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    # 风格注入设置
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    # 模型设置
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    # 代码运行设置
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()

    return opt

def main(opt):
    seed_everything(22) # 设置随机种子


    device = "cuda" if torch.cuda.is_available() else "cpu"


    wsi_extractor = WSIPatchExtractor()
    if opt.is_patch:
        #  在这一步利用clam生成重叠的点
        # 修改代码 auto_skip=False 不自动跳步
        wsi_extractor.process(
            source=opt.wsi_dir,
            save_dir=opt.out_h5,
            patch_size=opt.patch_size,
            step_size=opt.stride,
            patch_level=opt.patch_level,
            seg=True,
            patch=True,
            stitch=False,
            save_mask=False,
            auto_skip=False,
            num_files=opt.num_files
        )
    else: # 检查作用
        wsi_extractor.process(source=opt.wsi, save_dir=opt.out_h5, patch_size=opt.patch_size, step_size=opt.patch_size, patch_level=0, seg=True, patch=True, stitch=False, save_mask=False, num_files=opt.num_files)
    

    # 特征文件夹的创建
    feat_path_root = opt.precomputed
    os.makedirs(opt.output, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}") 
    model = load_model_from_config(model_config, f"{opt.ckpt}") 
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model) # 初始化DDIM采样器 DDIM继承自model 作为DDPM/ldm的一个推广
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)  # 设置采样器的步长和eta
    time_range = np.flip(sampler.ddim_timesteps)
    """
    建立时间步长和索引之间的映射
    idx time_dict = {981: 0, 961: 1, ... 1: 50} 1-50索引 1-1000timestep
    time_idx_dict = {0: 981, 1: 961, ... 49: 21, 50: 1}
    """
    idx_time_dict = {} # 把当前时间步timestep映射到feat map中的索引位置
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed() # 获取随机种子
    opt.seed = seed

    global feat_maps # 初始化全局变量
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i) # [B, num_heads, N, head_dim]
        save_feature_map(xt, 'z_enc', i) # [B, C, H, W]（latent）保存图像本身在潜空间的内容，可以可视化图像的演化过程

    # 保存当前时间步
    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        # stable diffusion的上采样 output_blocks有12个，在命令行输入的时候确定了再那几个图层进行保存qkv
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                # 包含多个子模块 并且是spatial transformer 通常在block[1]中
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)

    # 保存输出块的特征图 主要是为了简化调用save_feature_maps
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    # 保存单个时间步的 q//k/v 特征图
    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    # 加载图像并进行风格迁移
    start_step = opt.start_step # 从命令行获取开始步长 default=49
    precision_scope = autocast if opt.precision=="autocast" else nullcontext # 根据命令行参数设置精度
    uc = model.get_learned_conditioning([""])   # 获取模型的无条件学习条件，也就是输入文本
    shape = [opt.C, opt.patch_size // opt.f, opt.patch_size // opt.f] # 设置形状 default=[4, 64, 64]


    begin = time.time() # 开始计时

    # 修正后的代码块
    h5_dir = os.path.join(opt.out_h5, "patches")

    h5_files = sorted(Path(h5_dir).glob("*.h5"))
    wsi_files = {f.stem: f for f in Path(opt.wsi_dir).glob("*") if f.suffix.lower() in [".svs", ".tif", ".tiff"]}
    """ wsi_files = {
    "22811he": Path("/path/to/22811he.svs"),
    "22812he": Path("/path/to/22812he.tif"),
    ...
    } """
    for h5_file in h5_files:
        stem = h5_file.stem
        if stem not in wsi_files:
            print(f"⚠️ 跳过 {h5_file.name}，未找到对应 WSI 文件")
            continue
        
        wsi_path = wsi_files[stem]
        print(f"=== 开始处理: {stem} ===")
        print(f"WSI: {wsi_path}")
        print(f"H5 : {h5_file}")

        # dataset = WSIDataset([str(wsi_path)], [str(h5_file)], patch_size=opt.patch_size, level=opt.patch_level)
        dataset = WSIDataset(str(wsi_path), str(h5_file), patch_size=opt.patch_size, level=opt.patch_level)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # 遍历风格图片
        sty_img_list = sorted(Path(opt.sty).glob("*.*"))
        for sty_path in sty_img_list:
            init_sty = load_img(str(sty_path)).to(device)
            seed = -1
            sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_path.name).split('.')[0] + '_sty.pkl')
            sty_z_enc = None

            # 检查是否存在与预计算的风格特征文件
            if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
                print("Precomputed style feature loading: ", sty_feat_name)
                with open(sty_feat_name, 'rb') as h:
                    sty_feat = pickle.load(h)
                    sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
            else:
                init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty)) # 第一步vae 获得图片的潜空间表示z_0 输出[B, C, H//8, W//8]，latent空间下的图像表示
                sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), 
                                                   num_steps=ddim_inversion_steps, 
                                                   unconditional_conditioning=uc, 
                                                   end_step=start_step,
                                                   callback_ddim_timesteps=save_feature_timesteps,
                                                   img_callback=ddim_sampler_callback)
                sty_feat = copy.deepcopy(feat_maps) # 保存callback特征信息
                sty_z_enc = feat_maps[0]['z_enc'] 

            # 1为当前的WSI和风格创建特定的输出目录
            current_output_dir = Path(opt.output) / stem / sty_path.stem # output/wsi_file_stem/style_image_stem/
            current_output_dir.mkdir(parents=True, exist_ok=True)

            tiff_output_filename = f"{stem}_stylized_{sty_path.stem}_reconstructed.tiff"
            tiff_output_path = Path(opt.output) / stem / tiff_output_filename

            thumb_output_filename = f"{stem}_stylized_{sty_path.stem}_reconstructed.png"
            thumb_output_path = Path(opt.output) / stem / thumb_output_filename

        
            # for i, (region_tensor, coord) in enumerate(dataloader):
            for i, (region_tensor, coord) in enumerate(tqdm(dataloader, desc="Processing patches", total=len(dataloader))):
                region_tensor = region_tensor.to(device) # 将图像数据移动到指定设备（通常是 GPU）
                coord = coord.numpy()[0] # 获取坐标并转换为 numpy 数组

                region_z_0 = model.get_first_stage_encoding(model.encode_first_stage(region_tensor)) # 第一步vae 获得图片的潜空间表示z_0 输出[B, C, H//8, W//8]，latent空间下的图像表示
                region_z_enc, _ = sampler.encode_ddim(
                        region_z_0.clone(),
                        num_steps=ddim_inversion_steps,
                        unconditional_conditioning=uc,
                        end_step=start_step,
                        callback_ddim_timesteps=save_feature_timesteps,
                        img_callback=ddim_sampler_callback
                    )
                region_feat = copy.deepcopy(feat_maps)
                region_z_enc = feat_maps[0]['z_enc']

                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            # 进行AdaIn处理
                            if opt.without_init_adain:
                                adain_z_enc = region_z_enc
                            else:
                                adain_z_enc = adain(region_z_enc, sty_z_enc)
                        # 合并特征图
                        feat_maps = feat_merge(opt, region_feat, sty_feat, start_step=start_step)
                        if opt.without_attn_injection:
                            feat_maps = None

                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        batch_size=1,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=adain_z_enc,
                                                        injected_features=feat_maps,
                                                        start_step=start_step,
                                                        )
                        # 解码生成图像
                        x_samples_ddim = model.decode_first_stage(samples_ddim) #  latent 空间的图像 [B, 4, H, W] 解码成 RGB 图像 [B, 3, H*8, W*8]
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0) # 解码后的图像像素值在 [-1, 1] 之间，这一步把它缩放到 [0, 1] 区间（即正常图像的浮点数表示），并截断异常值
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy() # 将张量移到 CPU，并把通道维从 [B, 3, H, W] → [B, H, W, 3]，并转换为numpy数组，方便后续PIL保存
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2) # 又重新转换成 [B, 3, H, W] 的 PyTorch tensor
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        
                        
                        # 2. 使用坐标创建唯一的文件名
                        output_filename = f"{stem}_stylized_{sty_path.stem}_x{coord[0]}_y{coord[1]}.png"
                        output_filepath = current_output_dir / output_filename
                        
                        # 3. 保存图像
                        img.save(output_filepath)

            if len(feat_path_root) > 0:
                print("Save features")
                if not os.path.isfile(sty_feat_name):
                    with open(sty_feat_name, 'wb') as h:
                        pickle.dump(sty_feat, h)

            reconstruct_with_original_background(tile_dir=current_output_dir,original_wsi_path=wsi_path,output_path=tiff_output_path)
            bigtiff_to_thumbnail(tiff_path=tiff_output_path, thumbnail_path=thumb_output_path, scale_factor=0.1)  # 可以调缩放比例

        dataset.close()


    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    opt = get_opt()
    main(opt)


