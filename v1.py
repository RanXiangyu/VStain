import argparse, os #解析命令行参数
# import h5py
import numpy as np
import os
from PIL import Image
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from utils.hdf5 import get_sorted_h5_files, get_sorted_wsi_files, read_h5_coords
from utils.feature_hook import FeatureHook

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# from create_patches_fp import WSIPatchExtractor
from patches_utils.create_patches_fp import WSIPatchExtractor


def load_model_from_config(config_path, ckpt_path, device="cuda", verbose=False):
    # verbose是否打印缺失/多余的 key
    print(f"[INFO] Loading model from: {ckpt_path}")
    config = OmegaConf.load(config_path)
    
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"[INFO] Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # 打印缺失
    model = instantiate_from_config(config.model)
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    
    if verbose:
        if len(missing_keys) > 0:
            print("[WARNING] Missing keys:")
            print(missing_keys)
        if len(unexpected_keys) > 0:
            print("[WARNING] Unexpected keys:")
            print(unexpected_keys)

    model.to(device)
    model.eval()
    return model


def get_opt():
    parser = argparse.ArgumentParser()
    # 文件路径设置
    parser.add_argument('--wsi', help='Path to WSI file', required=True)
    parser.add_argument('--sty'， help='Path to style image', required=True)
    parser.add_argument('--out', help='Output directory', required=True)
    # 切片和hdf5设置
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches')
    parser.add_argument('--stride', type=int, default=224, help='步长')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num_files', type=int, default=4, help='需要处理的文件数量')
    # 模型设置
    parser.add_argument('model_config', type=str, help='model coonfiguration 文件')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='预训练权重路径')
    parser.add_argument('--ckpt', type=str, help='Path to the model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    # 风格注入控制
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    # ddim设置
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    # 代码运行设置
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    # 模块设置
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')

    opt = parser.parse_args()

    return opt


def load_img(path):
    image = Image.open(path).convert('RGB')
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0 # 将图像转换为numpy数组并归一化到[0, 1]范围
    image = image[None].transpose(0, 3, 1, 2) # 添加批次维度并调整通道顺序,形状从 [H, W, 3] 变为 [1, H, W, 3]
    image = torch.from_numpy(image) # 转换为PyTorch张量并调整像素范围
    return 2.*image - 1.

def feature_extractor(img_dir, img_name, feature_dir, model, sampler, ddim_inversion_steps, uc, time_idx_dict, start_step, save_feature_timesteps, ddim_sampler_callback, save_feat=False):
    global feat_maps

    img_feature = None
    img_z_enc = None

    img_path = os.path.join(img_dir, img_name)
    init_img = load_img(img_path).to('cuda')
    img_feat_name = os.path.join(feature_dir, os.path.basename(img_name).split('.')[0] + '_sty.pkl')

    if len(feature_dir) > 0 and os.path.exists(img_feat_name):
        print(f"Loading style feature from {img_feat_name}")
        with open(img_feat_name, 'rb') as f:
            img_feature = pickle.load(f)
            img_z_enc = torch.clone(img_feature[0]['z_enc'])
    else:
        init_img = model.get_first_stage_encoding(model.encode_first_stage(init_img))  # [1, 4, 64, 64] z_0
        img_z_enc, _ = sampler.encode_ddim(init_img.clone(), num_steps = ddim_inversion_steps, \
                                        unconditional_conditioning = uc, \
                                        end_step = time_idx_dict[ddim_inversion_steps - 1 - start_step], \
                                        callback_ddim_timesteps = save_feature_timesteps, \
                                        img_callback = ddim_sampler_callback)
        img_feature = copy.deepcopy(feat_maps)
        img_z_enc = feat_maps[0]['z_enc']
        
        if save_feat and len(feature_dir) > 0:
                print(f"Saving style feature to {img_feat_name}")
                with open(img_feat_name, 'wb') as f:
                    pickle.dump(img_feature, f)

    return img_feature, img_z_enc

def main():
    opt = get_opt()

    extractor = WSIPatchExtractor(
        source=opt.wsi,
        save_dir=opt.out,
        patch_size=opt.patch_size,
        step_size=opt.stride,
        patch_level=0,
        seg = False,
        patch = True,  
        stitch = False,
        save_mask = False,
        auto_skip = True,
        num_files=opt.num_files,
    )

    extractor.extract_patches()

    # 文件夹准备
    feature_dir = os.path.join(opt.out, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    vstained_imgs_dir = os.path.join(opt.out, 'vstained_imgs')
    os.makedirs(stained_dir, exist_ok=True)

    qkv_extraction_block_indices = list(map(int, opt.attn_layer.split(','))) # 在unet的哪些block当中提取qkv
    ddim_inversion_steps = opt.ddim_inv_steps # ddim反演步数 50 (encode_ddim)
    save_feature_timesteps = ddim_steps = opt.save_feat_steps # 特征提取/风格注入的时间步 = 从noise生成图像的正向采样步骤（sample_ddim）

    model = load_model_from_config(config_path=opt.model_config, ckpt_path=opt.ckpt, device='cuda')
    vae = model.first_stage_model
    unet = model.model.diffusion_model
    text_encoder = model.cond_stage_model

    # ddimsampler准备
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) # ddim_steps就是循环采样的步数

    # 获取ddim的时间步和索引映射 并反转 
    time_range = np.flip(sampler.ddim_timesteps)
    """
    ddim_timesteps 在ddim.py中由 make_ddim_timesteps() 生成
        在采样过程中，DDIM 实际是从 t+1 到 t 反推图像，所以需要将时间步整体右移 1
        ddim_timesteps = [0+1, 16+1, ..., 784+1] 是正序的
    在sample的过程中实际上是调用 ddim.py中的ddim_sampling() 函数，该函数中进行以下循环：
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps) # 进度条包装
        for i, step in enumerate(iterator):

    
    """
    idx_time_dict = {} # 去噪时间步：ddim顺序索引
    time_idx_dict = {} # ddim索引：去噪时间步
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    # 初始化全局变量
    global feat_maps
    feat_maps = [{'config':{'gamma' : opt.gamma, 'T' : opt.T }} for _ in range(ddim_steps)]

    # 初始化特征钩子
    feature_hook = FeatureHook(
        feat_maps=feat_maps,
        idx_time_dict=idx_time_dict,
        attn_layer_idics=qkv_extraction_block_indices,
        unet_model=unet
    )
    ddim_sampler_callback = feature_hook.ddim_sampler_callback
    save_qkv_callback = feature_hook.save_qkv_callback
    save_qkv = feature_hook.save_qkv

    # 获取h5文件和wsi文件
    h5_files = get_sorted_h5_files(opt.out)
    wsi_files = get_sorted_wsi_files(opt.wsi)

    # 染色风格图片 特征计算存储
    uc = model.get_learned_conditioning([""])   # 获取模型的无条件学习条件，也就是输入文本

    # 风格图片特征提取
    sty_img_list = sorted(os.listdir(opt.sty))  # 获取风格图片列表 
    for sty_img in sty_img_list:
        sty_feature, sty_z_enc = feature_extractor(opt.sty, sty_img, feature_dir, model, sampler, ddim_inversion_steps, uc, time_idx_dict, opt.start_step, save_feature_timesteps, ddim_sampler_callback, save_feat=True)


    # 遍历所有h5文件
    for idx, h5_file in tqdm(enumerate(h5_files), total=len(h5_files), desc="Processing H5 files"):
        # 读取坐标列表 打开slide
        coords_list = read_h5_coords(h5_files[idx])
        slide = openslide.OpenSlide(wsi_files[idx])
        W, H = slide.dimensions

        # 创建全景图需要的张量
        latent = torch.randn((1, self.unet.in_channels, H // 8, W // 8), device=self.device)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        
        for step_idx = 

        for coord in coords_list:
            x, y = coord

            region = slide.read_region((x, y), 0, (x + opt.patch_size, y + opt.patch_size))
            region = region.convert('RGB')

            # 进行处理




