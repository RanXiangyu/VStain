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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

'''
    python run_styleid.py \
    --cnt /data2/ranxiangyu/kidney_patch/patch_png/level1/22811he \
    --sty /data2/ranxiangyu/styleid_out/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/he2masson \
    --gamma 0.75 --T 1.5
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats \

    python main.py \
    --wsi /data2/ranxiangyu/vstain/wsi \
    --sty /data2/ranxiangyu/vstain/sty \
    --out /data2/ranxiangyu/vstain \
    --out_h5 /data2/ranxiangyu/vstain/h5 \
    --stride 480 \
    --batch_size 8 \
    --ddim_inv_steps 3 \
    --save_feat_steps 2 \
    --start_step 2 \
    --is_patch 

'''

feat_maps = [] # 全局变量，用于存储特征图
# test
def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
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


# 定义加载图片函数
def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

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
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()

    return opt

def main(opt):
    feat_path_root = opt.precomputed

    seed_everything(22) # 设置随机种子
    # 创建输出文件夹 如果有则不创建
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    # 创建特征文件夹 如果有则不创建
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}") # 加载模型配置文件 ldm/models/ldm/stable-diffusion-v1/v1-inference.yaml
    model = load_model_from_config(model_config, f"{opt.ckpt}") #从配置点checkpoint文件和配置文件加载模型 ldm/models/ldm/stable-diffusion-v1/model.ckpt

    # 设置设备和采样器
    # 通过 argparse 解析， 用于指定在哪些 U-Net block 中提取自注意力层的特征，并不需要在所有层数当中提取QKV
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # 判断是否有cuda
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model) # 初始化DDIM采样器 DDIM继承自model 作为DDPM/ldm的一个推广
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)  # 设置采样器的步长和eta
    time_range = np.flip(sampler.ddim_timesteps) # 获取时间步长的反转顺序 make_ddim_timesteps()，有uniform/quad两种采样方式，返回序列[1-n]的时间步骤，此处还进行了一个反转，反转为[n~1]
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

    # 定义回掉函数 
    # 在采用过程调用保存特征图
    """
    ddim_sampler_callback
    ├── save_feature_maps_callback
    │   └── save_feature_maps（提取 output_blocks 中的注意力 QKV）
    │       └── save_feature_map（逐个保存 Q/K/V）
    └── save_feature_map（保存 xt）

    blocks = unet_model.input_blocks  # 编码器阶段  提取局部特征
    blocks = unet_model.middle_block  # 瓶颈  全局语义特征
    blocks = unet_model.output_blocks # 解码器阶段  控制重建细节
    """
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
            block_idx += 1 # 多余的添加，并不会对block_idx产生影响，block_idx会被enumerate覆盖

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
    # DDIM一般是进行50步采样，默认值为49代表着从倒数第二步开始采样
    # 一般情况下，第50步的采样是最接近原始图像的
    # start——step控制起始步数
    precision_scope = autocast if opt.precision=="autocast" else nullcontext # 根据命令行参数设置精度
    uc = model.get_learned_conditioning([""])   # 获取模型的无条件学习条件，也就是输入文本
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f] # 设置形状 default=[4, 64, 64]
    sty_img_list = sorted(os.listdir(opt.sty))  # 获取风格图片列表 
    cnt_img_list = sorted(os.listdir(opt.cnt)) # 获取内容图片列表

    begin = time.time() # 开始计时
    # 遍历风格图片
    for sty_name in sty_img_list:
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)  #
        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        # 检查是否存在与预计算的风格特征文件
        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty)) # 第一步vae 获得图片的潜空间表示z_0 输出[B, C, H//8, W//8]，latent空间下的图像表示
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            """
            # 对init_sty（也就是latent x0）进行DDIM反演采样，推向z_T
            # 每当时间步是 callback_ddim_timesteps 中的值，就将当前特征加入feat_maps
            # encode_ddim(输入图像latent x0， 采样步骤，conditioning条件编码比如文本， unconditional_conditioning无条件编码，CFG scale, end_step结束步长，callback_ddim_timesteps指定在某些t调用回调——为了触发img_callback，img_callback图像回调函数)
            # 返回sty_z_enc是采样最后一步的x_T，即DDIM反向的起点 和feat_maps，但返回值没有用到，encode是为了在ddim的过程中调用回掉函数
            # time_idx_dict[ddim_inversion_steps-1-start_step = 0] =981 怀疑这一步写错了 end_step=0-50
            """
            sty_feat = copy.deepcopy(feat_maps) # 保存callback特征信息
            sty_z_enc = feat_maps[0]['z_enc'] # 提取第一个 step 时的 latent 编码，通常是最后一个 DDIM 时间步（最噪声的 z_T） ｜｜ 提取 z_enc 作为该风格图像的最终 latent 表示 ｜｜确保能够重构出图片

        # 遍历内容图片
        for cnt_name in cnt_img_list:
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            init_cnt = load_img(cnt_name_).to(device)
            cnt_feat_name = os.path.join(feat_path_root, os.path.basename(cnt_name).split('.')[0] + '_cnt.pkl')
            cnt_feat = None

            # ddim inversion encoding 预处理
            if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                print("Precomputed content feature loading: ", cnt_feat_name)
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h) 
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                    end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                    callback_ddim_timesteps=save_feature_timesteps,
                                                    img_callback=ddim_sampler_callback)
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']

            # 开始风格迁移
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        # inversion
                        # 生成输出文件名
                        output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"
                        # 打印反演结束时间
                        print(f"Inversion end: {time.time() - begin}")
                        # 进行AdaIn处理
                        if opt.without_init_adain:
                            adain_z_enc = cnt_z_enc
                        else:
                            adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                        # 合并特征图
                        feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                        # 如果命令行参数的without_attn_injection为真，则不注入特征图
                        if opt.without_attn_injection:
                            feat_maps = None

                        # inference shsape = [opt.C, opt.H // opt.f, opt.W // opt.f]  形状 default=[4, 64, 64]
                        # injected_features 用来控制风格 已经是merge之后的feat_maps
                        samples_ddim, intermediates = sampler.sample(S=ddim_steps,
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
                        """
                        手动 clamp + permute + numpy 更加复杂更加细粒度控制
                    
                        """
                        # 保存特征图    
                        img.save(os.path.join(output_path, output_name))
                        if len(feat_path_root) > 0:
                            print("Save features")
                            # 不保存内容特征图
                            # if not os.path.isfile(cnt_feat_name):
                            #     with open(cnt_feat_name, 'wb') as h:
                            #         pickle.dump(cnt_feat, h)
                            if not os.path.isfile(sty_feat_name):
                                with open(sty_feat_name, 'wb') as h:
                                    pickle.dump(sty_feat, h)
                            #     # 删除中间文件
                            # if os.path.isfile(cnt_feat_name):
                            #     os.remove(cnt_feat_name)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    opt = get_opt()
    main(opt)


