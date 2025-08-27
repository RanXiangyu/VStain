import argparse, os #解析命令行参数
# import h5py
import numpy as np
import os
from PIL import Image
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pickle
import copy
import tifffile


from utils.hdf5 import get_sorted_h5_files, get_sorted_wsi_files, read_h5_coords
from utils.feature_hook import FeatureHook

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# from create_patches_fp import WSIPatchExtractor
from wsi_core.create_patches_fp import WSIPatchExtractor

feat_maps = []


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

# wsi划分  用于背景处理
def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    # 潜在空间为 1/8
    panorama_height /= 8
    panorama_width /= 8
    # 计算在给定的宽度和高度下，需要多少个窗口；滑动窗口的标准计算方式
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)

    # 生成窗口坐标，逻辑是从左到右从上到下
    views = []
    for i in range(total_num_blocks): # i ： 0 - n-1
        w_start = int((i % num_blocks_width) * stride)
        h_start = int((i // num_blocks_width) * stride)
        views.append((w_start, h_start))  # 只需要左上角坐标
    return views

def get_opt():
    parser = argparse.ArgumentParser()
    # 文件路径设置
    parser.add_argument('--wsi', help='Path to WSI file', required=True)
    parser.add_argument('--sty', help='Path to style image', required=True)
    parser.add_argument('--out', help='Output directory', required=True)
    parser.add_argument('--out_h5', help='Output directory', required=True)
    # 切片和hdf5设置
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches')
    parser.add_argument('--stride', type=int, default=224, help='步长')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num_files', type=int, default=4, help='需要处理的文件数量')
    parser.add_argument("--is_patch", action="store_true", help="是否需要进行切片处理")
    # 模型设置
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model coonfiguration 文件')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt', help='Path to the model checkpoint')
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
    parser.add_argument('--C_latent', type=int, default=4, help='latent channels')
    # 代码运行设置
    # parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    # 在 get_opt() 函数中
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use (e.g., "0" or "0,1")')
    # 模块设置
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')

    opt = parser.parse_args()

    return opt


# 预处理图片 剪裁 归一化，调整通道
def preprocess_img(path):
    image = Image.open(path).convert('RGB')
    x, y = image.size
    # print(f"从 {path} 加载图片大小为 ({x}, {y}) ")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image) # (x, y)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0 # 将图像转换为numpy数组并归一化到[0, 1]范围 (512, 512, 3)
    image = image[None].transpose(0, 3, 1, 2) # 添加批次维度并调整通道顺序,形状从 [H, W, 3] 变为 [1, H, W, 3]
    image = torch.from_numpy(image) # 转换为PyTorch张量并调整像素范围  (1, 3, 512, 512)
    return 2.*image - 1.

def preprocess_region(region: Image.Image):
    image = region  # 已经是 PIL.Image.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # [1, 3, H, W]
    image = torch.from_numpy(image)
    return 2. * image - 1.  # 映射到 [-1, 1]



def extract_style_features(
    # --- 通用参数 ---
    model, 
    sampler, 
    uc,
    time_idx_dict, 
    save_feature_timesteps,
    # --- style参数 ---
    img_dir,
    img_name, 
    feature_dir, 
    start_step=49, 
    save_feat=False,
    ddim_inversion_steps=50, 
    ddim_sampler_callback=None
    ):

    global feat_maps
    # feat_maps = []
    
    img_feature = None
    img_z_enc = None
    
    img_path = os.path.join(img_dir, img_name)
    print(f"[INFO] 进行DDIM反演，获取特征图: {img_path}")
    init_img = preprocess_img(img_path).to('cuda')
    img_feat_name = os.path.join(feature_dir, os.path.basename(img_name).split('.')[0] + '_sty.pkl')

    # 1. 直接加载特征返回 style图片
    if os.path.exists(img_feat_name):
        print(f"[✓] 加载风格 Loading style feature from {img_feat_name}")
        with open(img_feat_name, 'rb') as f:
            img_feature = pickle.load(f)
            img_z_enc = torch.clone(img_feature[0]['z_enc'])
        return img_feature, img_z_enc

    # 2. 进行ddim反演，获取特征图
    init_img = model.get_first_stage_encoding(model.encode_first_stage(init_img))  # [1, 4, 64, 64] z_0
    img_z_enc, _ = sampler.encode_ddim(init_img.clone(), num_steps = ddim_inversion_steps, \
                                        unconditional_conditioning = uc, \
                                        end_step = time_idx_dict[ddim_inversion_steps - 1 - start_step], \
                                        callback_ddim_timesteps = save_feature_timesteps, \
                                        img_callback = ddim_sampler_callback)
    print(f"DDIM Inversion Steps: {ddim_inversion_steps}")
    print(f"Start Step: {start_step}")
    print(f"Timesteps to save features at: {save_feature_timesteps}")
    print(f"End Step Index: {ddim_inversion_steps - 1 - start_step}")
    print(f"End Step Timestep: {time_idx_dict[ddim_inversion_steps - 1 - start_step]}")

    img_feature = copy.deepcopy(feat_maps)

    # img_z_enc = feat_maps[0]['z_enc']
    img_z_enc = img_feature[0]['z_enc']


    if save_feat and len(feature_dir) > 0:
        print(f"保存风格 style feature to {img_feat_name}")
        with open(img_feat_name, 'wb') as f:
            pickle.dump(img_feature, f)

    return img_feature, img_z_enc

def wsi_decode(
        latent_tensor: torch.Tensor,
        model: nn.Module,
        patch_size: int,
        stride: int,
        downsample_factor: int = 8,
        device: str = 'cuda',
        verbose: bool = True
    ) -> np.ndarray:
    """
    Args:
        latent_tensor (torch.Tensor): 需要解码的完整latent张量，形状为 (1, C, H_latent, W_latent)。
        model (nn.Module): 包含 .decode_first_stage 方法的模型实例 (如 Stable Diffusion 的 VAE)。
        patch_size (int): 在latent空间中每个图块的大小。需要实验找到显存能承受的最大值以提高效率。
        stride (int): 在latent空间中滑动的步长。必须小于 patch_size 以形成重叠。
        downsampling_factor (int): VAE模型的下采样因子，通常是 8。
        device (str): 用于解码的设备，例如 'cuda' 或 'cpu'。
        verbose (bool): 是否显示进度条。
    Returns:
        np.ndarray: 解码并拼接完成的最终RGB图像，形状为 (H, W, 3)，数据类型为 uint8。
    
    if stride > patch_size:
        raise ValueError("Stride must be less than or equal to patch_size for overlapping.")

    with torch.no_grad():
        B, C, H_latent, W_latent = latent_tensor.shape

        H_pixel, W_pixel = H_latent * downsample_factor, W_latent * downsample_factor

        # 在cpu内存中创建最终输出图像画布
        output_image = np.zeros((H_pixel, W_pixel, 3), dtype=np.uint8)

        latent_tensor = latent_tensor.to("cpu") # 小块传输到gpu

        # 创建迭代器和进度条
        y_steps = range(0, H_latent, stride)
        x_steps = range(0, W_latent, stride)

        if verbose:
            pbar = tqdm(total=len(y_steps) * len(x_steps), desc="Tiled Decoding")

        for y in y_steps:
            for x in x_steps:
                # 1. 切分出一个patch
                y_end = min(y + patch_size, H_latent)
                x_end = min(x + patch_size, W_latent)
                latent_patch = latent_tensor[:, :, y:y_end, x:x_end]  # 获取

                # 2. 将小块送到GPU并执行编码
                latent_patch_gpu = latent_patch.to(device)
                decoded_patch_gpu = model.decode_first_stage(latent_patch_gpu)  # 解码

                # 3. 将结果转换格式并移回CPU
                decoded_patch_cpu = decoded_patch_gpu.squeeze(0)  # (C, H, W)
                decoded_patch_cpu = decoded_patch_cpu.permute(1, 2, 0) # (H, W, C)
                # decoded_patch_cpu = (decoded_patch_cpu.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() # 修改为以下
                # 1. 正确地将 [-1, 1] 范围映射到 [0, 1]
                decoded_patch_cpu = (decoded_patch_cpu + 1.0) / 2.0
                # 2. 再将 [0, 1] 范围映射到 [0, 255] 并转换类型
                decoded_patch_cpu = (decoded_patch_cpu.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

                # 4. 计算需要粘贴的区域和尺寸
                # 我们只粘贴由步长（stride）决定的“新”区域，以避免重叠区域的伪影
                y_pixel_start = y * downsampling_factor
                x_pixel_start = x * downsampling_factor
                
                 # 计算实际要从解码后patch中提取的区域大小
                h_paste_size = stride * downsampling_factor
                w_paste_size = stride * downsampling_factor

                # 处理图像边界，防止越界
                h_paste_size = min(h_paste_size, H_pixel - y_pixel_start)
                w_paste_size = min(w_paste_size, W_pixel - x_pixel_start)

                # 5. 从解码的patch中取出有效部分，粘贴到CPU大画布上
                output_image[
                    y_pixel_start : y_pixel_start + h_paste_size,
                    x_pixel_start : x_pixel_start + w_paste_size,
                    :
                ] = decoded_patch_cpu[
                    :h_paste_size,
                    :w_paste_size,
                    :
                ]
                
                if verbose:
                    pbar.update(1)
        
        if verbose:
            pbar.close()

         # 处理除以零的情况
        overlap_count[overlap_count == 0] = 1
        output_image /= overlap_count

        # 转换回 uint8 图像格式
        output_image = (output_image + 1.0) / 2.0 # 从 [-1, 1] 转换到 [0, 1]
        output_image = (output_image.clip(0, 1) * 255).astype(np.uint8)


    return output_image
    """
    if stride > patch_size:
        raise ValueError("Stride must be less than or equal to patch_size for overlapping.")

    with torch.no_grad():
        B, C, H_latent, W_latent = latent_tensor.shape
        H_pixel, W_pixel = H_latent * downsample_factor, W_latent * downsample_factor

        # 在CPU内存中创建最终输出图像画布
        output_image = np.zeros((H_pixel, W_pixel, 3), dtype=np.uint8)

        y_steps = range(0, H_latent, stride)
        x_steps = range(0, W_latent, stride)

        if verbose:
            pbar = tqdm(total=len(y_steps) * len(x_steps), desc="Tiled Decoding")

        for y in y_steps:
            for x in x_steps:
                y_end = min(y + patch_size, H_latent)
                x_end = min(x + patch_size, W_latent)
                
                # 直接在GPU上进行切片，latent_patch仍在GPU上
                latent_patch = latent_tensor[:, :, y:y_end, x:x_end]

                decoded_patch_gpu = model.decode_first_stage(latent_patch)

                decoded_patch_cpu = decoded_patch_gpu.squeeze(0).permute(1, 2, 0)

                # 1. 先将 [-1, 1] 范围映射到 [0, 1]
                decoded_patch_cpu = (decoded_patch_cpu + 1.0) / 2.0
                # 2. 再将 [0, 1] 范围映射到 [0, 255] 并转换类型
                decoded_patch_cpu = (decoded_patch_cpu.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

                y_pixel_start = y * downsample_factor
                x_pixel_start = x * downsample_factor
                
                h_paste_size = stride * downsample_factor
                w_paste_size = stride * downsample_factor

                h_paste_size = min(h_paste_size, H_pixel - y_pixel_start)
                w_paste_size = min(w_paste_size, W_pixel - x_pixel_start)

                output_image[
                    y_pixel_start : y_pixel_start + h_paste_size,
                    x_pixel_start : x_pixel_start + w_paste_size,
                    :
                ] = decoded_patch_cpu[
                    :h_paste_size,
                    :w_paste_size,
                    :
                ]
                
                if verbose:
                    pbar.update(1)
        
        if verbose:
            pbar.close()

    return output_image


def main():
    opt = get_opt()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    if opt.is_patch:
        wsi_extractor = WSIPatchExtractor()

        wsi_extractor.process(
            source=opt.wsi,
            save_dir=opt.out_h5,
            patch_size=opt.patch_size,
            step_size=opt.stride,
            patch_level=0,
            seg=True,
            patch=True,
            stitch=False,
            save_mask=False,
            auto_skip=True,
            num_files=opt.num_files
        )

    # wsi_extractor.extract_patches()

    # 文件夹准备
    feature_dir = os.path.join(opt.out, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    vstained_imgs_dir = os.path.join(opt.out, 'vstained_imgs')
    os.makedirs(vstained_imgs_dir, exist_ok=True)

    qkv_extraction_block_indices = list(map(int, opt.attn_layer.split(','))) # 在unet的哪些block当中提取qkv
    ddim_inversion_steps = opt.ddim_inv_steps # ddim反演步数 50 (encode_ddim)
    # save_feature_timesteps = ddim_steps = opt.save_feat_steps # 特征提取/风格注入的时间步 = 从noise生成图像的正向采样步骤（sample_ddim） 50
    ddim_steps = opt.save_feat_steps

    model = load_model_from_config(config_path=opt.model_config, ckpt_path=opt.ckpt, device='cuda')
    vae = model.first_stage_model
    unet = model.model.diffusion_model
    text_encoder = model.cond_stage_model

    # ddimsampler准备
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) # ddim_steps就是循环采样的步数
    # 在这个地方schedule之后，sampler.ddim_timesteps 就是采样的时间步已经确定了

    save_feature_timesteps = ddim_steps

    # 获取ddim的时间步和索引映射 并反转 
    time_range = np.flip(sampler.ddim_timesteps)
    """
    ddim_timesteps 在ddim.py中由 make_ddim_timesteps() 生成
        在采样过程中，DDIM 实际是从 t+1 到 t 反推图像，所以需要将时间步整体右移 1
        ddim_timesteps = [0+1, 16+1, ..., 784+1] 是正序的
    在sample的过程中实际上是调用 ddim.py中的ddim_sampling() 函数，该函数中进行以下循环：
        timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps) # 进度条包装
        for i, step in enumerate(iterator):
    —— 所以在主程序的逻辑循环中，应该是reverse的sampler.ddim_timestep，也就是time_range
    idx time_dict = {981: 0, 961: 1, ... 1: 50} 1-50索引 1-1000timestep
    time_idx_dict = {0: 981, 1: 961, ... 49: 21, 50: 1}
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
    sty_ddim_sampler_callback = feature_hook.ddim_sampler_callback
    cnt_ddim_sampler_callback = feature_hook.content_q_update_callback

    # 获取h5文件和wsi文件
    h5_files = get_sorted_h5_files(opt.out_h5)
    wsi_files = get_sorted_wsi_files(opt.wsi)

    # 染色风格图片 特征计算存储
    uc = model.get_learned_conditioning([""])   # 获取模型的无条件学习条件，也就是输入文本

    # 风格图片特征提取
    sty_img_list = sorted(os.listdir(opt.sty))  # 获取风格图片列表 

    
    for sty_img in sty_img_list:
        sty_feat, sty_z_enc = extract_style_features(img_dir=opt.sty, img_name=sty_img, feature_dir=feature_dir,save_feat=True,start_step=opt.start_step, model=model, sampler=sampler, uc=uc, time_idx_dict=time_idx_dict, ddim_sampler_callback=sty_ddim_sampler_callback, ddim_inversion_steps=ddim_inversion_steps, save_feature_timesteps=save_feature_timesteps)

    # 遍历所有h5文件
    for idx, h5_file in tqdm(enumerate(h5_files), total=len(h5_files), desc="Processing H5 files"):
        # 读取坐标列表 打开slide
        coords_list = read_h5_coords(h5_files[idx])
        slide = openslide.OpenSlide(wsi_files[idx])
        W, H = slide.dimensions
        
        patch_size = opt.patch_size
        patch_size_latent = opt.patch_size // 8 # 在潜空间当中需要缩放8倍

        # 创建全景图需要的张量
        latent = torch.zeros((1, opt.C_latent, H // 8, W // 8), device=device)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)
        # blank = torch.zeros_like(latent) # 记录有哪些部分没有被去噪，clam已经去掉的部分
        # v1.py, main() 函数中
        blank = torch.zeros_like(latent, dtype=torch.bool)
        
        # 构建并保存原始图像的latent —— 用于融合
        original_z0 = torch.zeros_like(latent)
        views = get_views(panorama_height=H, panorama_width=W,window_size=512, stride=512)

        # 循环遍历coords_list，处理背景 不应该是coord_list
        print(f"开始处理背景")
        for coord in tqdm(views, desc="Building Original z0 background", unit="patch"):
            x_pixel, y_pixel = coord
            x_latent, y_latent = x_pixel // 8, y_pixel // 8

            region_img = slide.read_region((x_pixel, y_pixel), 0, (patch_size, patch_size)).convert("RGB")
            region_tensor = preprocess_region(region_img).to(device)  # 转换为张量并移动到设备上

            # 只进行VAE编码，不加噪
            z_0_patch = model.get_first_stage_encoding(model.encode_first_stage(region_tensor))
        
            original_z0[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent] += z_0_patch
            # print(f"处理背景完成，当前坐标: {coord}, 原始z0形状: {original_z0.shape}")


        # 初始latent的获取过程，ddim reverse
        print(f"开始获取初始latent")
        for coord in tqdm(coords_list, desc="Encoding patches", unit="patch"):
            x_pixel, y_pixel = coord
            x_latent = x_pixel // 8
            y_latent = y_pixel // 8

            region_img = slide.read_region((x_pixel, y_pixel), 0, (patch_size, patch_size)).convert("RGB")
            region_tensor = preprocess_region(region_img).to(device)  # 转换为张量并移动到设备上
            z_0_patch = model.get_first_stage_encoding(model.encode_first_stage(region_tensor))  # shape: [1, C, h//8, w//8]
            # encode_ddim是一个“加噪”的过程，实现的
            z_T_patch, _ = sampler.encode_ddim(
                    z_0_patch.clone(),
                    num_steps=ddim_inversion_steps,
                    unconditional_conditioning=uc,
                    end_step=time_idx_dict[ddim_inversion_steps - 1 - opt.start_step]
                )

            latent[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent] += z_T_patch
            count[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent] += 1
            blank[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent] = True # 记录有哪些部分被包含到去噪list当中 

        latent = torch.where(count > 0, latent / count, latent)  # 避免除以0

        print(f"初始latent获取完成")


        iterator = tqdm(time_range, desc='DDIM Sampler', total=ddim_inversion_steps)


        # 开启循环 遍历50步DDIM去噪
        for i,  step in enumerate(iterator):    
            print(f"开始第 {i+1} 步去噪，当前时间步: {step}")            
            count.zero_()
            value.zero_()

            # a. 为当前步骤准备参数
            index = ddim_inversion_steps - i - 1
            # .to(device) 确保 ts 张量和模型在同一设备上
            ts = torch.full((1,), step, device=device, dtype=torch.long)
            
            # 循环遍历每个窗口
            for coord in coords_list:
                x_pixel, y_pixel = coord # 在潜空间当中需要缩放8倍
                x_latent = x_pixel // 8
                y_latent = y_pixel // 8

                # 进行处理 在这一步为了简化，直接尝试采用每一步提取qkv
                # 1. 提取当前内容图的patch
                region_img = slide.read_region((x_pixel, y_pixel), 0, (patch_size, patch_size)).convert("RGB")
                region_tensor = preprocess_region(region_img).to(device)  # 转换为张量并移动到设备上
                # a. VAE编码
                z_0_patch = model.get_first_stage_encoding(model.encode_first_stage(region_tensor))  # shape: [1, C, h//8, w//8]
                # b. DDIM Inversion 捕获特征
                _, _ = sampler.encode_ddim(z_0_patch.clone(), 
                                            num_steps=ddim_inversion_steps,
                                            unconditional_conditioning=uc,
                                            end_step=i,
                                            # callback_ddim_timesteps=i,
                                            img_callback=cnt_ddim_sampler_callback)
                # 此处callback_ddim_timesteps不能为0，解决： 不传入callback_ddim_timesteps，则 encode__ddim() 会在每个时间步（np.flip(self.ddim_timesteps)）都调用 img_callback
                
                injected_features_i = feat_maps[i] 


                # 从全局latent当中提取出 patch view
                latent_patch = latent[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent]

                # 2. 执行单步去噪
                latents_view_denoised, _ = sampler.p_sample_ddim(
                    x=latent_patch,
                    c=None,
                    t=ts , index=index, 
                    unconditional_conditioning=uc,
                    injected_features=injected_features_i,
                )

                value[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent] += latents_view_denoised
                count[:, :, y_latent:y_latent+patch_size_latent, x_latent:x_latent+patch_size_latent] += 1

            # 融合所有patches -- new latent
            latent = torch.where(count > 0, value / count, value)
            print(f"第 {i+1} 步去噪完成，当前时间步: {step}")
        print(f"所有去噪步骤完成")
        
        # 循环结束之后，解码 latent 空间
        # a. 白色背景latent向量填充blank为false的部分
        final_combined_latent = torch.where(
            blank,                # 条件：如果这个区域被处理过 (True)
            latent,               # 就使用去噪生成的结果
            original_z0           # 否则 (False)，就使用原始H&E图的latent（即背景latent）
        )

        print("Starting final tiled decoding...")
        final_wsi_np = wsi_decode(
            latent_tensor=final_combined_latent,
            model=model,
            patch_size=128,  # 在latent空间，对应1024x1024像素，可根据显存调整
            stride=96,       # 在latent空间，对应768x768像素的步长
            device=device
        )
        
        # 不够大，保存图片超过了4gb
        print("使用 tifffile 保存为 BigTIFF 格式...")
        output_path = os.path.join(opt.out, f"final_decoded_{idx}_{opt.ddim_inv_steps}.tiff")
        # 使用 bigtiff=True 来确保支持大于4GB的文件
        tifffile.imwrite(output_path, final_wsi_np, bigtiff=True)
        print(f"图像已成功保存到: {output_path}")


if __name__ == "__main__":
    main()

