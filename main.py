import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
import time
import pickle
import math

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from styleid_package import adain, feat_merge, load_img, save_img_from_sample, load_model_from_config

import torchvision.transforms as transforms
import torch.nn.functional as F

feat_maps = []  # 全局变量，用于存储特征图


# ----- 新增 MultiDiffusion 相关函数 -----

def get_views_coordinates(panorama_height, panorama_width, window_size=512, stride_rate=0.5):
    """
    计算全景图像上的视图坐标
    """
    stride = int(window_size * stride_rate)
    
    # 计算在高度和宽度方向上需要的视图数量
    h_views = max(1, 1 + math.ceil((panorama_height - window_size) / stride))
    w_views = max(1, 1 + math.ceil((panorama_width - window_size) / stride))
    
    # 如果只需要一个视图，居中放置
    if h_views == 1:
        h_centers = [panorama_height // 2]
    else:
        h_centers = np.linspace(window_size // 2, panorama_height - window_size // 2, h_views).astype(int)
        
    if w_views == 1:
        w_centers = [panorama_width // 2]
    else:
        w_centers = np.linspace(window_size // 2, panorama_width - window_size // 2, w_views).astype(int)
    
    # 生成所有视图的坐标
    coordinates = []
    for h_center in h_centers:
        for w_center in w_centers:
            h_start = max(0, h_center - window_size // 2)
            h_end = min(panorama_height, h_center + window_size // 2)
            w_start = max(0, w_center - window_size // 2)
            w_end = min(panorama_width, w_center + window_size // 2)
            
            coordinates.append((h_start, h_end, w_start, w_end))
    
    return coordinates

def generate_attention_mask(window_size, margin=32):
    """
    生成注意力掩码，用于平滑地融合多个重叠区域
    """
    mask = np.ones((window_size, window_size))
    
    # 创建边缘渐变
    for i in range(margin):
        fade = np.sin(0.5 * np.pi * i / margin)
        mask[i, :] *= fade
        mask[window_size - i - 1, :] *= fade
        mask[:, i] *= fade
        mask[:, window_size - i - 1] *= fade
    
    return torch.from_numpy(mask).float()

def process_view(model, sampler, init_latent, uc, ddim_steps, opt, cnt_feat, sty_feat, start_step=49):
    """
    处理单个视图区域
    """
    global feat_maps
    
    # 使用 AdaIn 处理
    if opt.without_init_adain:
        adain_z_enc = init_latent
    else:
        # 假设我们已经有了相应的风格潜变量
        adain_z_enc = adain(init_latent, sty_feat[0]['z_enc'])
    
    # 合并特征图
    feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
    
    # 如果命令行参数的without_attn_injection为真，则不注入特征图
    if opt.without_attn_injection:
        feat_maps = None
    
    # 采样生成结果
    shape = init_latent.shape[1:]
    samples_ddim, _ = sampler.sample(
        S=ddim_steps,
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
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    
    return x_samples_ddim

def get_opt():
    parser = argparse.ArgumentParser()
    # 原有参数
    parser.add_argument('--cnt', default='./data/cnt')
    parser.add_argument('--sty', default='./data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM inversion steps')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='Feature save steps')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM start step')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='precomputed features')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast', help='["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    
    # 新增 MultiDiffusion 相关参数
    parser.add_argument('--panorama_mode', action='store_true', help='Enable panorama processing')
    parser.add_argument('--panorama_height', type=int, default=512, help='Panorama height')
    parser.add_argument('--panorama_width', type=int, default=1024, help='Panorama width')
    parser.add_argument('--view_size', type=int, default=512, help='View window size')
    parser.add_argument('--stride_rate', type=float, default=0.8, help='Stride rate between views')
    parser.add_argument('--blend_margin', type=int, default=32, help='Margin for blending')
    
    opt = parser.parse_args()
    return opt

def main_multidiffusion(opt):
    feat_path_root = opt.precomputed

    seed_everything(22)
    # 创建输出文件夹
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    # 创建特征文件夹
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    # 设置设备和采样器
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    
    # 建立时间步长和索引之间的映射
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma': opt.gamma,
                'T': opt.T
                }} for _ in range(50)]

    # 定义回掉函数
    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    # 保存特定块的特征图
    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    # 保存输出块的特征图
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    # 保存单个特征图
    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    # 设置参数
    start_step = opt.start_step
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    # 获取风格和内容图片列表
    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))
    
    begin = time.time()
    
    # 遍历风格图片
    for sty_name in sty_img_list:
        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)
        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        # 检查是否存在预计算的风格特征文件
        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc,
                                               end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
                                               callback_ddim_timesteps=save_feature_timesteps,
                                               img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']

        # 遍历内容图片
        for cnt_name in cnt_img_list:
            cnt_name_ = os.path.join(opt.cnt, cnt_name)
            
            # 判断是否使用全景模式
            if opt.panorama_mode:
                # 读取原始图像大小
                with Image.open(cnt_name_) as img:
                    orig_width, orig_height = img.size
                
                # 设置全景图像尺寸（如果未指定则使用原始尺寸）
                if opt.panorama_width == 1024 and opt.panorama_height == 512:
                    panorama_width, panorama_height = orig_width, orig_height
                else:
                    panorama_width, panorama_height = opt.panorama_width, opt.panorama_height
                
                print(f"Processing panorama image with size: {panorama_width}x{panorama_height}")
                
                # 创建空白全景图像用于存储结果
                result_image = Image.new("RGB", (panorama_width, panorama_height))
                weight_map = np.zeros((panorama_height, panorama_width))
                
                # 计算视图坐标
                view_coordinates = get_views_coordinates(
                    panorama_height, 
                    panorama_width,
                    window_size=opt.view_size,
                    stride_rate=opt.stride_rate
                )
                
                print(f"Total {len(view_coordinates)} views to process")
                
                # 获取混合掩码
                blend_mask = generate_attention_mask(opt.view_size, margin=opt.blend_margin).to(device)
                
                # 处理每个视图
                for idx, (h_start, h_end, w_start, w_end) in enumerate(view_coordinates):
                    print(f"Processing view {idx+1}/{len(view_coordinates)}: ({h_start}:{h_end}, {w_start}:{w_end})")
                    
                    # 提取视图部分
                    view_img = Image.open(cnt_name_).convert("RGB")
                    view_img = view_img.resize((panorama_width, panorama_height), Image.Resampling.LANCZOS)
                    view_img = view_img.crop((w_start, h_start, w_end, h_end))
                    
                    # 确保视图大小为 view_size
                    actual_h = h_end - h_start
                    actual_w = w_end - w_start
                    
                    if actual_h != opt.view_size or actual_w != opt.view_size:
                        view_img = view_img.resize((opt.view_size, opt.view_size), Image.Resampling.LANCZOS)
                    
                    # 转换为张量并加载到设备
                    init_cnt = load_img(None, target_size=(opt.view_size, opt.view_size)).to(device)
                    init_cnt[0] = transforms.ToTensor()(view_img) * 2.0 - 1.0
                    
                    # 获取内容特征
                    init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                    cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, 
                                                      unconditional_conditioning=uc,
                                                      end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
                                                      callback_ddim_timesteps=save_feature_timesteps,
                                                      img_callback=ddim_sampler_callback)
                    cnt_feat = copy.deepcopy(feat_maps)
                    cnt_z_enc = feat_maps[0]['z_enc']
                    
                    # 处理视图并生成结果
                    with torch.no_grad():
                        with precision_scope("cuda"):
                            with model.ema_scope():
                                view_result = process_view(
                                    model, sampler, cnt_z_enc, uc, ddim_steps, 
                                    opt, cnt_feat, sty_feat, start_step
                                )
                    
                    # 转换为PIL图像
                    view_result = view_result[0].permute(1, 2, 0).cpu().numpy()
                    view_result = (view_result * 255).astype(np.uint8)
                    view_img = Image.fromarray(view_result)
                    
                    # 将结果放入全景图中，应用混合掩码
                    mask_np = blend_mask.cpu().numpy()
                    
                    # 确保掩码和视图大小匹配
                    if view_img.size[0] != opt.view_size or view_img.size[1] != opt.view_size:
                        view_img = view_img.resize((opt.view_size, opt.view_size), Image.Resampling.LANCZOS)
                        
                    # 确保区域大小不超过实际尺寸
                    h_end = min(h_start + opt.view_size, panorama_height)
                    w_end = min(w_start + opt.view_size, panorama_width)
                    
                    # 处理边缘情况
                    effective_h = h_end - h_start
                    effective_w = w_end - w_start
                    
                    # 将结果放入全景图
                    view_array = np.array(view_img)[:effective_h, :effective_w]
                    mask_part = mask_np[:effective_h, :effective_w]
                    
                    result = np.array(result_image)
                    result[h_start:h_end, w_start:w_end] = \
                        result[h_start:h_end, w_start:w_end] * (1 - mask_part)[..., np.newaxis] + \
                        view_array * mask_part[..., np.newaxis]
                    
                    # 更新权重图
                    weight_map[h_start:h_end, w_start:w_end] += mask_part
                    
                    # 更新结果图像
                    result_image = Image.fromarray(result.astype(np.uint8))
                
                # 处理权重图中的零权重区域（如果有）
                if np.min(weight_map) == 0:
                    print("Warning: Some areas have zero weight. Filling with nearest valid pixels.")
                    zero_mask = weight_map == 0
                    valid_mask = ~zero_mask
                    
                    # 简单填充：用最近的有效像素填充
                    result_array = np.array(result_image)
                    for c in range(3):  # RGB通道
                        channel = result_array[..., c]
                        if np.any(valid_mask):
                            # 创建一个简单的扩散效果
                            for _ in range(5):  # 几次迭代以扩散值
                                temp = channel.copy()
                                # 水平扩散
                                temp[:-1, :][zero_mask[:-1, :]] = channel[1:, :][zero_mask[:-1, :]]
                                temp[1:, :][zero_mask[1:, :]] = channel[:-1, :][zero_mask[1:, :]]
                                # 垂直扩散
                                temp[:, :-1][zero_mask[:, :-1]] = channel[:, 1:][zero_mask[:, :-1]]
                                temp[:, 1:][zero_mask[:, 1:]] = channel[:, :-1][zero_mask[:, 1:]]
                                channel = temp
                            result_array[..., c] = channel
                    
                    result_image = Image.fromarray(result_array.astype(np.uint8))
                
                # 保存全景图像结果
                output_name = f"{os.path.basename(cnt_name).split('.')[0]}_panorama_stylized_{os.path.basename(sty_name).split('.')[0]}.png"
                result_image.save(os.path.join(output_path, output_name))
                
            else:
                # 原始的非全景模式处理
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
                    cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, 
                                                      unconditional_conditioning=uc,
                                                      end_step=time_idx_dict[ddim_inversion_steps-1-start_step],
                                                      callback_ddim_timesteps=save_feature_timesteps,
                                                      img_callback=ddim_sampler_callback)
                    cnt_feat = copy.deepcopy(feat_maps)
                    cnt_z_enc = feat_maps[0]['z_enc']

                # 开始风格迁移
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"
                            
                            if opt.without_init_adain:
                                adain_z_enc = cnt_z_enc
                            else:
                                adain_z_enc = adain(cnt_z_enc, sty_z_enc)
                                
                            feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                            
                            if opt.without_attn_injection:
                                feat_maps = None

                            # inference
                            samples_ddim, _ = sampler.sample(
                                S=ddim_steps,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=adain_z_enc,
                                injected_features=feat_maps,
                                start_step=start_step,
                            )
                            
                            # 保存结果
                            save_img_from_sample(model, samples_ddim, os.path.join(output_path, output_name))
                            
                            # 保存特征
                            if len(feat_path_root) > 0:
                                print("Save features")
                                if not os.path.isfile(sty_feat_name):
                                    with open(sty_feat_name, 'wb') as h:
                                        pickle.dump(sty_feat, h)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    opt = get_opt()
    main_multidiffusion(opt)