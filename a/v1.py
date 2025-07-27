import argparse, os #解析命令行参数
import h5py
import numpy as np
import os
from PIL import Image
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# from create_patches_fp import WSIPatchExtractor
from patches_utils.create_patches_fp import WSIPatchExtractor



def get_sorted_h5_files(output_dir):
    """
    获取 output_dir/patches 目录下的所有 H5 文件（不递归子目录），并按字典序排序
    Args:
        output_dir (str): 上级目录路径（不包括 patches）
    Returns:
        list: 字典序排序后的 H5 文件路径列表
    """
    h5_files = []
    patches_dir = os.path.join(output_dir, "patches")

    # 确保路径存在
    if not os.path.isdir(patches_dir):
        print(f"目录不存在：{patches_dir}")
        return []

    # 遍历 patches 目录中的文件（不递归）
    for file in os.listdir(patches_dir):
        if file.endswith('.h5'):
            h5_path = os.path.join(patches_dir, file)
            h5_files.append(h5_path)

    # 按文件名字典序排序
    h5_files.sort()

    print(f"在 '{patches_dir}' 中找到 {len(h5_files)} 个 H5 文件")
    return h5_files


def get_sorted_wsi_files(wsi_dir):
    """
    获取 output_dir/patches 目录下的所有 wsi 文件（不递归子目录），并按字典序排序
    """
    wsi_files = []

    # 确保路径存在
    if not os.path.isdir(wsi_dir):
        print(f"目录不存在：{wsi_dir}")
        return []

    # 遍历 patches 目录中的文件（不递归）
    for file in os.listdir(wsi_dir):
        if file.endswith('.svs ') or file.endswith('.tif') or file.endswith('.tiff'):
            wsi_path = os.path.join(patches_dir, file)
            wsi_files.append(wsi_path)

    # 按文件名字典序排序
    wsi_files.sort()

    print(f"在 '{wsi_dir}' 中找到 {len(wsi_files)} 个 wsi 文件")
    return wsi_files


def read_h5_coords(h5_path):
    """
    读取H5文件中的坐标信息
    Args:
        h5_path (str): H5文件路径
    Returns:
        np.array: 坐标数组  [[  160 20778][  160 20906][  160 21034]
    """
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
    return coords

# def load_model_from_config(config_path, ckpt_path, device='cuda'):
#     config = OmegaConf.load(config_path)
#     model = load_model_from_config(model_config, ckpt_path) 
#     model = model.to(device)


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
    save_feature_maps(unet_model.output_blocks , i, "output_block")

# 保存单个特征图
def save_feature_map(feature_map, filename, time):
    global feat_maps
    cur_idx = idx_time_dict[time]
    feat_maps[cur_idx][f"{filename}"] = feature_map

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi', help='Path to WSI file', required=True)
    parser.add_argument('--sty')
    parser.add_argument('--out', help='Output directory', required=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches')
    parser.add_argument('--stride', type=int, default=224, help='步长')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num_files', type=int, default=4, help='需要处理的文件数量')
    parser.add_argument('model_config', type=str, help='Path to the model configuration file')
    parser.add_argument('--ckpt', type=str, help='Path to the model checkpoint')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')

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

    model = load_model_from_config(config_path=opt.model_config, ckpt_path=opt.ckpt, device='cuda')
    vae = model.first_stage_model
    une = model.model.diffusion_model
    text_encoder = model.cond_stage_model

    h5_files = get_sorted_h5_files(opt.out)
    wsi_files = get_sorted_wsi_files(opt.wsi)

    for idx, h5_file in tqdm(enumerate(h5_files), total=len(h5_files), desc="Processing H5 files"):
        coords_list = read_h5_coords(h5_files[idx])
        slide = openslide.OpenSlide(wsi_files[idx])
        W, H = slide.dimensions

        # 创建全景图需要的张量
        latent = torch.randn((1, self.unet.in_channels, H // 8, W // 8), device=self.device)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        for coord in coords_list:
            x, y = coord

            region = slide.read_region((x, y), 0, (x + opt.patch_size, y + opt.patch_size))
            region = region.convert('RGB')

            # 进行处理



