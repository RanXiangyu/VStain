import argparse, os #解析命令行参数
# import h5py
import numpy as np
import os
from PIL import Image
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import hdf5_utils

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# from create_patches_fp import WSIPatchExtractor
from patches_utils.create_patches_fp import WSIPatchExtractor


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
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')



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
    unet = model.model.diffusion_model
    text_encoder = model.cond_stage_model

    # ddimsampler准备
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    # 获取反转时间步


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




