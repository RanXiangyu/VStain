import argparse, os #解析命令行参数
import h5py
import numpy as np
import os
from PIL import Image
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt

# from create_patches_fp import WSIPatchExtractor
from patches_utils.create_patches_fp import WSIPatchExtractor

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

def extract_patches(arg):
    # 存储所有patch的列表
    patches = []
    
    source_path = os.path.join(arg.h5_source)
    slide_list = os.listdir(source_path)
    
    for slide in slide_list:
        slide_id = slide.split('.h5')[0]
        # 读取svs的文件
        wsl = openslide.OpenSlide(os.path.join(arg.source, slide_id + '.' + arg.type))
        # 将SVS图片转化为numpy
        img = np.array(wsl.read_region((0, 0), 0, wsl.dimensions))
 
        save_path = os.path.join(arg.save_dir, slide_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
 
        with h5py.File(os.path.join(source_path, slide), 'r') as hdf5_file:
            coord = hdf5_file['coords'][:]
            print(coord.shape)
            for x_y in coord:  # 每个h5文件中保存的坐标
                x = x_y[0] # x坐标
                y = x_y[1] # y坐标
                
                # 提取patch
                patch = img[y:y+arg.patch_size, x:x + arg.patch_size, :]
                
                # 保存到列表
                patches.append(patch)
                
                # 同时保存为PNG文件（可选）
                cv2.imwrite(os.path.join(save_path, slide_id + '_' + str(x) + '_' + str(y) + '.png'), patch)


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

    h5_files = get_sorted_h5_files(opt.out)
    
    coords_list = read_h5_coords(h5_files[0])

    for coord in coords_list:
        x, y = coord
    



