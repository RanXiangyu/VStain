import h5py
import numpy as np
import os

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
        if file.endswith('.svs') or file.endswith('.tif') or file.endswith('.tiff'):
            wsi_path = os.path.join(wsi_dir, file)
            wsi_files.append(wsi_path)

    # 按文件名字典序排序
    wsi_files.sort()

    print(f"在 '{wsi_dir}' 中找到 {len(wsi_files)} 个 wsi 文件")
    return wsi_files


def read_h5_coords(h5_path):
    """
    读取H5文件中的坐标信息列表
    Args:
        h5_path (str): H5文件路径
    Returns:
        np.array: 坐标数组  [[  160 20778][  160 20906][  160 21034]
    """
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
    return coords
