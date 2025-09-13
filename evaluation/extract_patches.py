import os
import h5py
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

def read_h5_coords(h5_path: str) -> np.ndarray:
    """
    读取H5文件中的坐标信息列表。
    
    此函数专门用于读取 H5 文件中名为 'coords' 的数据集。

    Args:
        h5_path (str): H5文件路径。

    Returns:
        np.ndarray: 坐标数组, 例如: [[160, 20778], [160, 20906], ...]。
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 文件未找到: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        if 'coords' not in f:
            raise KeyError(f"在 H5 文件 '{h5_path}' 中未找到名为 'coords' 的数据集。")
        coords = f['coords'][:]
    return coords

def extract_patches_from_wsi(
    h5_path: str,
    wsi_path: str,
    output_dir: str,
    patch_size: int,
    level: int = 0
):
    """
    从 H5 文件读取坐标，并使用这些坐标从 WSI 文件中提取图块并保存。
    (版本已更新，调用 read_h5_coords 函数)

    Args:
        h5_path (str): 包含坐标的 H5 文件的路径。
        wsi_path (str): Whole Slide Image (WSI) 文件的路径 (例如 .svs, .tif)。
        output_dir (str): 保存提取出的图块的目标文件夹路径。
        patch_size (int): 要提取的每个图块的边长（假设为正方形）。
        level (int, optional): 从 WSI 的哪一层级提取图塊。默认为 0 (最高分辨率)。
    """
    # --- 1. 输入验证 ---
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI 文件未找到: {wsi_path}")

    print(f"开始处理 WSI: {os.path.basename(wsi_path)}")
    print(f"坐标来源 H5: {os.path.basename(h5_path)}")

    # --- 2. 创建输出目录 ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"图块将保存至: {output_dir}")

    try:
        # --- 3. 调用新函数读取坐标 ---
        coords = read_h5_coords(h5_path)
        print(f"成功从 H5 文件加载 {len(coords)} 个坐标。")

        # --- 4. 打开 WSI 文件并开始提取 ---
        slide = openslide.OpenSlide(wsi_path)
        
        print(f"WSI 尺寸 (level 0): {slide.level_dimensions[0]}")
        print(f"提取层级: {level}, 提取尺寸: {patch_size}x{patch_size}")

        wsi_basename = os.path.splitext(os.path.basename(wsi_path))[0]

        # 使用 tqdm 显示进度条
        for i, (x, y) in enumerate(tqdm(coords, desc="正在提取图块")):
            x, y = int(x), int(y)

            # 读取图块
            patch = slide.read_region((x, y), level, (patch_size, patch_size))

            # 将图块转换为 RGB 格式
            patch_rgb = patch.convert('RGB')
            
            # 定义输出文件名
            output_filename = f"{wsi_basename}_level{level}_x{x}_y{y}.png"
            output_path = os.path.join(output_dir, output_filename)

            # 保存图块
            patch_rgb.save(output_path)
            
        slide.close()
        print("\n所有图块提取并保存成功！")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        if 'slide' in locals() and slide:
            slide.close()

