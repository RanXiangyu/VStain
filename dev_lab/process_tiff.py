import os
import subprocess
import shutil
import argparse
import rasterio
import numpy as np

"""
python process_tiffs.py /data2/ranxiangyu/vstain --tile-size 1024
"""

def read_tiff(file_path):
    """
    使用 rasterio 读取 TIFF 文件，并返回图像数组及元数据。
    """
    with rasterio.open(file_path) as src:
        # 读取所有波段的数据为一个 NumPy 数组
        image_data = src.read()
        # 获取元数据，这对于后续写入操作至关重要
        profile = src.profile
    return image_data, profile

from rasterio.windows import Window

def crop_tiff_by_window(input_file, output_file, x_offset, y_offset, width, height):
    """
    根据给定的像素偏移和尺寸裁切TIFF文件。
    
    参数:
    x_offset (int): 裁切窗口左上角的列号 (X坐标)
    y_offset (int): 裁切窗口左上角的行号 (Y坐标)
    width (int):    裁切窗口的宽度
    height (int):   裁切窗口的高度
    """
    with rasterio.open(input_file) as src:
        # 定义一个裁切窗口
        window = Window(x_offset, y_offset, width, height)
        
        # 获取源文件的元数据 profile
        profile = src.profile
        
        # 计算并更新裁切后图像的地理变换参数
        # 这是确保裁切后的图像在GIS软件中位置正确的关键步骤
        profile['transform'] = rasterio.windows.transform(window, src.transform)
        profile['width'] = width
        profile['height'] = height

        # 读取窗口内的数据并写入新文件
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(src.read(window=window))

# 使用示例
output_path = 'output_cropped.tif'
crop_tiff_by_window(tiff_path, output_path, x_offset=100, y_offset=200, width=500, height=400)

print(f"图像已成功裁切并保存至: {output_path}")

if __name__ == "__main__":
    # 使用示例
    tiff_path = 'input.tif'
    image_array, meta_profile = read_tiff(tiff_path)

    # 打印图像数据的形状 (波段数, 高度, 宽度)
    print(f"图像尺寸: {image_array.shape}")
    # 打印元数据信息
    print("元数据 Profile:")
    print(meta_profile)

