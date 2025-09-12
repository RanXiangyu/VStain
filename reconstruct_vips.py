import os
import re
# import pyvips # --- [移除] 不再需要 pyvips
import openslide
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import tifffile # --- [新增] 导入 tifffile 用于保存金字塔TIFF

# 关闭Pillow对巨大图像的解压炸弹保护，因为WSI本身就是巨大图像
Image.MAX_IMAGE_PIXELS = None



def bigtiff_to_thumbnail(tiff_path, thumbnail_path, scale_factor=0.1):
    """
    将 BigTIFF 文件生成缩略图
    """
    with Image.open(tiff_path) as img:
        width, height = img.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        # 生成缩略图
        img.thumbnail(new_size, Image.LANCZOS)
        img.save(thumbnail_path)
        print(f"缩略图已保存到: {thumbnail_path}")


def save_pyramidal_tiff(path, image, tile_size=(512, 512), levels=3):
    import tifffile
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        # 写原始图像
        tif.write(
            image,
            photometric='rgb',
            tile=tile_size,
            compression='zlib'
        )
        # 写金字塔层
        for level in range(1, levels):
            downsampled = image[::2**level, ::2**level]
            tif.write(
                downsampled,
                photometric='rgb',
                tile=tile_size,
                compression='zlib',
                subfiletype=1   # 表示这是 pyramid 层
            )


def reconstruct_with_original_background(tile_dir, output_path, original_wsi_path):
    """
    使用 Pillow 和 tifffile 重建图像，不依赖 pyvips。
    """
    # 1. 打开原始WSI文件并获取其最高分辨率（Level 0）的尺寸
    print(f"Reading original WSI: {original_wsi_path}")
    try:
        wsi = openslide.OpenSlide(original_wsi_path)
        full_width, full_height = wsi.dimensions
    except Exception as e:
        print(f"Error reading original WSI file: {e}")
        return
        
    print(f"Original WSI dimensions: {full_width} x {full_height}")

    # 2. 从WSI的低分辨率图层创建背景
    print("Creating background from a low-resolution WSI level...")
    best_level = wsi.get_best_level_for_downsample(32)
    level_dims = wsi.level_dimensions[best_level]
    
    # 使用OpenSlide读取这个低分辨率图层，得到一个Pillow图像对象
    low_res_pil = wsi.read_region((0, 0), best_level, level_dims).convert("RGB")
    wsi.close()

    # --- [修改] 使用 Pillow 将低分辨率背景放大到完整尺寸，作为我们的画布
    print("Resizing background to full resolution using Pillow...")
    # 使用 LANCZOS 算法可以获得较好的缩放质量
    canvas_pil = low_res_pil.resize((full_width, full_height), Image.LANCZOS)
    
    # 3. 找到所有风格化图块并将其粘贴到画布上
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.png')]
    if not tile_files:
        print(f"Warning: No tile files found in '{tile_dir}'. An image with only the background will be created.")
    else:
        all_tiles = []
        # 正则表达式已根据您之前的要求修正
        coord_pattern = re.compile(r'_x(\d+)_y(\d+)\.png')
        for filename in tile_files:
            match = coord_pattern.search(filename)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                full_path = os.path.join(tile_dir, filename)
                all_tiles.append({'path': full_path, 'x': x, 'y': y})

        print(f"Pasting {len(all_tiles)} stylized tiles onto the original background using Pillow...")
        for tile_info in tqdm(all_tiles, desc="Stitching Tiles"):
            # --- [修改] 使用 Pillow 打开图块并粘贴
            with Image.open(tile_info['path']) as tile_image:
                # 确保图块是RGB格式
                tile_image = tile_image.convert("RGB")
                # 将图块粘贴到画布的指定位置
                canvas_pil.paste(tile_image, (tile_info['x'], tile_info['y']))

    # 4. 将最终图像保存为金字塔TIFF格式
    print(f"Saving final pyramidal TIFF to {output_path} using tifffile...")
    
    # --- [修改] 使用 tifffile 保存图像
    # 首先将Pillow图像转换为Numpy数组
    final_image_np = np.array(canvas_pil)
    
    # 使用 tifffile.imwrite 进行保存
    # tile=(256, 256) -> 将图像切分为256x256的块，这是WSI的常见做法
    # photometric='rgb' -> 指定颜色空间
    # compression='jpeg' -> 使用JPEG压缩
    # pyramid=True -> 自动生成金字塔图层，这是WSI的核心
    save_pyramidal_tiff(output_path, final_image_np, tile_size=(512, 512), levels=4)

    
    print("Reconstruction complete.")
    print(f"Final WSI saved to: {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stitch tiles onto a background (pyvips-free version).")
    parser.add_argument('--tile_dir', type=str, required=True, help='Directory containing the stylized tiles.')
    # parser.add_argument('--tiff_output_path', type=str, required=True, help='Path to save the final .tiff file.')
    # parser.add_argument('--thumb_output_path', type=str, required=True, help='Path to save the final .tiff file.')
    parser.add_argument('--original_wsi', type=str, required=True, help='Path to the original WSI file to use as background.')
    args = parser.parse_args()
    
    base_name = os.path.basename(os.path.dirname(args.tile_dir))
    # 当前目录名作为 style_name
    style_name = os.path.basename(args.tile_dir)
    # 输出文件名
    tiff_output_filename = f"{base_name}_stylized_{style_name}_reconstructed.tiff"
    # 输出目录为 tile_dir 的上一级目录
    tiff_output_path = os.path.join(os.path.dirname(args.tile_dir), tiff_output_filename)

    thumb_output_filename = f"{base_name}_stylized_{style_name}_reconstructed.png"
    # 输出目录为 tile_dir 的上一级目录
    thumb_output_path = os.path.join(os.path.dirname(args.tile_dir), thumb_output_filename)


    print(tiff_output_path)
    print(thumb_output_path)

    reconstruct_with_original_background(args.tile_dir, tiff_output_path, args.original_wsi)
    bigtiff_to_thumbnail(tiff_output_path, thumb_output_path, scale_factor=0.1)  # 可以调缩放比例


"""
python reconstruct_vips.py --tile_dir /mnt/hfang/data/VStain/output/22811he/pas  --original_wsi /home/hfang/rxy/kidney_wsi/22811he.svs


"""