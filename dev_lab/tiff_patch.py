import os
import subprocess
import shutil

# --- 全局配置 ---
# 脚本会自动检测使用 'convert' 还是 'magick'
IMAGEMAGICK_CMD = 'convert'

def check_imagemagick_command():
    """
    检查ImageMagick命令是否存在，并根据需要自动切换到 'magick'。
    返回 True 表示命令可用, False 表示不可用。
    """
    global IMAGEMAGICK_CMD
    if shutil.which(IMAGEMAGICK_CMD) is not None:
        return True
    
    # 如果 'convert' 找不到, 尝试 'magick' (适用于 ImageMagick 7+)
    elif shutil.which('magick') is not None:
        IMAGEMAGICK_CMD = 'magick'
        print("信息: 检测到 'magick' 命令，将使用它。")
        return True
    else:
        print(f"错误: 命令 'convert' 或 'magick' 未找到。")
        print("请确保 ImageMagick 已经安装并添加到了系统的 PATH 环境变量中。")
        return False

def tile_tiff(tiff_path, output_folder, tile_size, output_format='png'):
    """
    将指定的TIFF文件裁剪为多个指定大小的方块。

    :param tiff_path: 输入的TIFF文件的完整路径。
    :param output_folder: 保存所有裁剪出图块的文件夹。
    :param tile_size: 方块的边长（像素）。
    :param output_format: 输出图块的格式 ('png', 'jpg', 'jpeg', etc.)。
    :return: 成功时返回 True，失败时返回 False。
    """
    print("--- 开始执行切片任务 ---")
    
    # 1. 输入验证
    if not os.path.isfile(tiff_path):
        print(f"错误: 输入文件不存在 -> {tiff_path}")
        return False

    if not isinstance(tile_size, int) or tile_size <= 0:
        print(f"错误: 'tile_size' 必须是一个正整数，当前为 {tile_size}")
        return False

    # 2. 准备输出目录
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"图块将保存到: {output_folder}")
    except OSError as e:
        print(f"错误: 创建输出文件夹失败 -> {output_folder}\n{e}")
        return False
        
    # 3. 构建 ImageMagick 命令
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    
    # ImageMagick 会自动处理编号。'%d' 会被替换为 0, 1, 2, ...
    output_filename_template = f"{base_name}_tile_%d.{output_format}"
    output_path_template = os.path.join(output_folder, output_filename_template)

    # 命令结构: convert [输入文件] -crop [尺寸] +repage [输出模板]
    # -crop {size}x{size}: 将图片切割成 size x size 的网格
    # +repage: 重置每个图块的页面信息，确保每个图块都是独立的、无偏移的图像
    cmd = [
        IMAGEMAGICK_CMD,
        tiff_path,
        '-crop', f'{tile_size}x{tile_size}',
        '+repage',
        output_path_template
    ]

    print(f"正在执行命令: {' '.join(cmd)}")

    # 4. 执行命令
    try:
        # 使用 subprocess.run 来执行命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 切片任务成功完成！")
        # 如果需要，可以打印 ImageMagick 的输出信息
        if result.stdout:
            print("输出信息:\n", result.stdout)
        return True
    except FileNotFoundError:
        print(f"错误: 无法执行 '{IMAGEMAGICK_CMD}'。请再次确认 ImageMagick 已正确安装。")
        return False
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败 (例如, 文件损坏或参数错误)
        print("✗ 切片任务执行失败。")
        print(f"返回码: {e.returncode}")
        print(f"错误信息:\n{e.stderr}")
        return False


if __name__ == '__main__':
    # --- 使用示例：批量处理整个文件夹 ---
    
    # 1. 在运行前，请先确保ImageMagick已安装
    if not check_imagemagick_command():
        exit(1) # 如果找不到命令，则退出程序

    # --- 用户配置 ---
    # 1. 指定包含所有待处理TIFF文件的源文件夹
    # 在Windows上路径可能像: r'C:\Users\YourUser\Desktop\All_Tiffs'
    SOURCE_FOLDER = '/data2/ranxiangyu/vstain'

    # 2. 指定你想要的方块大小（例如 512x512 像素）
    BLOCK_SIZE = 1024
    
    # 3. 指定输出图块的格式 (例如 'jpeg' 或 'png')
    OUTPUT_FORMAT = 'png'
    # --- 结束配置 ---


    # --- 脚本主逻辑 ---
    # 检查源文件夹是否存在
    if not os.path.isdir(SOURCE_FOLDER):
        print("-" * 60)
        print(f"错误: 配置的源文件夹不存在 -> '{SOURCE_FOLDER}'")
        print("请打开脚本文件，修改 'SOURCE_FOLDER' 变量为您正确的路径。")
        print("-" * 60)
        exit(1)

    print(f"开始扫描并处理文件夹: {SOURCE_FOLDER}")
    
    tiff_files_found = 0
    # 遍历源文件夹中的所有项目
    for filename in os.listdir(SOURCE_FOLDER):
        # 判断是否是文件，并且后缀是否是 .tif 或 .tiff (忽略大小写)
        if os.path.isfile(os.path.join(SOURCE_FOLDER, filename)) and filename.lower().endswith(('.tif', '.tiff')):
            tiff_files_found += 1
            print(f"\n{'='*20} [{tiff_files_found}] 正在处理文件: {filename} {'='*20}")
            
            # 构建当前TIFF文件的完整路径
            current_tiff_path = os.path.join(SOURCE_FOLDER, filename)
            
            # 为当前TIFF文件创建一个对应的输出文件夹
            # 例如: 对于 'image_A.tif', 创建名为 'image_A' 的文件夹
            base_name = os.path.splitext(filename)[0]
            output_folder_for_tiff = os.path.join(SOURCE_FOLDER, base_name)
            
            # 调用核心的切片函数
            # 输入是当前TIFF文件，输出是为它创建的专用文件夹
            tile_tiff(
                tiff_path=current_tiff_path, 
                output_folder=output_folder_for_tiff, 
                tile_size=BLOCK_SIZE, 
                output_format=OUTPUT_FORMAT
            )

    # 循环结束后给出总结
    if tiff_files_found == 0:
        print("\n处理完毕：在指定的源文件夹中没有找到任何 TIFF 文件。")
    else:
        print(f"\n{'='*25} 全部处理完毕 {'='*25}")
        print(f"总共处理了 {tiff_files_found} 个 TIFF 文件。")