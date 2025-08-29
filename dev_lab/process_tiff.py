import tifffile
import numpy as np
import os
from tqdm import tqdm

def crop_bigtiff(input_path, output_path, x, y, width, height):
    """
    从一个BigTIFF文件中裁剪指定区域并保存为新文件。

    该函数设计为内存高效型，仅从磁盘读取所需区域。

    Args:
        input_path (str): 输入的BigTIFF文件路径。
        output_path (str): 裁剪后文件的保存路径。
        x (int): 裁剪区域左上角的X坐标（水平方向）。
        y (int): 裁剪区域左上角的Y坐标（垂直方向）。
        width (int): 裁剪区域的宽度。
        height (int): 裁剪区域的高度。
    """
    try:
        # 使用 tifffile.TiffFile 以只读模式打开文件，这不会立即加载图像数据
        with tifffile.TiffFile(input_path) as tif:
            # 获取第一页（通常是主图像）
            # TiffFile.asarray 方法支持通过 key 参数直接从文件读取一个切片
            # 这样可以避免将整个图像加载到内存中
            page = tif.pages[0]
            
            # 获取原图尺寸 (height, width, channels) or (height, width)
            original_shape = page.shape
            original_height, original_width = original_shape[0], original_shape[1]
            
            print(f"原图尺寸 (高x宽): {original_height} x {original_width}")

            # --- 边界检查 ---
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                print("错误：坐标和尺寸必须是正数。")
                return
            if x + width > original_width or y + height > original_height:
                print("错误：裁剪区域超出了原图边界。")
                print(f"请求的区域 X: [{x}, {x+width-1}], Y: [{y}, {y+height-1}]")
                print(f"允许的最大范围 X: [0, {original_width-1}], Y: [0, {original_height-1}]")
                return

            # 使用 tqdm 显示处理进度
            with tqdm(total=1, desc=f"正在裁剪 {os.path.basename(input_path)}") as pbar:
                
                # 定义要读取的区域 (切片)
                # numpy的切片格式是 [y_start:y_end, x_start:x_end]
                crop_slice = (slice(y, y + height), slice(x, x + width))

                # 使用 key 参数直接从磁盘读取指定区域
                # 这是处理大文件最有效率的方式
                cropped_array = tif.asarray(key=crop_slice)
                
                pbar.update(1)

        # 检查输出目录是否存在，如果不存在则创建
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 使用 tifffile.imwrite 保存裁剪后的numpy数组
        # bigtiff=True 确保如果裁剪结果仍然很大，可以正确保存
        with tqdm(total=1, desc=f"正在保存到 {os.path.basename(output_path)}") as pbar:
            tifffile.imwrite(output_path, cropped_array, bigtiff=True)
            pbar.update(1)
            
        print(f"文件裁剪成功！已保存至: {output_path}")
        print(f"裁剪后尺寸 (高x宽): {cropped_array.shape[0]} x {cropped_array.shape[1]}")

    except FileNotFoundError:
        print(f"错误：输入文件未找到 -> {input_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 参数设置 ---
    # 1. 指定你的BigTIFF文件路径
    # 注意：请将 'path/to/your/large_image.tif' 替换为你的实际文件路径
    # 在Windows上路径可能像这样：r'C:\Users\YourUser\Desktop\large_image.tif'
    input_file = 'large_image.tif' 
    
    # 为了能运行这个示例，我们先创建一个虚拟的大文件
    if not os.path.exists(input_file):
        print(f"未找到示例文件 '{input_file}'，正在创建一个10000x10000的虚拟文件...")
        # 创建一个 10000x10000 像素的虚拟 BigTIFF 文件
        dummy_data = np.zeros((10000, 10000), dtype=np.uint8)
        dummy_data[4000:6000, 4000:6000] = 255 # 在中间画一个白色方块
        tifffile.imwrite(input_file, dummy_data, bigtiff=True)
        print("虚拟文件创建成功。")

    # 2. 指定裁剪后文件的保存路径
    output_file = 'cropped_image.tif'

    # 3. 定义你想要裁剪的区域
    # 从哪个点开始裁剪 (左上角坐标)
    top_left_x = 4500
    top_left_y = 4500

    # 你需要裁剪出的宽度和高度
    crop_width = 1000
    crop_height = 1000

    # --- 调用函数执行裁剪 ---
    crop_bigtiff(
        input_path=input_file,
        output_path=output_file,
        x=top_left_x,
        y=top_left_y,
        width=crop_width,
        height=crop_height
    )

    