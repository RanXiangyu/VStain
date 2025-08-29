# 打印h5文件的全部信息到txt文件
import h5py
import numpy as np
import sys

def print_h5_structure(name, obj, txt_file):
    """
    递归地打印HDF5文件结构和内容的函数。
    这是被h5py的visititems方法调用的回调函数。
    """
    # 根据对象类型（组或数据集）写入不同信息
    if isinstance(obj, h5py.Group):
        txt_file.write(f"组 (Group): {name}\n")
        txt_file.write("-" * (len(name) + 12) + "\n")
    elif isinstance(obj, h5py.Dataset):
        txt_file.write(f"  数据集 (Dataset): {name}\n")
        txt_file.write(f"    - 形状 (Shape): {obj.shape}\n")
        txt_file.write(f"    - 数据类型 (Dtype): {obj.dtype}\n")
        
        # 读取并写入数据集的内容
        try:
            data = obj[...] # 读取数据集所有数据
            txt_file.write(f"    - 数据内容 (Data):\n")
            # 使用Numpy的tostring方法进行格式化输出，避免数据过长时出现省略号
            txt_file.write(np.array2string(data, threshold=sys.maxsize, max_line_width=np.inf))
            txt_file.write("\n")
        except TypeError as e:
            txt_file.write(f"    - 无法读取数据内容 (可能是字符串等特殊类型): {e}\n")
            # 对于无法直接用numpy array2string处理的数据类型（例如某些字符串）
            # 尝试直接读取
            try:
                data = obj[()]
                txt_file.write(f"    - 数据内容 (Data):\n{data}\n")
            except Exception as read_e:
                txt_file.write(f"    - 尝试直接读取失败: {read_e}\n")


    # 检查并写入对象的属性（元数据）
    if obj.attrs:
        txt_file.write(f"    - 属性 (Attributes):\n")
        for key, value in obj.attrs.items():
            txt_file.write(f"      - {key}: {value}\n")
    
    txt_file.write("\n") # 每个对象后加一个空行，方便阅读


def read_h5_to_txt(h5_file_path, txt_file_path):
    """
    读取H5文件的全部信息并保存到TXT文件中。

    参数:
    h5_file_path (str): 输入的H5文件路径。
    txt_file_path (str): 输出的TXT文件路径。
    """
    try:
        # 以写入模式打开TXT文件
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"H5文件 '{h5_file_path}' 内容分析报告\n")
            txt_file.write("=" * 40 + "\n\n")
            
            # 以只读模式打开H5文件
            with h5py.File(h5_file_path, 'r') as h5_file:
                # 使用visititems方法遍历文件中的所有对象（组和数据集）
                # 它会自动进行递归遍历
                h5_file.visititems(lambda name, obj: print_h5_structure(name, obj, txt_file))
        
        print(f"成功！已将 '{h5_file_path}' 的内容写入到 '{txt_file_path}'")

    except FileNotFoundError:
        print(f"错误：无法找到H5文件 '{h5_file_path}'")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # --- 读取H5文件并写入TXT ---
    # 定义你的H5文件路径和希望输出的TXT文件路径
    input_h5_file = '/data2/ranxiangyu/kidney_patch/kidney_patch_512/level0/patches/22811he.h5'  # <--- 请将这里替换成你的H5文件路径
    output_txt_file = 'level0.txt' # <--- 这是输出的TXT文件名

    read_h5_to_txt(input_h5_file, output_txt_file)