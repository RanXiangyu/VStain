import openslide
import tifffile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import re
import os

# 作为一个示例，如果 read_h5_coords 不存在，可以先用这个代替
import h5py
def read_h5_coords(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return f['coords'][:]

class WSIDataset_evaluation(Dataset):
    def __init__(self, wsi_path, stained_path, h5_path, patch_size=512, patch_level=0, transform=None):
        """
        用于从WSI文件和坐标文件中加载Patch对的数据集（修正版）。

        Args:
            wsi_path (str): 原始SVS文件的路径 (作为内容图像)。
            stained_path (str): 虚拟染色后的TIFF文件路径 (作为风格化图像)。
            h5_path (str): H5文件的路径，包含 'coords' 数据集。
            patch_size (int): 提取的patch的边长。
            patch_level (int): 从WSI中读取的层级。
            transform (callable, optional): 应用于图像对的转换。
        """
        # --- 修正初始化逻辑 ---
        # 打开文件句柄并使用清晰的变量名
        self.wsi_handle = openslide.OpenSlide(wsi_path)
        self.tiff_handle = tifffile.TiffFile(stained_path)
        self.tiff_page = self.tiff_handle.pages[0] # 假设内容在第一页

        self.coords = read_h5_coords(h5_path)
        self.patch_size = patch_size
        self.patch_level = patch_level

        # --- 修正默认 Transform ---
        if transform is None:
            # FID 使用的 Inception 模型通常需要 ImageNet 的归一化参数
            # 如果你的 Inception 模型有特殊要求，请使用对应的参数
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)), # InceptionV3 的标准输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        x, y = int(coord[0]), int(coord[1])
        
        # 定义 patch 大小的元组
        patch_dims = (self.patch_size, self.patch_size)

        # --- 修正和清理 Patch 读取逻辑 ---
        # 1. 从SVS文件中读取内容patch (使用正确的句柄和变量名)
        content_patch_pil = self.wsi_handle.read_region(
            (x, y), 
            self.patch_level, 
            patch_dims
        ).convert('RGB')
        
        # 2. 从TIFF文件中读取风格化patch
        # stylized_patch_np = self.tiff_page.asarray(
        #     key=slice(y, y + self.patch_size), 
        #     col=slice(x, x + self.patch_size)
        # )
        # Create the slices for the y (row) and x (column) dimensions
        y_slice = slice(y, y + self.patch_size)
        x_slice = slice(x, x + self.patch_size)

        # Slice the TiffPage object directly to read only the desired region
        # stylized_patch_np = self.tiff_page.asarray(key=(y_slice, x_slice))
        stylized_patch_np = self.tiff_page.asarray()[y_slice, x_slice]
        # 确保 numpy 数组是 HWC 格式并且是 uint8 类型
        if stylized_patch_np.ndim == 2: # 如果是灰度图，扩展为3通道
            stylized_patch_np = np.stack([stylized_patch_np]*3, axis=-1)
        stylized_patch_pil = Image.fromarray(stylized_patch_np).convert('RGB')

        # 3. 应用图像变换 (使用正确的属性名 self.transform)
        content_patch = self.transform(content_patch_pil)
        stylized_patch = self.transform(stylized_patch_pil)

        return content_patch, stylized_patch, np.array([x, y])

    def close(self):
        """添加一个关闭文件句柄的方法，在评估结束后调用。"""
        self.wsi_handle.close()
        self.tiff_handle.close()
        print("WSI and TIFF file handles closed.")

def make_dict(file_list, prefix_only=False):
    """
    将文件路径转成 {id: path} 字典
    prefix_only=True 表示只取文件名前缀（处理 stained 文件的情况）
    """
    d = {}
    for f in file_list:
        fname = os.path.basename(f)
        if prefix_only:
            # 取第一个下划线前的部分
            key = fname.split("_")[0]
        else:
            key = os.path.splitext(fname)[0]
        d[key] = f
    return d


class MultiWSIDataset_evaluation(Dataset):
    def __init__(self, wsi_dir, stained_dir, h5_dir, patch_size=512, patch_level=0, transform=None):
        self.patch_size = patch_size
        self.patch_level = patch_level

        wsi_files = glob.glob(os.path.join(wsi_dir, '*.svs'))
        stained_files = glob.glob(os.path.join(stained_dir, '*.tiff'))
        h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))

        # 建立索引字典
        wsi_dict = make_dict(wsi_files)                # 22811he → xxx/22811he.svs
        h5_dict = make_dict(h5_files)                  # 22811he → xxx/22811he.h5
        # stained_dict = make_dict(stained_files, True)  # 22811he → xxx/22811he_stylized_masson_reconstructed.tiff

        # 对 stained 文件，保留原图 id + 染色类型
        stained_list = []
        for f in stained_files:
            fname = os.path.basename(f) # 取文件名
            base, _ = os.path.splitext(fname)
            # 假设格式: 22811he_stylized_masson_reconstructed
            parts = base.split('_') # 用下划线分割
            wsi_id = parts[0]
            style = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]  # masson 或 pasm
            stained_list.append((wsi_id, style, f))

        # 构造 file_pairs：每种染色作为一条记录
        self.file_pairs = []
        for wsi_id, style, stained_path in stained_list:
            if wsi_id in wsi_dict and wsi_id in h5_dict:
                self.file_pairs.append({
                    "id": wsi_id,
                    "style": style,
                    "wsi": wsi_dict[wsi_id],
                    "h5": h5_dict[wsi_id],
                    "stained": stained_path
                })

        # 找到三者共有的 id
        if not self.file_pairs:
            raise ValueError("No matching WSI, H5 and stained files found.")


        print(f"Matched {len(self.file_pairs)} WSI datasets.")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # --- 核心逻辑：预计算索引 ---
        self.wsi_handles = [None] * len(self.file_pairs)
        self.tiff_handles = [None] * len(self.file_pairs)
        
        self.patch_counts = []
        print("Scanning H5 files to build index...")
        for pair in tqdm(self.file_pairs):
            with h5py.File(pair['h5'], 'r') as f:
                self.patch_counts.append(len(f['coords']))
        
        # 计算每个WSI的patch数目的累加和，用于快速索引
        self.cumulative_patch_counts = np.cumsum([0] + self.patch_counts)
        self.total_patches = self.cumulative_patch_counts[-1]

    def __len__(self):
        return self.total_patches

    def __getitem__(self, global_idx):
        # 根据全局索引找到它属于哪个WSI
        wsi_idx = np.searchsorted(self.cumulative_patch_counts, global_idx, side='right') - 1
        
        # 计算在该WSI内部的局部索引
        local_idx = global_idx - self.cumulative_patch_counts[wsi_idx]
        
        # --- 文件句柄懒加载 (Lazy Loading) ---
        # 只有在第一次需要访问某个WSI时才打开它
        if self.wsi_handles[wsi_idx] is None:
            self.wsi_handles[wsi_idx] = openslide.OpenSlide(self.file_pairs[wsi_idx]['wsi'])
            self.tiff_handles[wsi_idx] = tifffile.TiffFile(self.file_pairs[wsi_idx]['stained'])

        wsi_handle = self.wsi_handles[wsi_idx]
        tiff_page = self.tiff_handles[wsi_idx].pages[0]
        
        # 读取坐标并提取patch (与之前的逻辑相同)
        with h5py.File(self.file_pairs[wsi_idx]['h5'], 'r') as f:
            coord = f['coords'][local_idx]

        
        x, y = int(coord[0]), int(coord[1])
        patch_dims = (self.patch_size, self.patch_size)

        # --- 读取内容 patch ---
        content_patch_pil = wsi_handle.read_region(
            (x, y), 
            self.patch_level, 
            patch_dims
        ).convert('RGB')

        # --- 读取风格化 patch ---
        stylized_patch_np = tiff_page.asarray(
            key=slice(y, y + self.patch_size), 
            col=slice(x, x + self.patch_size)
        )

        if stylized_patch_np.ndim == 2: # 灰度转三通道
            stylized_patch_np = np.stack([stylized_patch_np]*3, axis=-1)
        stylized_patch_pil = Image.fromarray(stylized_patch_np).convert('RGB')

        # --- transform ---
        content_patch = self.transform(content_patch_pil)
        stylized_patch = self.transform(stylized_patch_pil)
        
        return content_patch, stylized_patch, np.array([x, y])

    def close(self):
        # 关闭所有已打开的文件句柄
        for handle in self.wsi_handles:
            if handle:
                handle.close()
        for handle in self.tiff_handles:
            if handle:
                handle.close()
        print("\nAll opened file handles have been closed.")


import os
import glob
import re
from torch.utils.data import Dataset
from PIL import Image

def parse_content_name(filename):
    # e.g. 22811he_level0_x10049_y10371.png
    m = re.match(r"(.*)_level\d+_x(\d+)_y(\d+)\.png", filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None

def parse_stylized_name(filename, stain_type):
    # e.g. 22811he_stylized_masson_x10049_y10371.png
    pattern = rf"(.*)_stylized_{stain_type}_x(\d+)_y(\d+)\.png"
    m = re.match(pattern, filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None

class VirtualStainDataset(Dataset):
    def __init__(self, content_dir, stylized_root, stain_type="masson", transform=None):
        self.content_dir = content_dir
        self.stylized_root = stylized_root
        self.transform = transform

        # 兼容单个字符串和列表
        if isinstance(stain_type, str):
            self.stain_types = [stain_type]
        else:
            self.stain_types = stain_type

        # 1. 建立 content 索引
        content_files = glob.glob(os.path.join(content_dir, "*.png"))
        self.index_map = {}
        for f in content_files:
            parsed = parse_content_name(os.path.basename(f))
            if parsed:
                wsi, x, y = parsed
                self.index_map[(wsi, x, y)] = f

        # 2. 遍历所有 stain_type 的 stylized 文件
        self.pairs = []
        for stain in self.stain_types:
            stylized_files = glob.glob(os.path.join(stylized_root, "*", stain, "*.png"))
            for f in stylized_files:
                parsed = parse_stylized_name(os.path.basename(f), stain)
                if parsed:
                    wsi, x, y = parsed
                    if (wsi, x, y) in self.index_map:
                        self.pairs.append((self.index_map[(wsi, x, y)], f, stain))

        print(f"Built dataset with {len(self.pairs)} pairs for stains: {self.stain_types}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_content, path_stylized, stain = self.pairs[idx]

        content = Image.open(path_content).convert("RGB")
        stylized = Image.open(path_stylized).convert("RGB")

        if self.transform:
            content = self.transform(content)
            stylized = self.transform(stylized)

        return {
            "content": content,
            "stylized": stylized,
            "path_content": path_content,
            "path_stylized": path_stylized,
            "stain_type": stain
        }
