import openslide
import tifffile
from torch.utils.data import Dataset
# 假设您有一个 utils.hdf5.read_h5_coords 函数
# from utils.hdf5 import read_h5_coords 
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

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
        stylized_patch_np = self.tiff_page.asarray(
            key=slice(y, y + self.patch_size), 
            col=slice(x, x + self.patch_size)
        )
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



class MultiWSIDataset_evaluation(Dataset):
    def __init__(self, file_pairs, patch_size=512, patch_level=0, transform=None):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.transform = transform # ... (与之前相同的 transform 逻辑) ...

        # --- 核心逻辑：预计算索引 ---
        self.wsi_handles = [None] * len(file_pairs)
        self.tiff_handles = [None] * len(file_pairs)
        
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
        # np.searchsorted 效率很高
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
        # ... (后续的 patch 读取和 transform 逻辑与原 Dataset 完全相同) ...
        # ... 返回 content_patch, stylized_patch, np.array([x, y]) ...
        return #... (省略，与原版相同)

    def close(self):
        # 关闭所有已打开的文件句柄
        for handle in self.wsi_handles:
            if handle:
                handle.close()
        for handle in self.tiff_handles:
            if handle:
                handle.close()
        print("\nAll opened file handles have been closed.")