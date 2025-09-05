import openslide
import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import pathlib 

class WSIDataset(Dataset):
    def __init__(self, wsi_paths, h5_paths, patch_size=512, level=0, transform=None):
        self.patch_size = patch_size
        self.level = level
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])# 需要进行逆归一化

        # 自动匹配wsi文件和h5文件  1. 创建一个从 H5 文件主名到其完整路径的映射，方便快速查找
        wsi_map = {pathlib.Path(p).stem: p for p in wsi_paths}
        # wsi_map = {
        #     "12345HE": "/data/wsis/12345HE.svs",
        #     "67890HE": "/data/wsis/67890HE.svs"
        # }
        
        self.wsi_paths = []
        self.h5_paths = []

        # 2. 遍历 H5 路径列表，寻找匹配的 WSI 文件
        for h5_path in h5_paths:
            h5_stem = pathlib.Path(h5_path).stem # 获取 H5 文件的主名
            if h5_stem in wsi_map:
                # 如果找到了同名的 WSI 文件，则将这对文件路径保存下来
                # 注意：要保持 self.wsi_paths 和 self.h5_paths 顺序和内容的一致性
                self.wsi_paths.append(wsi_map[h5_stem])
                self.h5_paths.append(h5_path)

        if not self.wsi_paths:
            raise ValueError("未找到任何匹配的 WSI 和 H5 文件对，请检查文件主名是否一致")
        
        # 打开并保留所有已配对 WSI 文件的句柄
        self.wsi_handles = [openslide.OpenSlide(p) for p in self.wsi_paths]

        # 创建索引地图
        self.index_map = []
        print("正在为匹配的文件创建索引地图...")
        for wsi_idx, h5_path in enumerate(tqdm(self.h5_paths)):
            with h5py.File(h5_path, 'r') as hf:
                coords = hf['coords'][:]
                for coord in coords:
                    self.index_map.append((wsi_idx, int(coord[0]), int(coord[1])))
        
        # 定义图像转换流程
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def __len__(self):
        """返回所有 WSI 中切片的总数。"""
        return len(self.index_map)


    def __getitem__(self, idx):
        """
        从正确的 WSI 文件中读取并返回单个切片及其坐标。
        """
        wsi_idx, x, y = self.index_map[idx]
        wsi_handle = self.wsi_handles[wsi_idx]
        tile_pil = wsi_handle.read_region(
            (x, y), 
            self.level, 
            (self.patch_size, self.patch_size)
        )
        tile_rgb = tile_pil.convert("RGB")
        tile_tensor = self.transform(tile_rgb)
        return tile_tensor, np.array([x, y])

    def close(self):
        """关闭所有已打开的 WSI 文件句柄。"""
        print("正在关闭所有 WSI 文件句柄...")
        for handle in self.wsi_handles:
            handle.close()