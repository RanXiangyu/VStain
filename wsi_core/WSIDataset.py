import openslide
import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset


class WSIDataset(Dataset):
    """
    单个 WSI + H5 对应的数据集
    """
    def __init__(self, wsi_path, h5_path, patch_size=512, level=0, transform=None):
        self.patch_size = patch_size
        self.level = level
        self.wsi_path = wsi_path
        self.h5_path = h5_path

        # 打开 WSI 句柄
        self.wsi_handle = openslide.OpenSlide(wsi_path)

        # 读取 patch 坐标
        with h5py.File(h5_path, 'r') as hf:
            self.coords = hf['coords'][:]

        # 定义图像转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        tile_pil = self.wsi_handle.read_region(
            (int(x), int(y)), 
            self.level, 
            (self.patch_size, self.patch_size)
        )
        tile_rgb = tile_pil.convert("RGB")
        tile_tensor = self.transform(tile_rgb)
        return tile_tensor, 

    def close(self):
        """关闭当前 WSI 句柄"""
        self.wsi_handle.close()


class MultiWSIDataset(ConcatDataset):
    """
    多个 WSI Dataset 的拼接（适合训练）
    """
    def __init__(self, wsi_paths, h5_paths, patch_size=512, level=0, transform=None):
        if len(wsi_paths) != len(h5_paths):
            raise ValueError("wsi_paths 与 h5_paths 长度不一致！")

        datasets = [
            WSIDataset(wsi, h5, patch_size=patch_size, level=level, transform=transform)
            for wsi, h5 in zip(wsi_paths, h5_paths)
        ]
        super().__init__(datasets)
        self.datasets = datasets

    def close(self):
        """关闭所有子 Dataset 的句柄"""
        for d in self.datasets:
            d.close()


""" 
此dataset会将h5文件全部排列
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
        return len(self.index_map)


    def __getitem__(self, idx):
        wsi_idx, x, y = self.index_map[idx]
        wsi_handle = self.wsi_handles[wsi_idx]
        tile_pil = wsi_handle.read_region(
            (x, y), 
            self.level, 
            (self.patch_size, self.patch_size)
        )
        tile_rgb = tile_pil.convert("RGB")
        tile_tensor = self.transform(tile_rgb)
        return tile_tensor, np.array([x, y]) # 返回图像和坐标

    def get_wsi_info(self, idx=None):
        if idx is not None:
            if idx < 0 or idx >= len(self.index_map):
                raise IndexError(f"索引 {idx} 超出范围 (0 ~ {len(self.index_map)-1})")
            wsi_idx, _, _ = self.index_map[idx]
            return wsi_idx
        else:
            return len(set([w[0] for w in self.index_map]))

    def close(self):
        print("正在关闭所有 WSI 文件句柄...")
        for handle in self.wsi_handles:
            handle.close()

"""