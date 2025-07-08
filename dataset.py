import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class WSIDataset(Dataset):
    def __init__(self, h5_file, dataset_name, transform=None):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.transform = transform

        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f[self.dataset_name])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            image = f[self.dataset_name][idx]
        
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image

# 定义图像转换
transform = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.Resize((512, 512), transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建Dataset和DataLoader
h5_file = 'path/to/your/h5file.h5'
dataset_name = 'kidney'
dataset = WSIDataset(h5_file, dataset_name, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
# /data2/ranxiangyu/kidney_patch