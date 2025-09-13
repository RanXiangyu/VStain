import argparse
import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Grayscale
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import inception
import image_metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_patches import extract_patches_from_wsi, read_h5_coords
from wsi_core.WSIDataset_evaluation import VirtualStainDataset
import os
os.environ["TORCH_HOME"] = "/home/hfang/rxy/ckpt"

'''
 python evaluation/eval.py \
    --wsi_path /home/hfang/rxy/kidney_wsi/ \
    --stained_path /mnt/hfang/data/VStain/output/ \
    --h5_path /mnt/hfang/data/VStain/h5/patches/ \
    --ckpt_path /home/hfang/rxy/ckpt/inceptionv3.pth \
    --patch_size 512 --patch_level 0 \
    --batch_size 2 --num_workers 1 --content_metric lpips \
    --wsi_patch_output_path /home/hfang/rxy/kidney_patch
'''

# 定义常量和辅助类
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']


# 用于加载图像路径的数据集类
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files # 包含所有图像文件的路径
        self.transforms = transforms # 图像预处理变换

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i): # 获取单个样本
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

# 计算FID的核心公式
def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# 遍历所有输入图像，用 Inception 模型提取特征（通常是 2048 维）  输出形状：(num_images, 2048)
def get_activations(dataloader, model, device='cpu'):
    model.eval()

    N = len(dataloader.dataset)
    features_content = np.empty((N, 2048))
    features_stylized = np.empty((N, 2048))

    start_idx = 0

    pbar = tqdm(total=N)
    for batch in dataloader:
        content = batch['content'].to(device)
        stylized = batch['stylized'].to(device)
        batch_size = content.shape[0]

        with torch.no_grad():
            feat_content = model(content, return_features=True).cpu().numpy()
            feat_stylized = model(stylized, return_features=True).cpu().numpy()

        features_content[start_idx:start_idx+batch_size] = feat_content
        features_stylized[start_idx:start_idx+batch_size] = feat_stylized

        start_idx += batch_size
        pbar.update(batch_size)

    pbar.close()
    return features_content, features_stylized



# 用 get_activations 得到 (N,2048) 特征矩阵
def compute_activation_statistics(dataloader, model, device='cpu'):
    feat_content, feat_stylized = get_activations(dataloader, model, device)    
    mu_content = np.mean(feat_content, axis=0)
    sigma_content = np.cov(feat_content, rowvar=False)
    mu_stylized = np.mean(feat_stylized, axis=0)
    sigma_stylized = np.cov(feat_stylized, rowvar=False)
    return mu_content, sigma_content, mu_stylized, sigma_stylized



def compute_fid(dataloader, device, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}. Please download it first or provide the correct path.")
    
    ckpt = torch.load(ckpt_path) 

    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()


    mu_content, sigma_content, mu_stylized, sigma_stylized = \
        compute_activation_statistics(dataloader, model, device)

    fid_value = compute_frechet_distance(mu_stylized, sigma_stylized, mu_content, sigma_content)
    return fid_value

def compute_content_distance(dataloader, device, content_metric='lpips'):
    """Computes the distance for the given paths.
    Args:
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet' 选择距离计算的方法
    Returns:
        平均内容距离
    """
    metric_list = ['alexnet', 'ssim', 'ms-ssim']
    if content_metric in metric_list:
        metric = image_metrics.Metric(content_metric).to(device)
    elif content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(device)
    elif content_metric == 'vgg':
        metric = image_metrics.LPIPS_vgg().to(device)
    else:
        raise ValueError(f'Invalid content metric: {content_metric}')

    # 计算内容距离
    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(dataloader.dataset))

    for batch in dataloader:
        with torch.no_grad():
            content = batch["content"].to(device)
            stylized = batch["stylized"].to(device)

            batch_dist = metric(stylized, content)
            N += stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(stylized.shape[0])

    pbar.close()

    return dist_sum / N




"""以下计算CSFD"""

def compute_patch_simi(dataloader, device):
    metric = image_metrics.PatchSimi(device=device).to(device)
    metric.eval()

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(dataloader.dataset))

    for batch in dataloader:

        with torch.no_grad():
            content = batch["content"].to(device)
            stylized = batch["stylized"].to(device)
            batch_dist = metric(stylized, content)  # [B] tensor
            N += stylized.shape[0]
            dist_sum += torch.sum(batch_dist).item()

        pbar.update(stylized.shape[0])

    pbar.close()
    return dist_sum / N


def compute_cfsd(dataloader, device):

    print('Compute CFSD value...')

    simi_val = compute_patch_simi(dataloader, device)
    simi_dist = f'{simi_val:.4f}'
    return simi_dist



def get_opt():
    parser = argparse.ArgumentParser()
    # 文件路径设置
    parser.add_argument('--sty', default = '/data2/ranxiangyu/vstain/sty')
    parser.add_argument('--wsi_path', default = '/data2/ranxiangyu/vstain/wsi')
    parser.add_argument('--stained_path', default = '/data2/ranxiangyu/vstain/wsi',help='风格化图像路径 output目录')
    parser.add_argument('--h5_path', default = '/data2/ranxiangyu/vstain/h5/patches')
    parser.add_argument('--wsi_patch_output_path', default = '/data2/ranxiangyu/vstain/h5/patches')
    parser.add_argument('--ckpt_path', type=str, default='/home/hfang/rxy/ckpt/inceptionv3.pth', help='Path to the Art Inception model checkpoint (.pth file).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')

    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches to extract.')
    parser.add_argument('--patch_level', type=int, default=0, help='WSI level to extract patches from.')


    # parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--content_metric', type=str, default='lpips', choices=['lpips', 'vgg', 'alexnet', 'ssim', 'ms-ssim'], help='Content distance.')
    parser.add_argument("--is_wsi_patch", action="store_true", help="是否需要进行wsi的切片和保存处理")

    parser.add_argument('--stain_type', nargs='+', default=["masson", "pas","pasm"], help='List of stain types to evaluate.')

    # parser.add_argument("--without_", action='store_true')
    opt = parser.parse_args()

    return opt

def log_print(message, log_file):
    print(message)                  # 输出到控制台
    with open(log_file, "a") as f:  # 追加写入文件
        f.write(message + "\n")

def main(opt):
  
    device = "cuda" if torch.cuda.is_available() else "cpu"        
    stain_name = "_".join(opt.stain_type)
    log_file = f"{stain_name}_evaluation_results.txt"


    if opt.is_wsi_patch:
        print("Extracting patches from WSI files...")

        wsi_dir = Path(opt.wsi_path)

        h5_files = glob.glob(os.path.join(opt.h5_path, '*.h5'))
        h5_files.sort()

        for h5_file in h5_files:
            base_name = os.path.splitext(os.path.basename(h5_file))[0]
            possible_exts = [".svs", ".tif", ".tiff"]
            wsi_path = None
            for ext in possible_exts:
                candidate = os.path.join(wsi_dir, base_name + ext)
                if os.path.exists(candidate):
                    wsi_path = candidate
                    break

        # 限制调用个数
            extract_patches_from_wsi(h5_path=h5_file, wsi_path=wsi_path, output_dir=opt.wsi_patch_output_path, 
                                     patch_size=opt.patch_size, level=opt.patch_level)

    transform = ToTensor()

    dataset = VirtualStainDataset(
        content_dir=opt.wsi_patch_output_path,   #wsi切片保存路径
        stylized_root=opt.stained_path,  # 风格化图像路径
        stain_type=opt.stain_type,   # 多个类型
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    fid_value = compute_fid(dataloader, device, opt.ckpt_path)
    print("FID:", fid_value)

    content_dist = compute_content_distance(dataloader, device, content_metric=opt.content_metric)
    print(f"Content Distance ({opt.content_metric}):", content_dist.item())


    simi_dist = compute_cfsd(dataloader, device)
    print("CFSD:", simi_dist)


    log_print(f"FID: {fid_value}", log_file)

    log_print(f"Content Distance ({opt.content_metric}): {content_dist.item()}", log_file)

    log_print(f"CFSD: {simi_dist}", log_file)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)