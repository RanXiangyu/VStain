import argparse
import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
import torch
from sklearn.linear_model import LinearRegression
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Grayscale
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

import utils
import inception
import image_metrics

# 将当前文件的上级目录（也就是项目根目录）添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsi_core.WSIDataset_evaluation import WSIDataset_evaluation

"""
CUDA_VISIBLE_DEVICES=1 python evaluation/evaluation.py \
    --wsi_path /home/hfang/rxy/kidney_wsi/22811he.svs \
    --stained_path /mnt/hfang/data/VStain/output/22811he/22811he_stylized_masson_reconstructed.tiff \
    --h5_path /mnt/hfang/data/VStain/h5/patches/22811he.h5 \
    --ckpt_path /home/hfang/rxy/ckpt/inceptionv3.pth \
    --patch_size 512 --patch_level 0 \
    --batch_size 1 --num_workers 0 --content_metric lpips
"""

Image.MAX_IMAGE_PIXELS = None
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']

    
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

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# WSI 版本的特征提取函数
def get_activations_wsi(wsi_dataloader, model, image_type, device='cuda'):
    model.eval()
    num_images = len(wsi_dataloader.dataset)
    pred_arr = np.empty((num_images, 2048))
    start_idx = 0
    pbar = tqdm(total=num_images, desc=f"Getting activations for {image_type} images")
    for batch_content, batch_stylized, _ in wsi_dataloader:
        batch = batch_content if image_type == 'content' else batch_stylized
        batch = batch.to(device)
        with torch.no_grad():
            features = model(batch, return_features=True)
        pred_arr[start_idx:start_idx + features.shape[0]] = features.cpu().numpy()
        start_idx += features.shape[0]
        pbar.update(batch.shape[0])
    pbar.close()
    return pred_arr

# WSI 版本的统计量计算
def compute_activation_statistics_wsi(wsi_dataloader, model, image_type, device):
    act = get_activations_wsi(wsi_dataloader, model, image_type, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def compute_fid_wsi(wsi_dataloader, device, ckpt_path):
    
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}. Please download it first or provide the correct path.")
    
    print(f"--- Loading Inception model from: {ckpt_path} ---")

    # device_obj = torch.device(device)
    ckpt = torch.load(ckpt_path) 
    # 在这里删除了删除map_location，不给出 map_location参数的时候自动根据 ckpt 文件加载

    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    mu1, sigma1 = compute_activation_statistics_wsi(wsi_dataloader, model, 'stylized', device)
    mu2, sigma2 = compute_activation_statistics_wsi(wsi_dataloader, model, 'content', device)
    
    return compute_frechet_distance(mu1, sigma1, mu2, sigma2)

# WSI 版本的内容距离计算
def compute_content_distance_wsi(wsi_dataloader, content_metric, device):
    device_obj = torch.device(device)
    if content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(device_obj)
    else: # 添加对其他度量的支持
        raise ValueError(f'Invalid content metric: {content_metric}')

    dist_sum = 0.0
    pbar = tqdm(total=len(wsi_dataloader.dataset), desc=f"Computing {content_metric}")
    for batch_content, batch_stylized, _ in wsi_dataloader:
        batch_content = batch_content.to(device_obj)
        batch_stylized = batch_stylized.to(device_obj)
        with torch.no_grad():
            batch_dist = metric(batch_stylized, batch_content)
        dist_sum += torch.sum(batch_dist).item()
        pbar.update(batch_stylized.shape[0])
    pbar.close()
    return dist_sum / len(wsi_dataloader.dataset)


def get_opt():
    parser = argparse.ArgumentParser()
    # 文件路径设置
    parser.add_argument('--sty', default = '/data2/ranxiangyu/vstain/sty')
    parser.add_argument('--wsi_path', default = '/data2/ranxiangyu/vstain/wsi')
    parser.add_argument('--stained_path', default = '/data2/ranxiangyu/vstain/wsi')
    parser.add_argument('--h5_path', default = '/data2/ranxiangyu/vstain/h5/patches')
    parser.add_argument('--ckpt_path', type=str, default='/home/hfang/rxy/ckpt/inceptionv3.pth', help='Path to the Art Inception model checkpoint (.pth file).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')

    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patches to extract.')
    parser.add_argument('--patch_level', type=int, default=0, help='WSI level to extract patches from.')


    # parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--content_metric', type=str, default='lpips', choices=['lpips', 'vgg', 'alexnet', 'ssim', 'ms-ssim'], help='Content distance.')

    # parser.add_argument("--without_", action='store_true')
    opt = parser.parse_args()

    return opt
def main(opt):
    print("--- Initializing WSI Dataset ---")
    dataset = WSIDataset_evaluation(
        wsi_path=opt.wsi_path,
        stained_path=opt.stained_path,
        h5_path=opt.h5_path,
        patch_size=opt.patch_size,
        patch_level=opt.patch_level,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    
 

    try:
        print("\n--- Computing FID ---")
        artfid_value = compute_fid_wsi(wsi_dataloader = dataloader, device = opt.device, ckpt_path = opt.ckpt_path)

        print(f"\n--- Computing Content Distance ({opt.content_metric}) ---")
        content_dist_value = compute_content_distance_wsi(dataloader, opt.content_metric, opt.device)

        print("\n\n--- Evaluation Results ---")
        print(f'ArtFID: {artfid_value:.4f}')
        print(f'{opt.content_metric.upper()}: {content_dist_value:.4f}')

    finally:
        dataset.close()

if __name__ == '__main__':
    opt = get_opt()
    main(opt)