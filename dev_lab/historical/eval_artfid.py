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
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import utils
import inception
import image_metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_patches import extract_patches_from_wsi, read_h5_coords
# from wsi_core.WSIDataset_evaluation import VirtualStainDataset
from wsi_core.WSIDataset_evaluation import VirtualStainDataset


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


# 遍历所有输入图像，用 Inception 模型提取特征（通常是 2048 维）  输出形状：(num_images, 2048)
def get_activations(files, model, batch_size=50, device='cpu', num_workers=1):
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=Compose([Resize(512),ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), 2048))

    start_idx = 0

    pbar = tqdm(total=len(files))
    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            features = model(batch, return_features=True)

        features = features.cpu().numpy()
        pred_arr[start_idx:start_idx + features.shape[0]] = features
        start_idx = start_idx + features.shape[0]

        pbar.update(batch.shape[0])

    pbar.close()
    return pred_arr


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

# 用 get_activations 得到 (N,2048) 特征矩阵
def compute_activation_statistics(files, model, batch_size=50, device='cpu', num_workers=1):
    act = get_activations(files, model, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def compute_fid(path_to_stylized, path_to_style, batch_size, device, ckpt_path, num_workers=1):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}. Please download it first or provide the correct path.")
    
    ckpt = torch.load(ckpt_path) 


    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    mu1, sigma1 = compute_activation_statistics(stylized_image_paths, model, batch_size, device, num_workers)
    mu2, sigma2 = compute_activation_statistics(style_image_paths, model, batch_size, device, num_workers)
    
    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, ckpt_path, num_points=15, num_workers=1):
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}. Please download it first or provide the correct path.")
    
    ckpt = torch.load(ckpt_path) 
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    # 断言 为了确保风格化图像和风格图像数量一致
    assert len(stylized_image_paths) == len(style_image_paths), \
           f'Number of stylized images and number of style images must be equal.({len(stylized_image_paths)},{len(style_image_paths)})'

    # 用inception模型提取特征
    activations_stylized = get_activations(stylized_image_paths, model, batch_size, device, num_workers)
    activations_style = get_activations(style_image_paths, model, batch_size, device, num_workers)
    # 图像的索引数组，用于随机采样
    activation_idcs = np.arange(activations_stylized.shape[0])

    fids = []
    fid_batches = np.linspace(start=5000, stop=len(stylized_image_paths), num=num_points).astype('int32')
    
    for fid_batch_size in fid_batches:
        # 打乱 随机采样
        np.random.shuffle(activation_idcs)
        idcs = activation_idcs[:fid_batch_size]
        
        act_style_batch = activations_style[idcs]
        act_stylized_batch = activations_stylized[idcs]

        mu_style, sigma_style = np.mean(act_style_batch, axis=0), np.cov(act_style_batch, rowvar=False)
        mu_stylized, sigma_stylized = np.mean(act_stylized_batch, axis=0), np.cov(act_stylized_batch, rowvar=False)
        
        fid_value = compute_frechet_distance(mu_style, sigma_style, mu_stylized, sigma_stylized)
        fids.append(fid_value)

    fids = np.array(fids).reshape(-1, 1)
    reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    print(f"不同 batch size 下的 FID: {fids.flatten()}")
    print(f"预测 batch_size → ∞ 时的 FID: {fid_infinity:.4f}")

    return fid_infinity


def compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric='lpips', device='cuda', num_workers=1, gray=False):
    """Computes the distance for the given paths.
    Args:
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet' 选择距离计算的方法
    Returns:
        平均内容距离
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # 获取图像路径并排序
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'
    
    # 图像预处理 transformers
    if gray:
        content_transforms = Compose([Resize(512), Grayscale(),
        ToTensor()])
    else:
        content_transforms = Compose([Resize(512),
        ToTensor()])
    
    # 创建 Dataset 和 DataLoader
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=content_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=content_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
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
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N




"""以下计算CSFD"""

def compute_patch_simi(path_to_stylized, path_to_content, batch_size, device, num_workers=1):
    """Computes the distance for the given paths.
    Args:
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
    Returns:
        (float) 成对图像的平均 patch similarity 距离
    """
    # device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # 根据路径获取图像路径并排序以匹配样式化图像与对应的内容图像
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    # 确保两者相等
    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    style_transforms = ToTensor() # 定义图像转换方法
    
    # 创建数据集和数据加载器
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=style_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=style_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    # 初始化用于计算距离的度量类
    metric = image_metrics.PatchSimi(device=device).to(device)

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    # 遍历样式化图像和内容图像的批次
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device)) 
            #得到一个逐图像的距离张量，累加和平均
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N


def compute_cfsd(path_to_stylized, path_to_content, batch_size, device, num_workers=1):
    print('Compute CFSD value...')

    # 计算 Patch Similarity，该函数返回样式化图像和内容图像的距离值
    simi_val = compute_patch_simi(path_to_stylized, path_to_content, 1, device, num_workers)
    simi_dist = f'{simi_val.item():.4f}'# 将距离值保留四位小数
    return simi_dist


# 用于获取指定目录中的图像路径
def get_image_paths(path, sort=False):
    """Returns the paths of the images in the specified directory, filtered by allowed file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        (list): List of image paths with allowed file extensions.

    """
    paths = []
    # 使用 glob 查找所有符合扩展名的文件
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{extension}')))
    if sort:
        paths.sort()
    return paths




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
    parser.add_argument('--mode', type=str, default='stain_dataloader', choices=['stain_dataloader', 'normal'], help='Evaluate ArtFID or ArtFID_infinity.')
    parser.add_argument("--is_wsi_patch", action="store_true", help="是否需要进行wsi的切片和保存处理")

    # parser.add_argument("--without_", action='store_true')
    opt = parser.parse_args()

    return opt

def main(opt):
  
    device = "cuda" if torch.cuda.is_available() else "cpu"        
    


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
        stain_type=["masson", "pas", "pasm"],   # 多个类型
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    simi_dist = compute_cfsd(dataloader, device)
    print("CFSD:", simi_dist)


    # cfsd = compute_cfsd(opt.tar,
    #                     opt.cnt,
    #                     opt.batch_size,
    #                     opt.device,
    #                     opt.num_workers)

    # print('CFSD:', cfsd)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)