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

import utils
import inception
import image_metrics
'''
python eval_artfid.py  --cnt /data2/ranxiangyu/kidney_patch/patch_png/level1/22811he \
--sty /data2/ranxiangyu/styleid_out/style \
 --tar /data2/ranxiangyu/styleid_out/style_out/he2masson \
 --batch_size 1 --num_workers 8 \
 --content_metric lpips --mode art_fid_inf --device cuda
'''

# 定义常量和辅助类
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'
# CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/tree/main/art_inception.pth'
# https://huggingface.co/matthias-wright/art_inception/tree/main

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
    """Numpy implementation of the Frechet Distance.
    
    Args:
        mu1 (np.ndarray): Sample mean of activations of stylized images.
        mu2 (np.ndarray): Sample mean of activations of style images.
        sigma1 (np.ndarray): Covariance matrix of activations of stylized images.
        sigma2 (np.ndarray): Covariance matrix of activations of style images.
        eps (float): Epsilon for numerical stability.

    Returns:
        (float) FID value.
    """

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

# 用 get_activations 得到 (N,2048) 特征矩阵
def compute_activation_statistics(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activation statistics used by the FID.
    
    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (np.ndarray, np.ndarray): mean of activations, covariance of activations
        
    """
    act = get_activations(files, model, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers=1):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # ckpt_file = utils.download(CKPT_URL)
    # 修改这里，使用本地文件路径
    ckpt_file = '/data2/ranxiangyu/checkpoints/art_inception.pth'
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    mu1, sigma1 = compute_activation_statistics(stylized_image_paths, model, batch_size, device, num_workers)
    mu2, sigma2 = compute_activation_statistics(style_image_paths, model, batch_size, device, num_workers)
    
    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_points=15, num_workers=1):
    """Computes the FID infinity for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        num_points (int): Number of FID_N we evaluate to fit a line. 用于拟合线性回归的 FID 点数量

    Returns:
        (float) FID infinity value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # 加载inception模型
    # ckpt_file = utils.download(CKPT_URL)
    # 修改为本地ckpt文件代码
    ckpt_file = '/data2/ranxiangyu/checkpoints/art_inception.pth'
    ckpt = torch.load(ckpt_file, map_location=device)
    
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
    # 创建一个从 5000 到 len(stylized_image_paths) 之间的等间距整数数组，用于不同批量大小的 FID 计算
    # 这里的5000是为了避免计算时内存不足
    # 生成num个在start和stylized_image_paths之间均匀分布的数值，astype('int32')将其转换为整数
    # linespace等间距数值
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


def compute_art_fid(path_to_stylized, path_to_style, path_to_content, batch_size, device, mode='art_fid_inf', content_metric='lpips', num_workers=1):
    print('Compute FID value...')
    if mode == 'art_fid_inf':
        fid_value = compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'art_fid':
        fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'style_loss':
        fid_value = compute_style_loss(path_to_stylized, path_to_style, batch_size, device, num_workers)
    else:
        fid_value = compute_gram_loss(path_to_stylized, path_to_style, batch_size, device, num_workers)
    
    print('Compute content distance...')
    cnt_value = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers)
    gray_cnt_value = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers, gray=True)

    art_fid_value = (cnt_value + 1) * (fid_value + 1)
    # fid_value = f'{fid_value.item():.4f}'
    # cnt_value = f'{content_dist.item():.4f}'
    # gray_cnt_value = f'{gray_content_dist.item():.4f}'
    # art_fid_value = (cnt_value + 1) * (fid_value + 1)
    return art_fid_value.item(), fid_value.item(), cnt_value.item(), gray_cnt_value.item(), 


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
    """Computes CFSD for the given paths.
    Args:
        batch_size (int): Batch size for computing activations.
        num_workers (int): Number of threads for data loading. cpu上完成，后台加载数据使用的cpu进程数量，备菜
    Returns:
        (float) CFSD value.
    """
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
# 创建一个新函数，用于调整样式图像路径列表以匹配目标图像数量
def adjust_paths(stylized_paths, input_paths):
    """
    调整样式图像路径列表，使其长度与风格化图像数量相匹配。
    通过重复引用同一组样式图像路径，而不是复制实际文件。
    
    Args:
        stylized_paths (list): 风格化图像路径列表
        input_paths (list): 样式图像路径列表
        
    Returns:
        list: 长度与 stylized_paths 匹配的样式图像路径列表
    """
    if len(stylized_paths) == len(input_paths):
        return input_paths
    
    # 计算需要重复的倍数
    ratio = len(stylized_paths) // len(input_paths)
    remainder = len(stylized_paths) % len(input_paths)
    
    # 重复样式图像路径以匹配风格化图像数量
    adjusted_input_paths = input_paths * ratio
    
    # 处理余数
    if remainder > 0:
        adjusted_input_paths.extend(input_paths[:remainder])
    
    print(f"调整样式图像路径: 原始 {len(input_paths)} 张图像扩展为 {len(adjusted_input_paths)} 张路径")
    return adjusted_input_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--content_metric', type=str, default='lpips', choices=['lpips', 'vgg', 'alexnet', 'ssim', 'ms-ssim'], help='Content distance.')
    parser.add_argument('--mode', type=str, default='art_fid_inf', choices=['art_fid', 'art_fid_inf'], help='Evaluate ArtFID or ArtFID_infinity.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--sty', type=str, required=True, help='Path to style images.')
    parser.add_argument('--cnt', type=str, required=True, help='Path to content images.')
    parser.add_argument('--tar', type=str, required=True, help='Path to stylized images.')
    args = parser.parse_args()
    
     # 获取图像路径
    stylized_paths = get_image_paths(args.tar, sort=True)
    style_paths = get_image_paths(args.sty, sort=True)
    content_paths = get_image_paths(args.cnt, sort=True)

    ''' 以下是为了确保风格化图像和样式图像数量一致 '''
    # 调整样式图像路径以匹配风格化图像数量
    adjusted_style_paths = adjust_paths(stylized_paths, style_paths)
    adjusted_content_paths = adjust_paths(stylized_paths, content_paths)
    
    # 创建临时目录来存储调整后的样式路径
    temp_style_dir = os.path.join(os.path.dirname(args.tar), "temp_style")
    os.makedirs(temp_style_dir, exist_ok=True)

    # 创建临时目录来存储调整后的内容路径
    temp_content_dir = os.path.join(os.path.dirname(args.tar), "temp_content")
    os.makedirs(temp_content_dir, exist_ok=True)

     # 为每个调整后的样式路径创建符号链接
    for i, path in enumerate(adjusted_style_paths):
        # 创建符号链接，指向原始风格图像
        link_path = os.path.join(temp_style_dir, f"style_{i:06d}.png")
        if os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.abspath(path), link_path)
     
    # 为每个调整后的内容路径创建符号链接
    for i, path in enumerate(adjusted_content_paths):
        link_path = os.path.join(temp_content_dir, f"content_{i:06d}.png")
        if os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.abspath(path), link_path)

    
    artfid, fid, lpips, lpips_gray = compute_art_fid(args.tar,
                                                    temp_style_dir, # 使用调整后的样式路径
                                                    temp_content_dir,  # 使用调整后的内容路径
                                                    args.batch_size,
                                                    args.device,
                                                    args.mode,
                                                    args.content_metric,
                                                    args.num_workers)

    cfsd = compute_cfsd(args.tar,
                        args.cnt,
                        args.batch_size,
                        args.device,
                        args.num_workers)

    print('ArtFID:', artfid, 'FID:', fid, 'LPIPS:', lpips, 'LPIPS_gray:', lpips_gray)
    print('CFSD:', cfsd)

if __name__ == '__main__':
    main()
