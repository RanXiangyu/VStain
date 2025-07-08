import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
import time

# -------------------
# 1. VGG19 编码器（仅使用前4层）
# -------------------
class VGGEncoder(nn.Module):
    def __init__(self, weights_path=None):
        super(VGGEncoder, self).__init__()
        # 创建未预训练的模型
        vgg = models.vgg19(pretrained=False).features
        
        # 如果提供了权重路径，加载本地权重
        if weights_path is not None and os.path.exists(weights_path):
            state_dict = torch.load(weights_path)
            # 如果权重是完整模型的权重（不只是features部分）
            if 'features.0.weight' in state_dict:
                vgg_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('features.'):
                        vgg_state_dict[key.replace('features.', '')] = value
                vgg.load_state_dict(vgg_state_dict)
            else:
                # 如果权重已经只是features部分
                vgg.load_state_dict(state_dict)
        else:
            # 如果未提供权重路径，则从网络下载
            vgg = models.vgg19(pretrained=True).features
            
        self.enc = nn.Sequential(*list(vgg.children())[:21])  # 到 relu4_1
        for param in self.enc.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.enc(x)


# -------------------
# 2. AdaIN 模块
# -------------------
def adain(content_feat, style_feat, eps=1e-5):
    # 计算每个通道的均值和标准差
    c_mean, c_std = mean_std(content_feat)
    s_mean, s_std = mean_std(style_feat)

    norm = (content_feat - c_mean) / (c_std + eps)
    return norm * s_std + s_mean


def mean_std(feat, eps=1e-5):
    # 输入为 BxCxHxW，按通道计算均值和方差
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2, keepdim=True) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2, keepdim=True).view(N, C, 1, 1)
    return feat_mean, feat_std


# -------------------
# 3. Decoder 模块（可自己拓展更复杂结构）
# -------------------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


# -------------------
# 4. 完整模型封装
# -------------------
class AdaINStyleTransfer(nn.Module):
    def __init__(self, vgg_weights_path=None):
        super(AdaINStyleTransfer, self).__init__()
        self.encoder = VGGEncoder(weights_path=vgg_weights_path)
        self.decoder = Decoder()

    def forward(self, content, style, alpha=1.0):
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        t = adain(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat  # 可调风格强度
        generated = self.decoder(t)
        return generated


# 加载图片
def load_image(path, size=512):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def process_directory_pair(content_dir, style_dir, output_dir, alpha=1.0, vgg_weights_path=None):
    """处理两个目录中的所有图像对"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取内容和风格图像列表
    content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 加载模型
    model = AdaINStyleTransfer(vgg_weights_path=vgg_weights_path).cuda().eval()
    
    # 对每个内容图像和风格图像组合进行处理
    for content_file in tqdm(content_images, desc="处理内容图像"):
        content_path = os.path.join(content_dir, content_file)
        content = load_image(content_path).cuda()
        
        for style_file in style_images:
            style_path = os.path.join(style_dir, style_file)
            style = load_image(style_path).cuda()
            
            # 生成输出文件名
            output_filename = f"{os.path.splitext(content_file)[0]}_style_{os.path.splitext(style_file)[0]}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 风格迁移处理
            with torch.no_grad():
                output = model(content, style, alpha=alpha)
            
            
            save_image(output.clamp(0, 1), output_path)
            
            # 打印一些调试信息以检查输出
            print(f"输出图像的范围：最小值={output.min().item():.4f}，最大值={output.max().item():.4f}")

def main():
    """主函数，解析命令行参数并启动处理"""
    parser = argparse.ArgumentParser(description='AdaIN风格迁移')
    parser.add_argument('--content_dir', type=str, required=True, help='内容图像文件夹路径')
    parser.add_argument('--style_dir', type=str, required=True, help='风格图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='results', help='输出文件夹')
    parser.add_argument('--alpha', type=float, default=1.0, help='风格强度 (0-1)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--vgg_weights', type=str, default=None, help='VGG19预训练权重路径')

    args = parser.parse_args()
    
    # 设置GPU
    torch.cuda.set_device(args.gpu)
    
    # 开始处理
    start_time = time.time()
    process_directory_pair(args.content_dir, args.style_dir, args.output_dir, args.alpha, args.vgg_weights)
    print(f"处理完成，耗时: {time.time() - start_time:.2f}秒")

if __name__ == "__main__":
    from torchvision.utils import save_image
    main()