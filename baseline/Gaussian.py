"""
此脚本用于生成并保存StyleID的去噪过程中每一步的图像
使用方法:
python generate_denoising_steps.py \
    --cnt /path/to/content \
    --sty /path/to/style \
    --output_path /path/to/output/steps \
    --precomputed /path/to/precomputed_feats
python baseline/Gaussian.py \
    --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/denoising_visualization \
    --precomputed /data2/ranxiangyu/styleid_out/style_out/precomputed_feats \
    --cnt_name 22811he_10049_12931.png \
    --sty_name masson.png
"""

import cv2
import numpy as np
import os

def add_gaussian_noise(image, noise_level):
    """给图像添加高斯噪声，noise_level 范围 0~1"""
    mean = 0
    stddev = noise_level * 255
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def gradually_add_noise(image_path, steps=10, output_dir='noisy_images'):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    for i in range(steps + 1):
        noise_level = i / steps  # 从 0 到 1
        noisy_image = add_gaussian_noise(image, noise_level)
        output_path = os.path.join(output_dir, f"step_{i:02d}.png")
        cv2.imwrite(output_path, noisy_image)
        print(f"已保存：{output_path}")

# 使用方法（你可以修改图片路径）
gradually_add_noise('/data2/ranxiangyu/styleid_out/style_out/styleid/22811he_10049_13955_stylized_pasm.png',
    output_dir='/data2/ranxiangyu/styleid_out/style_out/gaussian_noise_visualization_PASM_OUT'
)