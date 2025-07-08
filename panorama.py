"""
核心代码 全景图的生成
"""

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from tqdm import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

# 将全景图划分为多个重叠的小窗口，全景图的高度，全景图的宽度，每个处理窗口的大小，相邻窗口的步长
def get_views(panorama_height, panorama_width, window_size=64, stride=8):
    # 潜在空间为 1/8
    panorama_height /= 8
    panorama_width /= 8
    # 计算在给定的宽度和高度下，需要多少个窗口；滑动窗口的标准计算方式
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)

    # 生成窗口坐标
    # 逻辑是从左到右从上到下
    views = []
    for i in range(total_num_blocks): # i ： 0 - n-1
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

    # 文本处理方法
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    # 图像解码方法
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5):
        # guidance_scale：7.5 分类器自由引导的缩放因子
        if isinstance(prompts, str):
            # isinstance python 内建函数，判断prompts是否为字符串，如果是就转换为一个列表
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds 文本提示处理为嵌入
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Define panorama grid and get views
        # 初始化潜在变量，作为噪声输入，（batch_size， unet输入的通道数， height // 8， width // 8）
        latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        # 获取全景图的窗口划分
        views = get_views(height, width)
        # 创建累计和计数张量
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        # 设置扩散步骤
        self.scheduler.set_timesteps(num_inference_steps)
        # 扩散调度器（scheduler），负责定义：每一步应该去多少噪声；时间步 t 的顺序和数量；噪声预测与图像更新的公式（如 DDIM、DDPM、PNDM 等都对应不同 scheduler）

        # 在with代码块中使用自动混合精度来计算
        with torch.autocast('cuda'):
            """
            for t in timesteps:
                clear count & value
                for each sliding window:
                    extract latent_view
                    unet(noise prediction)
                    do CFG
                    denoise latent_view
                    add result to value
                    increment count
                fuse all patches → new latent
            """
            # enumerate同时拿到每一步的索引i和时间步t （tqdm添加进度条（一个扩散调度器设定的时间步列表））
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                # 清0张量
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views: # 遍历每个窗口

                    # 提取当前窗口的潜在变量，用于局部去噪
                    # TODO we can support batches, and pass multiple views at once to the unet
                    # 从潜在变量中提取当前窗口的视图，laten的大小和想要生成的图片大小一致（1/8）
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    # CFG公式会增强prompt的影响力，guidance_scale越大，prompt的影响力越大
                    # （复制为两个元素的列表[latent_view, latent_view]）  shape: [2, C, H, W]，因为CFG技术
                    """ CFG原理
                    	1. 一份图像 + 无条件 prompt（empty / null embedding）
                        2.	一份图像 + 有条件 prompt（真实文本 embedding）
                        3.	把两者同时输入 UNet → 得到两个噪声预测
                    """
                    latent_model_input = torch.cat([latent_view] * 2)

                    # predict the noise residual
                    # 预测当前步的噪声残差
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                    # perform guidance
                    # 做CFG
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the denoising step with the reference model
                    # 执行扩散第一步
                    latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']
                    # 将patch加入融合图像当中
                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step
                # 在当前时间步，所有patch融合
                latent = torch.where(count > 0, value / count, value)

        # Img latents -> imgs 解码返回
        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='a photo of the dolomites')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--outfile', type=str, default='out.png')
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = MultiDiffusion(device, opt.sd_version)

    img = sd.text2panorama(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # save image
    img.save(opt.outfile)
