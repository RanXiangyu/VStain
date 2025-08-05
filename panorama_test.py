def seed_everything(seed):

def get_views(panorama_height, panorama_width, window_size=64, stride=8):# 返回窗口坐标


class MultiDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def text2panorama(self, prompts, negative_prompts='', height=512, width=2048, num_inference_steps=50,
                      guidance_scale=7.5):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        latent = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        views = get_views(height, width)
        count = torch.zeros_like(latent)
        value = torch.zeros_like(latent)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                count.zero_()
                value.zero_()

                for h_start, h_end, w_start, w_end in views: # 遍历每个窗口
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end] # 取出当前窗口的 latent 部分

                    latent_model_input = torch.cat([latent_view] * 2) # 复制两份，用于 CFG 操作

                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample'] # UNet 去噪

                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view)['prev_sample']  # 更新 latent（一步）
                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised # 把 denoised 结果加进累计图
                    count[:, :, h_start:h_end, w_start:w_end] += 1 # 对应区域的更新次数 +1

                latent = torch.where(count > 0, value / count, value)

        imgs = self.decode_latents(latent)  # [1, 3, 512, 512]
        img = T.ToPILImage()(imgs[0].cpu())
        return img


