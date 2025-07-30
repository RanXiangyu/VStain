class FeatureHook:
    def __init__(self, feat_maps, idx_time_dict, attn_layer_idics, unet_model):
        """
        初始化 FeatureHook
        :param feat_maps: 全局特征存储列表，包含每个 timestep 的特征 dict
        :param idx_time_dict: 映射 timestep 到 feat_maps 索引
        :param attn_layer_idics: 哪些 output_blocks 中提取注意力特征
        :param unet_model: 稳定扩散模型中的 diffusion_model（U-Net）
        """
        self.feat_maps = feat_maps
        self.idx_time_dict = idx_time_dict
        self.attn_layer_idics = attn_layer_idics
        self.unet_model = unet_model

    # 保存单个时间步的 q//k/v 特征图
    def save_feature_map(self, feature_map, filename, time):
        cur_idx = idx_time_dict[time]
        self.feat_maps[cur_idx][f"{filename}"] = feature_map

    # 保存当前时间步的所有 q//k/v 特征图
    def save_qkv(self, blocks, time, feature_type="output_block"):
        block_idx = 0
        # stable diffusion的上采样 output_blocks有12个，在命令行输入的时候确定了再那几个图层进行保存qkv
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                # 包含多个子模块 并且是spatial transformer 通常在block[1]中
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    self.save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", time)
                    self.save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", time)
                    self.save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", time)
    
    # 保存当前时间步的所有特征图
    def save_qkv_callback(self, time):
        self.save_qkv(self.unet_model.output_blocks , time, "output_block")

    def ddim_sampler_callback(self, pred_x0, xt, time):
        self.save_qkv_callback(time) # [B, num_heads, N, head_dim]
        self.save_feature_map(xt, 'z_enc', time) # [B, C, H, W]（latent）保存图像本身在潜空间的内容，可以可视化图像的演化过程