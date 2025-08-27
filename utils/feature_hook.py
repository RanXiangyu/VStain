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
        # print(f"到达save_feature_map中的callback函数")
        
        cur_idx = self.idx_time_dict[time]
        self.feat_maps[cur_idx][f"{filename}"] = feature_map
        # 调用save_feature_map时，是对于全局变量feat_maps进行更新操作

    # 保存当前时间步的所有 q//k/v 特征图
    def save_style_kv(self, blocks, time, feature_type="output_block"):
        # print(f"到达save_style_kv中的callback函数")
        block_idx = 0
        # stable diffusion的上采样 output_blocks有12个，在命令行输入的时候确定了再那几个图层进行保存qkv
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                # 包含多个子模块 并且是spatial transformer 通常在block[1]中
                if block_idx in self.attn_layer_idics:
                    # self-attn
                    # q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    # self.save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", time)
                    self.save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", time)
                    self.save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", time)
    
    def save_q_only(self, blocks, time, feature_type="output_block"):
        """遍历U-Net的Blocks，只提取并保存q特征。"""
        # print(f"    -> Inside callback: Extracting Q-features at time {time}...")
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                # 假设 self_attn_output_block_indices 是一个包含目标层索引的列表
                if block_idx in self.attn_layer_idics:
                    q = block[1].transformer_blocks[0].attn1.q
                    # 只保存 q 特征
                    self.save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", time)
    
    # 保存当前时间步的所有特征图
    def save_style_kv_callback(self, time):
        self.save_style_kv(self.unet_model.output_blocks , time, "output_block")

    # 保存单个时间步的callback 用于encode_ddim中
    def content_q_update_callback(self, pred_x0, xt, time):
        self.save_q_only(self.unet_model.output_blocks, time, "output_block")
        # print(f"到达content_q_update_callback中的callback函数")


    def ddim_sampler_callback(self, pred_x0, xt, time):
        # print(f"到达ddim_sampler_callback中的callback函数")
        self.save_style_kv_callback(time) # [B, num_heads, N, head_dim]
        self.save_feature_map(xt, 'z_enc', time) # [B, C, H, W]（latent）保存图像本身在潜空间的内容，可以可视化图像的演化过程
        """==================== 添加的调试代码开始 ====================
        # 将 feat_maps 的详细信息保存到 feat_maps_debug.txt 文件中
        with open("feat_maps_debug.txt", "w", encoding="utf-8") as f:
            f.write("--- Feat Maps Debug Info ---\n\n")
            f.write(f"Length of feat_maps: {len(self.feat_maps)}\n")
            f.write(f"Type of feat_maps: {type(self.feat_maps)}\n\n")

            if len(self.feat_maps) > 0:
                f.write(f"--- Details of the first element ---\n")
                f.write(f"Type of first element: {type(self.feat_maps[0])}\n")
                if isinstance(self.feat_maps[0], dict):
                    f.write(f"Keys in first element: {list(self.feat_maps[0].keys())}\n")
            else:
                f.write("The feat_maps list is EMPTY.\n")
            
            f.write("\n--- Full content of feat_maps ---\n")
            f.write(str(self.feat_maps))

        print("[调试信息] feat_maps 的内容已保存到 feat_maps_debug.txt 文件中，请查看。") """
        