"""
这个文件记录一些在代码的编写过程中，没有最后被采纳的部分，作为备份
"""

# 调试代码部分

    """ ==================== 添加的调试代码开始 ====================
    # 将 feat_maps 的详细信息保存到 feat_maps_debug.txt 文件中
    with open("feat_maps_debug_v1.txt", "w", encoding="utf-8") as f:
        f.write("--- Feat Maps Debug Info ---\n\n")
        f.write(f"Length of feat_maps: {len(feat_maps)}\n")
        f.write(f"Type of feat_maps: {type(feat_maps)}\n\n")

        if len(feat_maps) > 0:
            f.write(f"--- Details of the first element ---\n")
            f.write(f"Type of first element: {type(feat_maps[0])}\n")
            if isinstance(feat_maps[0], dict):
                f.write(f"Keys in first element: {list(feat_maps[0].keys())}\n")
        else:
            f.write("The feat_maps list is EMPTY.\n")
        
        f.write("\n--- Full content of feat_maps ---\n")
        f.write(str(feat_maps))

    print("[调试信息] feat_maps 的内容已保存到 feat_maps_debug_v1.txt 文件中，请查看。")
    # ==================== 添加的调试代码结束 ==================== """


    # sty_feature, sty_z_enc = feature_extractor(opt.sty, sty_img, feature_dir, model, sampler, ddim_inversion_steps, uc, time_idx_dict, opt.start_step, save_feature_timesteps, ddim_sampler_callback, save_feat=True)

# 主循环

# b. DDIM Inversion 捕获特征
                # _ = feature_extractor(purpose='content', model=model, sampler=sampler, uc=uc, time_idx_dict=time_idx_dict, ddim_sampler_callback=ddim_sampler_callback, direct_latent_input=z_0_patch, target_step_num=i)


        # final_wsi_img = Image.fromarray(final_wsi_np) 
        # final_wsi_img.save(os.path.join(opt.out, f"final_decoded_{idx}.tiff"))

# 函数部分
# 特征提取 在特定的时间步 save_feature_timesteps
def feature_extractor(
    # --- 通用参数 ---
    purpose='style', # 模式 ('style' 或 'content')
    model, 
    sampler, 
    uc,
    time_idx_dict, 
    save_feature_timesteps,
    ddim_sampler_callback=None,
    ddim_inversion_steps=50, 
    # --- style参数 ---
    img_dir,
    img_name, 
    feature_dir, 
    start_step=49, 
    save_feat=False
    # --- content参数 ---
    direct_latent_input: torch.Tensor = None, # 直接接收content
    target_step_num=None,    # 👈 指定单个时间步（如30），默认为None不提取
    ):

    global feat_maps
    feat_maps = []
    
    img_feature = None
    img_z_enc = None
    feature = None # 指定时间步的特征

    if purpose == 'content':
        if direct_latent_input is None:
            raise ValueError("For 'content' purpose, provide the latent tensor via 'direct_latent_input'.")
        if target_step_num is None:
            raise ValueError("For 'content' purpose, specify the step number via 'target_step_num'.")
        if not (1 <= target_step_num <= ddim_inversion_steps):
            raise ValueError(f"'target_step_num' must be between 1 and {ddim_inversion_steps}.")

        # 1. 计算循环 i和 确切的时间步
        target_timestep_t = time_idx_dict[target_step_index] # 实际时间步 t

        # 2. 执行DDIM反演
        _, _ = sampler.encode_ddim(direct_latent_input.clone(), num_steps=ddim_inversion_steps,
                                            unconditional_conditioning=uc,
                                            end_step=target_step_num,
                                            callback_ddim_timesteps=[target_timestep_t],
                                            img_callback=ddim_sampler_callback)
        
        return None    
    
    elif purpose == 'style':
        img_path = os.path.join(img_dir, img_name)
        init_img = preprocess_img(img_path).to('cuda')
        img_feat_name = os.path.join(feature_dir, os.path.basename(img_name).split('.')[0] + '_sty.pkl')

        # 1. 直接加载特征返回 style图片
        if os.path.exists(img_feat_name):
            print(f"[✓] Loading style feature from {img_feat_name}")
            with open(img_feat_name, 'rb') as f:
                img_feature = pickle.load(f)
                img_z_enc = torch.clone(img_feature[0]['z_enc'])
            return img_feature, z_enc_startstep

        # 2. 进行ddim反演，获取特征图
        init_img = model.get_first_stage_encoding(model.encode_first_stage(init_img))  # [1, 4, 64, 64] z_0
        img_z_enc, _ = sampler.encode_ddim(init_img.clone(), num_steps = ddim_inversion_steps, \
                                            unconditional_conditioning = uc, \
                                            end_step = time_idx_dict[ddim_inversion_steps - 1 - start_step], \
                                            callback_ddim_timesteps = save_feature_timesteps, \
                                            img_callback = ddim_sampler_callback)
        
        img_feature = copy.deepcopy(feat_maps)
        # img_z_enc = feat_maps[0]['z_enc']
        img_z_enc = img_feature[0]['z_enc']


        if save_feat and len(feature_dir) > 0:
            print(f"Saving style feature to {img_feat_name}")
            with open(img_feat_name, 'wb') as f:
                pickle.dump(img_feature, f)

        return img_feature, img_z_enc