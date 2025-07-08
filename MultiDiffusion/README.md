# 代码结构
```
project
│   run_styleid.py  # 主运行脚本，用于执行风格迁移任务
│
├── diffusers_implementation  # 包含与 Diffusers 相关的实现代码
│   │   config.py  # 配置文件
│   │   run_styleid_diffusers.py  # 使用 Diffusers 库进行风格迁移的脚本
│   │   stable_diffusion.py  # 稳定扩散模型的实现
│   │   utils.py  # 工具函数
│   │
│   ├── evaluation  # 评估相关代码
│   │   │   eval_artfid.py # 计算ArtFid（freshet inception distance）指标的代码
│   │   │   eval_histogan.py # 计算Histogan指标的代码
│   │   │   eval.sh
│   │   │   image_metrics.py # psnr & ssim
│   │   │   inception.py # inception模型 v3
│   │   │   net.py # 神经网络
│   │   │   utils.py
│
├── ldm  # Latent Diffusion Model (LDM) 相关代码
│   │   lr_scheduler.py  # 学习率调度器
│   │   util.py  # 工具函数
│   │
│   ├── models  # 模型实现
│   ├── modules  # 模块实现
│   ├── ldm  # 存放 LDM 模型的目录
│
├── output  # 存放输出结果的目录
├── precomputed_feats  # 预计算特征的目录
├── src  # 源代码目录
│
└── taming-transformers  # Taming Transformers 相关代码
    └── util  # 工具函数目录

```

## 知识点
### 关于EMA技术
model.ema_scope() 是ldm模型自带的函数，用于在推理过程当中使用EMA参数进行前向传播

EMA的作用：
  
- 在训练的过程中，模型参数会不断更新。EMA通过对于参数的指数平均进行平滑，减少了参数的波动，从而得到更稳定的参数
- 减少过拟合，提高模型的性能（特别是在验证和测试阶段，EMA参数比直接使用训练参数更加的稳定）
- model.ema_scope是ldm自带的函数，用于在推理过程中使用EMA参数进行前向传播

inception模型：

- 使用不同的卷积核并行提取多尺度特征