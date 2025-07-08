import os
import shutil

# 定义源文件夹和目标文件夹
source_dir = '/data2/ranxiangyu/kidney_patch/patch_png/level0/22811pasm'
test_dir = '/data2/ranxiangyu/styleid_out/style_out/cyclegan_pasm/testB'
train_dir = '/data2/ranxiangyu/styleid_out/style_out/cyclegan_pasm/trainB'

# 创建目标文件夹（如果不存在的话）
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# 获取源文件夹中的所有文件，并按文件名排序
all_files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])

# 计算分割点
total_files = len(all_files)
split_index = total_files * 1 // 3  # 1:2分割

# 复制文件到目标文件夹
for file in all_files[:split_index]:
    shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

for file in all_files[split_index:]:
    shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

print(f"文件已成功分配：{split_index} 文件到文件夹 test_dir，{total_files - split_index} 文件到文件夹 train_dir。")