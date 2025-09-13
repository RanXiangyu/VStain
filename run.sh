 python evaluation/eval.py \
    --wsi_path /home/hfang/rxy/kidney_wsi/ \
    --stained_path /mnt/hfang/data/VStain/output/ \
    --h5_path /mnt/hfang/data/VStain/h5/patches/ \
    --ckpt_path /home/hfang/rxy/ckpt/inceptionv3.pth \
    --patch_size 512 --patch_level 0 \
    --batch_size 2 --num_workers 1 --content_metric lpips \
    --wsi_patch_output_path /home/hfang/rxy/kidney_patch --stain_type pasm

python evaluation/eval.py \
    --wsi_path /home/hfang/rxy/kidney_wsi/ \
    --stained_path /mnt/hfang/data/VStain/output/ \
    --h5_path /mnt/hfang/data/VStain/h5/patches/ \
    --ckpt_path /home/hfang/rxy/ckpt/inceptionv3.pth \
    --patch_size 512 --patch_level 0 \
    --batch_size 2 --num_workers 1 --content_metric lpips \
    --wsi_patch_output_path /home/hfang/rxy/kidney_patch --stain_type pas

 python evaluation/eval.py \
    --wsi_path /home/hfang/rxy/kidney_wsi/ \
    --stained_path /mnt/hfang/data/VStain/output/ \
    --h5_path /mnt/hfang/data/VStain/h5/patches/ \
    --ckpt_path /home/hfang/rxy/ckpt/inceptionv3.pth \
    --patch_size 512 --patch_level 0 \
    --batch_size 2 --num_workers 1 --content_metric lpips \
    --wsi_patch_output_path /home/hfang/rxy/kidney_patch