python run_styleid.py --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he --sty /data2/ranxiangyu/kidney_patch/style --output_path /data2/ranxiangyu/styleid_out/style_out

 python run_styleid_wsi.py \
    --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/styleid \
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats 


 python run_styleid_wsi.py \
    --cnt /data2/ranxiangyu/styleid_out/style_out/cyclegan/testA \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/styleid_no_injection \
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats \
    --without_attn_injection


 python run_styleid_wsi.py \
    --cnt /data2/ranxiangyu/styleid_out/style_out/cyclegan/testA \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/styleid_no_adain \
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats \
    --without_init_adain

CUDA_VISIBLE_DEVICES=0 python evaluation/eval_artfid.py  --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he \
   --sty /data2/ranxiangyu/kidney_patch/style \
   --tar /data2/ranxiangyu/styleid_out/style_out/styleid \
   --batch_size 8 --num_workers 8 \
   --content_metric lpips --mode art_fid_inf --device cuda

3195/3195 [00:20<00:00, 157.70it/s]
不同 batch size 下的 FID: [16.22879804 16.22879803 16.22879803 16.22879798 16.228798   16.22879796
 16.22879811 16.228798  ]
预测 batch_size → ∞ 时的 FID: 16.2288

python evaluation/eval_artfid.py  --cnt /data2/ranxiangyu/styleid_out/style_out/cyclegan/testA \
   --sty /data2/ranxiangyu/kidney_patch/style \
   --tar /data2/ranxiangyu/styleid_out/style_out/styleid_no_adain \
   --batch_size 4 --num_workers 8 \
   --content_metric lpips --mode art_fid_inf --device cuda

   不同 batch size 下的 FID: [16.67552261 16.67552262 16.67552266 16.6755226  16.67552259 16.67552259
 16.67552257 16.67552264]
预测 batch_size → ∞ 时的 FID: 16.6755

CUDA_VISIBLE_DEVICES=1 python evaluation/eval_artfid.py  --cnt /data2/ranxiangyu/styleid_out/style_out/cyclegan_pas/testA \
   --sty //data2/ranxiangyu/styleid_out/style_out/adain/sty_pas \
   --tar /data2/ranxiangyu/styleid_out/style_out/cyclegan_pas/output/B \
   --batch_size 4 --num_workers 8 \
   --content_metric lpips --mode art_fid_inf --device cuda
不同 batch size 下的 FID: [20.16834974 20.16834974 20.16834974 20.16834974 20.16834974 20.16834974
 20.16834974 20.16834974]
预测 batch_size → ∞ 时的 FID: 20.1683

CUDA_VISIBLE_DEVICES=0 python evaluation/eval_histogan.py --sty /data2/ranxiangyu/kidney_patch/style --tar /data2/ranxiangyu/styleid_out/style_out/styleid 
color matching loss: 0.48324906826019287

python evaluation/eval_histogan.py --tar /data2/ranxiangyu/styleid_out/style_out/styleid_no_adain \
   --sty /data2/ranxiangyu/kidney_patch/style
1065 3 355
color matching loss: 0.454760879278183

CUDA_VISIBLE_DEVICES=0 python evaluation/eval_histogan.py --sty /data2/ranxiangyu/styleid_out/style_out/cyclegan/style \
   --tar /data2/ranxiangyu/styleid_out/style_out/cyclegan/output/B 




python adain.py --content_dir /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he --style_dir /data2/ranxiangyu/kidney_patch/style --output_dir /data2/ranxiangyu/styleid_out/style_out/adain --alpha 0.5 --gpu 1 




CUDA_VISIBLE_DEVICES=0 python cyclegan.py --n_epochs 5 --batch_size 4
CUDA_VISIBLE_DEVICES=1 python baseline/cyclegan.py


python evaluation/eval_histogan.py --tar /data2/ranxiangyu/styleid_out/style_out/adain/he2masson \
   --sty /data2/ranxiangyu/styleid_out/style_out/adain/sty_masson

    
python evaluation/eval_artfid.py  --cnt /data2/ranxiangyu/styleid_out/style_out/cyclegan_masson/testA \
   --sty /data2/ranxiangyu/styleid_out/style_out/adain/sty_masson \
   --tar  /data2/ranxiangyu/styleid_out/style_out/adain/he2masson\
   --batch_size 4 --num_workers 8 \
   --content_metric lpips --mode art_fid_inf --device cuda

CUDA_VISIBLE_DEVICES=1 python evaluation/eval_artfid.py  --cnt /data2/ranxiangyu/styleid_out/style_out/cyclegan_pas/testA \
   --sty //data2/ranxiangyu/styleid_out/style_out/adain/sty_pas \
   --tar /data2/ranxiangyu/styleid_out/style_out/cyclegan_pas/output/A \
   --batch_size 4 --num_workers 8 \
   --content_metric lpips --mode art_fid_inf --device cuda
   

pas
ArtFID: 16.63277244567871 FID: 9.9496445481826 LPIPS: 0.5190240144729614 LPIPS_gray: 0.49000614881515503
CFSD: -0.0007
0.1655467450618744
ArtFID: 41.829368591308594 FID: 26.120571920072518 LPIPS: 0.542348325252533 LPIPS_gray: 0.5132366418838501
CFSD: -0.0007