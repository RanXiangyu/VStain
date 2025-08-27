import os
# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
# import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
import sys
import contextlib

'''
 python create_patches_fp.py \
  --source /data2/ranxiangyu/kidney_wsi \
  --save_dir /data2/ranxiangyu/patch_test \
  --patch_size 512 \
  --step_size 512 \
  --patch_level 1 \
  --seg \
  --patch \
  --stitch \
  --no_auto_skip \
  --save_mask \
  --num_files 1
'''

class WSIPatchExtractor:
	def __init__(self):
		# 设置OpenSlide的调试模式为安静模式 tiff文件格式警报
		os.environ['OPENSLIDE_DEBUG'] = 'quiet'

		self.default_seg_params = {
			'seg_level': -1,
			'sthresh': 8,
			'mthresh': 7,
			'close': 4,
			'use_otsu': False,
			'keep_ids': 'none',
			'exclude_ids': 'none'
		}

		self.default_filter_params = {
			'a_t': 100,
			'a_h': 16,
			'max_n_holes': 8
		}

		# 可视化参数的默认值
		self.default_vis_params = {
			'vis_level': -1,
			'line_thickness': 500
		}

		self.default_patch_params = {
			'use_padding': True,
			'contour_fn': 'four_pt'
		}


	def stitching(self, file_path, wsi_object, downscale = 64):
		# 读取切片结果并生成低分辨率的拼接图
		start = time.time()
		heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
		total_time = time.time() - start
		
		return heatmap, total_time

	def segment(self, WSI_object, seg_params = None, filter_params = None, mask_file = None):
		"""对wsi进行组织分割"""        
		if seg_params is None:
			seg_params = self.default_seg_params
		if filter_params is None:
			filter_params = self.default_filter_params
		### Start Seg Timer
		start_time = time.time()
		# Use segmentation file
		if mask_file is not None:
			WSI_object.initSegmentation(mask_file)
		# Segment	
		else:
			WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

		### Stop Seg Timers
		seg_time_elapsed = time.time() - start_time   
		return WSI_object, seg_time_elapsed

	def patching(self, WSI_object, **kwargs):
		"""对wsi进行切片处理"""
		### Start Patch Timer
		start_time = time.time()
		
		# Patch
		file_path = WSI_object.process_contours(**kwargs)
		
		### Stop Patch Timer
		patch_time_elapsed = time.time() - start_time
		return file_path, patch_time_elapsed


	def seg_and_patch(self, source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size=256, step_size=256, 
				  seg_params=None, filter_params=None, vis_params=None, patch_params=None,
				  patch_level=0,
				  use_default_params=False, 
				  seg=False, save_mask=True, 
				  stitch=False, 
				  patch=False, auto_skip=True, process_list=None, num_files=None):
		""" 批量处理多个WSI文件，包括组织分割、切片和拼接 """

		if seg_params is None:	
			seg_params = self.default_seg_params
		if filter_params is None:
			filter_params = self.default_filter_params
		if vis_params is None:
			vis_params = self.default_vis_params
		if patch_params is None:
			patch_params = self.default_patch_params

		# 加载数据
		slides = sorted(os.listdir(source))
		slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
		if process_list is None:
			df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
		
		else:
			df = pd.read_csv(process_list)
			df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

		mask = df['process'] == 1
		process_stack = df[mask]

		# 限制处理的文件数量
		if num_files is not None:
			process_stack = process_stack.iloc[:num_files]
		total = len(process_stack)

		legacy_support = 'a' in df.keys()
		if legacy_support:
			print('detected legacy segmentation csv file, legacy support enabled')
			df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
			'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
			'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
			'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
			'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

		seg_times = 0.
		patch_times = 0.
		stitch_times = 0.

		for i in tqdm(range(total)):
			df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
			idx = process_stack.index[i]
			slide = process_stack.loc[idx, 'slide_id']
			print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
			print('processing {}'.format(slide))
			
			df.loc[idx, 'process'] = 0
			slide_id, _ = os.path.splitext(slide)

			if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
				print('{} already exist in destination location, skipped'.format(slide_id))
				df.loc[idx, 'status'] = 'already_exist'
				continue

			# Inialize WSI
			full_path = os.path.join(source, slide)
			WSI_object = WholeSlideImage(full_path)

			# WSI_object = WholeSlideImage(full_path)
			# 修改跳过打不开的文件，并输出打不开的文件的列表
		# 	try:
		# 		WSI_object = WholeSlideImage(full_path)
		# 	except Exception as e:
		# 		 print(f"Error processing file {full_path}: {e}")
		# # 记录错误文件以便之后检查
		# with open("error_files.log", "a") as log_file:
		#     log_file.write(f"{full_path}\n")
		# return None, None  # 跳过该文件并返回空值

			if use_default_params:
				current_vis_params = vis_params.copy()
				current_filter_params = filter_params.copy()
				current_seg_params = seg_params.copy()
				current_patch_params = patch_params.copy()
				
			else:
				current_vis_params = {}
				current_filter_params = {}
				current_seg_params = {}
				current_patch_params = {}


				for key in vis_params.keys():
					if legacy_support and key == 'vis_level':
						df.loc[idx, key] = -1
					current_vis_params.update({key: df.loc[idx, key]})

				for key in filter_params.keys():
					if legacy_support and key == 'a_t':
						old_area = df.loc[idx, 'a']
						seg_level = df.loc[idx, 'seg_level']
						scale = WSI_object.level_downsamples[seg_level]
						adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
						current_filter_params.update({key: adjusted_area})
						df.loc[idx, key] = adjusted_area
					current_filter_params.update({key: df.loc[idx, key]})

				for key in seg_params.keys():
					if legacy_support and key == 'seg_level':
						df.loc[idx, key] = -1
					current_seg_params.update({key: df.loc[idx, key]})

				for key in patch_params.keys():
					current_patch_params.update({key: df.loc[idx, key]})

			if current_vis_params['vis_level'] < 0:
				if len(WSI_object.level_dim) == 1:
					current_vis_params['vis_level'] = 0
				
				else:	
					wsi = WSI_object.getOpenSlide()
					best_level = wsi.get_best_level_for_downsample(64)
					current_vis_params['vis_level'] = best_level

			if current_seg_params['seg_level'] < 0:
				if len(WSI_object.level_dim) == 1:
					current_seg_params['seg_level'] = 0
				
				else:
					wsi = WSI_object.getOpenSlide()
					best_level = wsi.get_best_level_for_downsample(64)
					current_seg_params['seg_level'] = best_level

			keep_ids = str(current_seg_params['keep_ids'])
			if keep_ids != 'none' and len(keep_ids) > 0:
				str_ids = current_seg_params['keep_ids']
				current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
			else:
				current_seg_params['keep_ids'] = []

			exclude_ids = str(current_seg_params['exclude_ids'])
			if exclude_ids != 'none' and len(exclude_ids) > 0:
				str_ids = current_seg_params['exclude_ids']
				current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
			else:
				current_seg_params['exclude_ids'] = []

			w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
			if w * h > 1e8:
				print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
				df.loc[idx, 'status'] = 'failed_seg'
				continue

			df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
			df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


			seg_time_elapsed = -1
			if seg:
				WSI_object, seg_time_elapsed = self.segment(WSI_object, current_seg_params, current_filter_params) 

			if save_mask:
				mask = WSI_object.visWSI(**current_vis_params)
				mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
				mask.save(mask_path)

			patch_time_elapsed = -1 # Default time
			if patch:
				current_patch_params.update({
					'patch_level': patch_level, 
					'patch_size': patch_size, 
					'step_size': step_size, 
					'save_path': patch_save_dir
					})
				file_path, patch_time_elapsed = self.patching(WSI_object = WSI_object,  **current_patch_params,)
			
			stitch_time_elapsed = -1
			if stitch:
				file_path = os.path.join(patch_save_dir, slide_id+'.h5')
				if os.path.isfile(file_path):
					heatmap, stitch_time_elapsed = self.stitching(file_path, WSI_object, downscale=64)
					stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
					heatmap.save(stitch_path)

			print("segmentation took {} seconds".format(seg_time_elapsed))
			print("patching took {} seconds".format(patch_time_elapsed))
			print("stitching took {} seconds".format(stitch_time_elapsed))
			df.loc[idx, 'status'] = 'processed'

			seg_times += seg_time_elapsed
			patch_times += patch_time_elapsed
			stitch_times += stitch_time_elapsed

		seg_times /= total
		patch_times /= total
		stitch_times /= total

		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		print("average segmentation time in s per slide: {}".format(seg_times))
		print("average patching time in s per slide: {}".format(patch_times))
		print("average stitching time in s per slide: {}".format(stitch_times))
			
		return seg_times, patch_times

	def process(self, source, save_dir, patch_size=256, step_size=256, 
                patch_level=0, seg=True, patch=True, stitch=True, 
                save_mask=True, auto_skip=True, process_list=None, 
                num_files=None, custom_seg_params=None, custom_filter_params=None, 
                custom_vis_params=None, custom_patch_params=None, 
                use_default_params=False):
		"""
        处理WSI文件，进行组织分割、切片和拼接
        Args:
            source (str): WSI文件源目录
            save_dir (str): 保存目录
            patch_size (int): 切片大小
            step_size (int): 步长大小
            patch_level (int): 切片层级
            seg (bool): 是否进行组织分割
            patch (bool): 是否进行切片
            stitch (bool): 是否进行拼接
            save_mask (bool): 是否保存掩码
            auto_skip (bool): 是否自动跳过已处理文件
            process_list (str): 处理列表文件路径
            num_files (int): 要处理的文件数量
            custom_seg_params (dict): 自定义分割参数
            custom_filter_params (dict): 自定义过滤参数
            custom_vis_params (dict): 自定义可视化参数
            custom_patch_params (dict): 自定义切片参数
            use_default_params (bool): 是否使用默认参数
        """		
		seg_params = custom_seg_params if custom_seg_params else self.default_seg_params
		filter_params = custom_filter_params if custom_filter_params else self.default_filter_params
		vis_params = custom_vis_params if custom_vis_params else self.default_vis_params
		patch_params = custom_patch_params if custom_patch_params else self.default_patch_params

		# 创建保存目录
		patch_save_dir = os.path.join(save_dir, 'patches')
		mask_save_dir = os.path.join(save_dir, 'masks')
		stitch_save_dir = os.path.join(save_dir, 'stitches')

		for directory in [patch_save_dir, mask_save_dir, stitch_save_dir]:
			os.makedirs(directory, exist_ok=True)
		
		return self.seg_and_patch(
			source=source,
			save_dir=save_dir,
			patch=patch,
			patch_save_dir=patch_save_dir,
			mask_save_dir=mask_save_dir,
			stitch_save_dir=stitch_save_dir,
			patch_size=patch_size,
			step_size=step_size,
			seg_params=seg_params,
			filter_params=filter_params,
			vis_params=vis_params,
			patch_params=patch_params,
			patch_level=patch_level,
			use_default_params=use_default_params,
			seg=seg,
			save_mask=save_mask,
			stitch=stitch,
			auto_skip=auto_skip,
			process_list=process_list,
			num_files=num_files
		)




	
