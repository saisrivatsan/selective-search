#-----------------------------------------
# Selective Search + RCNN Implementation
# Author: Sai Srivatsa Ravindranath
#-----------------------------------------

import os
import rcnn
import ssearch
import scipy.io
import numpy as np
import sim_features as sf
import matplotlib.pyplot as plt

from color_utils import convert_colorspace
from skimage.segmentation import felzenszwalb


def demo(image_name,color_space_list=None,ks=None,sim_feats_list=None,net='vgg16', cpu_mode=True):
	''' Object Recognition Demo : Selective Search + RCNN
	parameters
	----------
	image_name : filename of image stored in 'Data/img'
	
	color_space_list : list of colorspaces to be used. Refer color_utils for list of possible colorspaces.
	Default : [ 'HSV', 'LAB']

	ks : list felzenszwalb scale/threshold and minimum segment size.
	Default : [50, 100]
	
	'''
	
	blob_array = []	
	priority = []
	img = plt.imread('Data/img/' + image_name + '.jpg')
	seg_dir = 'Data/segments/'
	if color_space_list is None: color_space_list = ['HSV','LAB']
	if ks is None: ks = [50,100]
	if sim_feats_list is None: sim_feats_list = [[ sf.color_hist_sim(), sf.texture_hist_sim(), sf.size_sim(img.shape), sf.fill_sim(img.shape) ],[ sf.texture_hist_sim(), sf.size_sim(img.shape), sf.fill_sim(img.shape) ]]

	cc = convert_colorspace(img,color_space_list)
	seg_filename = [seg_dir + 'HSV/50/' + image_name +'.mat',seg_dir + 'HSV/100/' + image_name +'.mat', seg_dir + 'LAB/50/' + image_name +'.mat',seg_dir + 'LAB/100/' + image_name +'.mat']

	for i in range(len(color_space_list)):
		for j in range(len(ks)):
			for k in range(len(sim_feats_list)):
				_img = cc[i]
				_file = "%s%s/%d/%s.mat"%(seg_dir,color_space_list[i].upper(),ks[j],image_name)
				if not os.path.exists(_file):
					segment_mask = felzenszwalb(_img,scale=ks[j],sigma=0.8,min_size=ks[j])
					_temp_dict = dict()
					_temp_dict['blobIndIm'] = segment_mask + 1
					scipy.io.savemat(_file,_temp_dict)
				_blob_array = ssearch._ssearch(_img,ssearch.load_segment_mask(_file),sim_feats = sim_feats_list[k])
				blob_array.append(_blob_array)
				priority.append( np.arange(len(_blob_array),0,-1).clip(0,(len(_blob_array)+1)/2))
		
	bboxes = ssearch.remove_duplicate(blob_array,priority)
	bbox_dict = {}
	bbox_dict['boxes'] = np.vstack([np.asarray(bboxes)[:,2],np.asarray(bboxes)[:,1],np.asarray(bboxes)[:,4],np.asarray(bboxes)[:,3]]).T
	print('\nComputed %d proposals'%(len(bboxes)))
	scipy.io.savemat('Data/Boxes/' + image_name + '.mat',bbox_dict)
	rcnn.rcnn_demo(image_name,net=net, cpu_mode=cpu_mode)
	
	
	
