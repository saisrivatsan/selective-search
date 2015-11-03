#-----------------------------------------
# Selective Search + RCNN Implementation
# Author: Sai Srivatsa Ravindranath
#-----------------------------------------

import os
import glob
import hist
import time
import blobs
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from blobs import blob
from color_utils import convert_colorspace
from skimage.segmentation import mark_boundaries



def load_segment_mask(filename):
	'''Loads Segment mask pre-computed and stored at filename'''
	return scipy.io.loadmat(filename)['blobIndIm'] - 1

def compute_sim(blob_1,blob_2,sim_feats):
	''' Helper function to compute similarity '''
	similarity = 0
	for _sim_feat in sim_feats:
		similarity += _sim_feat(blob_1,blob_2)
	return similarity


def _ssearch(img,segment_mask,sim_feats=None):

	'''
	Performs selective_search on the given image
	parameters
	----------
		
	img : Input image
	
	segment_ mask :  Integer mask indicating segment labels of an image
	
	sim_feats : list of sim_features to be used
	Default(None) : [ color_hist_sim(),texture_hist_sim(),size_sim(img),fill_sim(img) ]
	
	returns
	--------
	
	blob_array : Array of blobs computed during the hierarchial process
	''' 
	a = time.time()	
	h = img.shape[0]
	w = img.shape[1]
	n_segments = len(set(segment_mask.flatten()))	
	blob_sizes = np.bincount(segment_mask.flatten())
	color_hists = hist.get_color_hist(img,segment_mask,n_bins=25)
	texture_hists = hist.get_texture_hist(img,segment_mask,n_orientation = 8, n_bins = 10)
	blob_array = []
	for i in range(n_segments):		
		blob_array.append(blob(i))
		_loc = np.argwhere(segment_mask==i)
		bbox = np.empty(4)
		bbox[0] = _loc[:,0].min()
		bbox[1] = _loc[:,1].min()
		bbox[2] = _loc[:,0].max()
		bbox[3] = _loc[:,1].max()
		blob_array[i].blob_size = blob_sizes[i]
		blob_array[i].bbox = bbox
		blob_array[i].color_hist = color_hists[i]
		blob_array[i].texture_hist = texture_hists[i]
	if sim_feats is None:
		sim_feats = [ sf.color_hist_sim(), sf.texture_hist_sim(), sf.size_sim(img), sf.fill_sim(img) ]
	neighbour_list = np.asarray(list(blobs.get_blob_neighbours(blob_array,segment_mask)))
	sim_list = np.vstack((neighbour_list.T,np.array([ compute_sim(blob_array[_idx[0]],blob_array[_idx[1]],sim_feats) for _idx in neighbour_list ]))).T

	while len(sim_list):
		
		# Get Max sim	
			
		sort_idx = np.argsort(sim_list[:,2])
		sim_list = sim_list[sort_idx]	
		blob_1 = blob_array[int(sim_list[-1][0])]
		blob_2 = blob_array[int(sim_list[-1][1])]		
		sim_list = sim_list[:-1]
		
		
		# Merge blobs
		
		t = len(blob_array)	
		blob_t = blobs.merge_blobs(blob_array,blob_1,blob_2,t)
		blob_array.append(blob_t)
			
		
		if len(sim_list)==0:
			break

		# Remove i,j from neighbour_list

		sim_list = sim_list[(sim_list[:,0]!=blob_1.blob_idx) & (sim_list[:,1]!=blob_1.blob_idx)]
		sim_list = sim_list[(sim_list[:,0]!=blob_2.blob_idx) & (sim_list[:,1]!=blob_2.blob_idx)]
		new_sim_list = np.array([[i,t,compute_sim(blob_array[i],blob_array[t],sim_feats)] for i in blob_t.neighbours])
		if len(new_sim_list):		
			sim_list = np.vstack((sim_list,new_sim_list))


	print('.'),	
	return blob_array

def remove_duplicate(blob_array,priority):
	''' Removes Duplicate Boxes 
	parameters
	-----------
	
	blob_array : array of blob_arrays for various strategies
	
	priority : array of priority arrays associated with blobs in blob_array
	
	returns
	-------
	bboxes : unique set of boxes with priorities. Shape [n_blobs,4]
	Note: box is of the form [xmin,xmax,ymin,ymax] ie img[xmin:xmax,ymin:ymax] denoted the selected region
	'''
	
	_boxes = [ tuple(_blob.bbox) for __blob in blob_array for _blob in __blob]
	_priority = np.asarray([ p for _p in priority for p in _p])
	_unq_boxes = set(_boxes)
	_boxes = np.asarray(_boxes)
	
	sort_idx = np.argsort(_priority)
	_priority = _priority[sort_idx]
	_boxes = _boxes[sort_idx]
	bboxes = []
	for _box,_p in zip(_boxes,_priority):
		if tuple(_box) in _unq_boxes:
			bboxes.append(np.append(_p,_box))
			_unq_boxes.remove(tuple(_box))	
	return bboxes

def ssearch_fast(filename,seg_dir = '/home/sai/Documents/selective_search/VOC2007/segments/',save_name=None):	

	''' Helper function for calling ssearch in fast mode '''
	a = time.time()
	blob_array = []	
	priority = []
	img = plt.imread(filename)
	cc = convert_colorspace(img,['hsv','LAB'])
	
	seg_dir = '/home/sai/Documents/selective_search/VOC2007/segments/'
	seg_filename = [seg_dir + 'HSV/50/' + filename[-10:-4] +'.mat',seg_dir + 'HSV/100/' + filename[-10:-4] +'.mat', seg_dir + 'LAB/50/' + filename[-10:-4] +'.mat',seg_dir + 'LAB/100/' + filename[-10:-4] +'.mat']

	for i,_file in enumerate(seg_filename):
		_img = cc[i/2]
		_blob_array = _ssearch(_img,load_segment_mask(_file),sim_feats = [ sf.color_hist_sim(), sf.texture_hist_sim(), sf.size_sim(img.shape), sf.fill_sim(img.shape) ])
		blob_array.append(_blob_array)
		priority.append( np.arange(len(_blob_array),0,-1).clip(0,(len(_blob_array)+1)/2))

		_blob_array = _ssearch(_img,load_segment_mask(_file),sim_feats = [ sf.texture_hist_sim(), sf.size_sim(img.shape), sf.fill_sim(img.shape) ])
		blob_array.append(_blob_array)
		priority.append( np.arange(len(_blob_array),0,-1).clip(0,(len(_blob_array)+1)/2))
			
	bboxes = remove_duplicate(blob_array,priority)
	print('Computed %d proposals'%(len(bboxes)))
	np.savetxt(save_name + '.txt',bboxes)
	print('Time taken: %f'%(time.time()-a))

def get_boxes(start_from=0):
	'''Helper function to get all boxes of VOC '''
	filenames = glob.glob('/home/sai/Documents/selective_search/VOC2007/JPEGImages/' + '*.jpg')
	filenames.sort()
	dest_dir = '/home/sai/Documents/selective_search/VOC2007/results/'
	for i,_file in enumerate(filenames):
		if i<start_from: continue
		ssearch_fast(_file,save_name=dest_dir + _file[-10:-4])
		print('%d of %d complete'%(i+1,len(filenames)))
