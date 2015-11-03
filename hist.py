#-----------------------------------------
# Selective Search + RCNN Implementation
# Author: Sai Srivatsa Ravindranath
#-----------------------------------------

""" Computes Color and Texture Histograms for blobs """

import numpy as np
import skimage.filters
import scipy.ndimage.filters
from sklearn.preprocessing import normalize

def get_color_hist(img,segment_mask,n_bins=25):
	''' 
	Computes color histograms for all the blobs
	parameters
	----------

	img : Input Image

	segment_ mask :  Integer mask indicating segment labels of an image

	returns
	-------
	
	hist : color_histogram of the blobs. Shape: [ n_segments , n_bins*n_color_channels ]
	'''
	if img.max()>1:	_img = img/255.0
	else: _img = img
	n_segments = len(set(segment_mask.flatten()))
	bins = np.linspace(0.0,1.0,n_bins+1)
	labels = range(n_segments + 1)
	bins = [labels, bins]
	hist = np.hstack([ np.histogram2d(segment_mask.flatten(), _img[:, :, i].flatten(), bins=bins)[0] for i in range(img.shape[-1]) ])
	hist = normalize(hist,norm='l1',axis=1)
	
	return hist

def get_texture_hist(img,segment_mask,n_orientation = 8, n_bins = 10):
	''' 
	Computes texture histograms for all the blobs
	parameters
	----------

	img : Input Image

	segment_ mask :  Integer mask indicating segment labels of an image

	returns
	-------
	
	hist : texture histogram of the blobs. Shape: [ n_segments , n_bins*n_orientations*n_color_channels ]
	'''
	filt_img = skimage.filters.gaussian_filter(img, sigma = 1.0, multichannel = True).astype(np.float32)
	op = np.array([[-1.0, 0.0, 1.0]])
	grad_x = np.array([scipy.ndimage.filters.convolve(filt_img[:,:,i], op) for i in range(img.shape[-1])])
	grad_y = np.array([scipy.ndimage.filters.convolve(filt_img[:,:,i], op.T) for i in range(img.shape[-1])])
	_theta = np.arctan2(grad_y, grad_y)
	theta = np.zeros(img.shape)
	for i in range(img.shape[-1]):theta[:,:,i] = _theta[i]
	n_segments = len(set(segment_mask.flatten()))
	labels = range(n_segments + 1)	
	bins_orientation = np.linspace(-np.pi, np.pi, n_orientation + 1)
	bins_intensity = np.linspace(0.0, 1.0, n_bins + 1)
	bins = [labels, bins_orientation, bins_intensity]
	_temp = [ np.vstack([segment_mask.flatten(), theta[:,:,i].flatten(), filt_img[:,:,i].flatten()]).T for i in range(img.shape[-1])]
	hist = np.hstack([ np.histogramdd(_temp[i], bins = bins)[0] for i in range(img.shape[-1]) ])
	hist = np.reshape(hist,(n_segments,n_orientation*n_bins*img.shape[-1]))
	hist = normalize(hist,norm='l1',axis=1)
	return hist
