#-----------------------------------------
# Selective Search + RCNN Implementation
# Author: Sai Srivatsa Ravindranath
#-----------------------------------------



import time
import copy
import numpy as np

class blob:
	""" 

	Blob : An image region or segment

	Parameters
	----------
	
	blob_idx : Blob Index
	
	blob_size : no of pixels that constitute the blob_size

	bbox : A tight bounding box that encloses the blob_size

	neighbours : blob_idx of the neighbouring blobs

	color_hist : Color histogram of the blob

	texture_hist : Texture histogram of the blob_size

	"""


	def __init__(self,idx,blob_size=None,bbox=None):

		self.blob_idx = idx

		if not blob_size is None:
			self.blob_size = blob_size

		if not bbox is None:
			self.bbox = bbox

		self.neighbours = set()

		self.color_hist = []

		self.texture_hist = []



def get_blob_neighbours(blob_array,segment_mask):

	""" Set the neighbour attribute of blob class

	Parameters
	----------

	blob_array : Array of blobs

	segment_mask : Integer mask indicating segment labels of an image

	Returns
	-------

	neighbour_set : Set of neighbours ordered as tuples

	"""


	idx_neigh = np.where(segment_mask[:,:-1]!=segment_mask[:,1:])
	x_neigh = np.vstack((segment_mask[:,:-1][idx_neigh],segment_mask[:,1:][idx_neigh])).T
	x_neigh = np.sort(x_neigh,axis=1)
	x_neigh = set([ tuple(_x) for _x in x_neigh])

	idy_neigh = np.where(segment_mask[:-1,:]!=segment_mask[1:,:])
	y_neigh = np.vstack((segment_mask[:-1,:][idy_neigh],segment_mask[1:,:][idy_neigh])).T
	y_neigh = np.sort(y_neigh,axis=1)
	y_neigh = set([ tuple(_y) for _y in x_neigh])

	neighbour_set = x_neigh.union(y_neigh)

	for _loc in neighbour_set:
		blob_array[_loc[0]].neighbours.add(_loc[1])
		blob_array[_loc[1]].neighbours.add(_loc[0])
	return neighbour_set
				

def merge_blobs(blob_array,blob_1,blob_2,t):

	""" Merges two blobs and updates the blob_dict

	Parameters 
	-----------

	blob_dict : Dictionary of blobs with their id as key

	blob_id1, blob_id2 : The ids of the blobs to be merged

	t : The id to be assigned to the new blob

	"""

	blob_t = blob(t)

	blob_t.blob_size = blob_1.blob_size + blob_2.blob_size

	blob_t.neighbours = blob_1.neighbours.union(blob_2.neighbours)
	
	for idx in blob_1.neighbours:
		if idx ==t: continue
		blob_array[idx].neighbours.remove(blob_1.blob_idx)
		blob_array[idx].neighbours.add(t)	

	for idx in blob_2.neighbours:
		if idx==t: continue
		blob_array[idx].neighbours.remove(blob_2.blob_idx)
		blob_array[idx].neighbours.add(t)	

	blob_t.neighbours.remove(blob_1.blob_idx)
	blob_t.neighbours.remove(blob_2.blob_idx)

		

	blob_t.bbox = np.empty(4)
	blob_t.bbox[0] = min(blob_1.bbox[0], blob_2.bbox[0])
	blob_t.bbox[1] = min(blob_1.bbox[1], blob_2.bbox[1])
	blob_t.bbox[2] = max(blob_1.bbox[2], blob_2.bbox[2])
	blob_t.bbox[3] = max(blob_1.bbox[3], blob_2.bbox[3])
	
	# Merge color_hist
	blob_t.color_hist = (blob_1.color_hist*blob_1.blob_size + blob_2.color_hist*blob_2.blob_size)/blob_t.blob_size

	# Merge texture_hist
	blob_t.texture_hist = (blob_1.texture_hist*blob_1.blob_size + blob_2.texture_hist*blob_2.blob_size)/blob_t.blob_size

	return blob_t

	
	
