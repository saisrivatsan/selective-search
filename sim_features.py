#-----------------------------------------
# Selective Search + RCNN Implementation
# Author: Sai Srivatsa Ravindranath
#-----------------------------------------

""" 
Computes similarity using different features
1. Color Histogram similarity
2. Texture Histogram similarity
3. Size similarity ( Higher value if two blobs are small)
4. Fill similarity ( Higher value if two blobs fit well)
"""

import numpy as np


def color_hist_sim():
	return lambda blob_1, blob_2 : np.minimum(blob_1.color_hist,blob_2.color_hist).sum()

def texture_hist_sim():
	return lambda blob_1, blob_2 : np.minimum(blob_1.texture_hist,blob_2.texture_hist).sum()

def size_sim(shape):
	return lambda blob_1, blob_2 : 1 - (blob_1.blob_size + blob_2.blob_size)*1.0/(shape[0]*shape[1])

def fill_sim(shape):
	return lambda blob_1, blob_2 : 1 - compute_fill(blob_1, blob_2, shape)


def compute_fill(blob_1,blob_2,shape):
	""" Computer Fill

	Parameters
	----------

	blob_1,blob_2 : Blobs for which fill is to be computed

	img : Input image

	Returns
	--------

	fill :  | (BBox \intersection (\complement (blob1 \union blob2))) |  / | img |

	"""


	BBox = [[]]*4
	BBox[0] = min(blob_1.bbox[0],blob_1.bbox[0])
	BBox[1] = min(blob_1.bbox[1],blob_1.bbox[1])
	BBox[2] = max(blob_1.bbox[2],blob_1.bbox[2])
	BBox[3] = max(blob_1.bbox[3],blob_1.bbox[3])

	BBox_size = abs(BBox[0]-BBox[2])*abs(BBox[1]-BBox[3])
	fill = (BBox_size - blob_1.blob_size - blob_2.blob_size)*1.0/(shape[0]*shape[1])
	return fill
