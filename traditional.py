
# Author: Nihesh Anderson
# File: traditional.py

import cv2
import numpy as np
import scipy.misc

SCALE_FACTOR = 10
BLIND_SPOT_FRACTION = [1.2, 1.5]

def Segment(img):

	ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

	output = np.zeros((img.shape[0], img.shape[1]))

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j][2]>95 and img[i][j][1]>40 and img[i][j][0]>20 and img[i][j][2]>img[i][j][1] and img[i][j][2]>img[i][j][0] and abs(img[i][j][2]-img[i][j][1])>15 and ycrcb_img[i][j][1]>135 and ycrcb_img[i][j][2]>85 and ycrcb_img[i][j][0]>80 and ycrcb_img[i][j][1] <= (1.5862*ycrcb_img[i][j][2])+20 and ycrcb_img[i][j][1] >= (0.3448*ycrcb_img[i][j][2])+76.2069 and ycrcb_img[i][j][1] >= (-4.5652*ycrcb_img[i][j][2])+234.5652 and ycrcb_img[i][j][1] <= (-1.15*ycrcb_img[i][j][2])+301.75 and ycrcb_img[i][j][1]<=(-2.2857*ycrcb_img[i][j][2])+432.85):
				output[i][j] = 255
			else:
				output[i][j] = 0

	return output

def DetectHand(difference):

	cluster = []
	for i in range(difference.shape[0]):
		for j in range(difference.shape[1]):
			if(difference[i][j]):
				cluster.append([i,j])
	return np.mean(cluster, 0)

def Normalise(hand, dim):

	global BLIND_SPOT_FRACTION

	center = dim//2
	mean_shifted = (hand - center)*(BLIND_SPOT_FRACTION)
	out = (center+mean_shifted).astype(int)
	out[0] = min(out[0], dim[0]-1)
	out[1] = min(out[1], dim[1]-1)
	print(out, dim)

	return out

def MarkHand(img, coords):
	out = np.copy(img)
	for i in range(-3,4):
		for j in range(-3,4):
			try:
				out[coords[0]+i][coords[1]+j][2] = 255
			except:
				pass
	return out 

if(__name__ == "__main__"):

	camera = cv2.VideoCapture(0)
	ret, frame1 = camera.read()
	frame1 = scipy.misc.imresize(frame1, np.asarray(frame1.shape)//SCALE_FACTOR)
	frame1 = Segment(frame1)

	while(True):

		ret, frame2 = camera.read()
		frame2 = Segment(scipy.misc.imresize(frame2, np.asarray(frame2.shape)//SCALE_FACTOR))
		
		difference = frame2-frame1
		hand = DetectHand(difference)
		hand = Normalise(hand, np.asarray(frame2.shape))

		difference = np.reshape(difference, (difference.shape[0], difference.shape[1], 1))
		difference = np.repeat(difference, 3, 2)
		
		difference = MarkHand(difference, hand)
		cv2.imshow("Output", difference)

		cv2.waitKey(1000//60)
		frame1 = frame2