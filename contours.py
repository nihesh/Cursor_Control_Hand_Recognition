
# Author: Nihesh Anderson
# File: traditional.py

import cv2
import numpy as np
import scipy.misc
import pyautogui
from traditional import Segment
from skimage import data, img_as_float
from skimage.segmentation import chan_vese

SCALE_FACTOR = 10
BLIND_SPOT_FRACTION = [1.2, 1.5]

def DetectHand(frame):

	"""
	Write a custom method that accepts a frame from webcam and returns  
	"""
	frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
	frame = Segment(frame)
	frame = frame.astype(np.uint8)
	contours, heirarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	max = -1
	maxc = -1
	for cnt in contours:
		area=cv2.contourArea(cnt) #contour area
		if(area > max):
			max = area
			maxc = cnt
	img = np.zeros(frame.shape)		
	cv2.drawContours(img,[maxc],-1,(255),2)
	return img
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
				out[coords[0]+i][coords[1]+j] = 255
			except:
				pass
	return out 

def MoveCursor(hand, dim):

	"""
	Smoothening has to be done here
	"""

	screenWidth, screenHeight = pyautogui.size()
	out = np.asarray([0,0])

	out[0] = 1+hand[0]*(int(screenWidth/dim[0]))
	out[1] = 1+hand[1]*(int(screenWidth/dim[1]))
	
	print(out)
	pyautogui.moveTo(out[0], out[1])

if(__name__ == "__main__"):

	camera = cv2.VideoCapture(0)

	while(True):

		ret, frame2 = camera.read()

		hand = DetectHand(frame2)
		#hand = Normalise(hand, np.asarray(frame2.shape))

		#MoveCursor(hand, frame2.shape)
		
		#marked = MarkHand(frame2, hand)
		cv2.imshow("Output", hand)

		cv2.waitKey(1000//60)

