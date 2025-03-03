import cv2
import os
import pickle
import matplotlib.pyplot as plt
import random
from utils import *

def display_video(fin, groupName):

	outputDir = 'samples/'+groupName+'/'
	create_directory(outputDir)
	vidDir = 'videos/'+groupName+'/'

	# inputDir = 'outputs/outputs_'+groupName+'/'
	# dirs = os.listdir(inputDir)
	# file = 'gradient_10-22-2020_17-09.output'

	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)  = pickle.load(f1)
	f1.close()

	file = settings['file']
	print('Displaying '+ file)

	# Read the video
	cap = cv2.VideoCapture(vidDir+file)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'H264')
	out = cv2.VideoWriter(outputDir + file.split('.')[0]+'.avi', fourcc, settings['fps'], (settings['width'], settings['height']))  # Adjust resolution as needed

	# print(settings['file'])
	# print(settings['firstInd'])
	# print(pos2.shape)

	for j in range(settings['startInd']+len(angles)):
		ret, frame = cap.read()

		if j< settings['startInd']:
			continue

		if settings['flip'] == 1:
			# Calculate the center of the frame
			center = (frame.shape[1] // 2, frame.shape[0] // 2)

			# Flip the frame around its center by 180 degrees
			matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
			frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


		i = j - settings['startInd']

		x, y = pos[i,:]
		theta = angles[i]
		d = 0.5
		x1, y1 = x+d*np.cos(theta), y+d*np.sin(theta)
		
		x, y = int(np.round(x*settings['scale'])), int(np.round(y*settings['scale']))
		x1, y1 = int(np.round(x1*settings['scale'])), int(np.round(y1*settings['scale']))

		cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
		cv2.line(frame, (x, y), (x1, y1), (0, 0, 255), 1)  # Draw a red line

		# Write the frame into the output video
		out.write(frame)

		# # Display the frame
		# cv2.imshow('Frame', frame)

		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break

	# Release everything if job is finished
	cap.release()
	out.release()
	cv2.destroyAllWindows()


def sample_group(groupName, num=3, seed=10):
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	dirs = [file for file in dirs if (file.split('.')[-1] == 'output')]

	random.seed(seed)
	dirs = random.sample(dirs, num)
	print(dirs)

	for file in dirs:
		display_video(inputDir+file, groupName)

def display_specific(file, groupName):
	if file.split('.')[-1] != 'output':
		file = file.split('.')[0] + '.output'
	inputDir = 'outputs/outputs_' + groupName + '/'
	display_video(inputDir+file, groupName)
