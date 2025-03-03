import numpy as np
import random
import cv2
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import atan2, cos, sin, sqrt, pi
import pickle
from collections import Counter
from scipy.signal import medfilt, savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter1d

import matplotlib.animation as animation
from utils import *
from functions_tracking_settings import *
import datetime
# import pynumdiff.optimize


def smooth_and_deriv(x, dt, window_length = 9, polyorder = 3, sigma = 2):
	xhat = savgol_filter(x, window_length=window_length, polyorder=polyorder) 
	dxdt = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv = 1) / dt
	dxdt = gaussian_filter1d(dxdt, sigma = sigma, mode='nearest')

	# xhat = np.copy(x)
	# dxdt = np.gradient(xhat, dt)

	return xhat, dxdt

def getOrientation(pts):
    X = pts[:,0,:]
    # Perform PCA analysis
    mean = np.mean(X,axis=0)
    Y = X - mean
    eig, evec = np.linalg.eig(Y.T@Y)
    pc1 = evec[:,np.argmax(eig)]
    angle = atan2(pc1[1], pc1[0]) # orientation in radians
    return mean, angle

def rotate1(origin, points, angles):
	# Rotates points around the an "origin" by angle pi
	p0 = points - origin
	points[:,0] = origin[0] + np.cos(angles) * p0[:, 0] - np.sin(angles) * p0[:,1]
	points[:,1] = origin[1] + np.sin(angles) * p0[:, 0] + np.cos(angles) * p0[:,1]
	return points
	
def findFarthestDist(pts):
	if len(pts) < 1000:
		# Compute pairwise distances between points and return max
		return np.max(cdist(pts, pts))
	else:
		# Finds the largest distance between a cloud of points
		# by calculating the convex hull (reduces num of points)
		# then calculates distances between all convexhull points
		hull = ConvexHull(pts)
		hpts=pts[hull.vertices]

		# Compute pairwise distances between points on the convex hull
		# and return the maximum distance
		return np.max(cdist(hpts, hpts))

def decomposeVelocity(vx_1,vy_1,denoisedThetas):
	#projects the translational velocity onto a new coordinate system
	#where rather than x and y velocity, we have forward/backward and left/right slip velocity 
	transV = np.zeros_like(vx_1)
	slipV = np.zeros_like(vx_1)

	for num in range(len(vx_1)):
		trueV = np.array([vx_1[num],vy_1[num]])
		cDir = np.array([np.cos(denoisedThetas[num]),np.sin(denoisedThetas[num])])
		cDir_tang = np.array([-np.sin(denoisedThetas[num]),np.cos(denoisedThetas[num])])
		transV[num] = np.dot(trueV,cDir)
		slipV[num] = np.dot(trueV,cDir_tang)

	return (transV,slipV)

def generateFlipIntervals(spots):
	# function to generate intervals from possible change points
	flipInt = []
	spots = [0] + spots
	for k in range(len(spots)-1):
		interval = [spots[k], spots[k+1]-1]
		flipInt.append(interval)
	return flipInt

def fixFlips(transV, angles, flipInt):
	# take intervals from generateFlipIntervals, checks if transV is 
	# positive on average on those intervals, and flips the direction if it isnt
	for fI in flipInt:
		transV_interval = transV[fI[0]:fI[1]+1]
		mean_vel = np.average(transV_interval)
		if mean_vel<0:
			angles[fI[0]:fI[1]+1] = (angles[fI[0]:fI[1]+1] + np.pi)%(2*np.pi)
	return angles

def fixOrientation(angles,velsX,velsY,lowDat):
	# Function to take angles which are correct up to pi and fix it 

	firstInd = next((i for i in range(0,angles.shape[0]) if ~np.isnan(angles[i])),None) # ignores first few nan values
	angles[firstInd:] = angles[firstInd:] + 2*np.pi*(angles[firstInd:]<0)
	angs1 = angles[firstInd:]
	angles = angles[firstInd:] 
	lowDat = lowDat[firstInd:]

	switch_spots = []
	for j in range(1,len(angles)-firstInd):
		angdiff = angle_diff(angs1[j-1], angles[j])
		if angdiff < np.pi/2:
			angs1[j] = angles[j]
		else:
			angs1[j] = (angles[j]-np.pi)%(2*np.pi)
			angdiff = np.pi-angdiff

		if angdiff > 0.25*(np.pi/2):
			switch_spots.append(j)
			continue
		
		if lowDat[j]!=lowDat[j-1]:
			switch_spots.append(j)
	
	(transV,slipV) = decomposeVelocity(velsX,velsY,angs1)
	flipInt = generateFlipIntervals(switch_spots)
	angs1 = fixFlips(transV, np.copy(angs1), flipInt)
	angs1 = np.unwrap(angs1)

	# if mode=='denoise':
	# 	final_angles = savgol_filter(angs1, window_length=mult*9, polyorder=3) 
	# else:
	# 	final_angles = angs1

	return angs1

def track_video(fin, thresh2percentile=0.7, gap=0, frob_thresh = 30000, mode='regular', plotmaxproj=True):

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

	cap = cv2.VideoCapture(fin)
	vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	width  = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
	height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Fix fps because some of the videos are messed up
	if width > 1100 :
		fps = 15
		lowres_flag = False
	elif width < 800:
		fps = 30
		lowres_flag = True

	# print(width,height)
	groupName = fin.split('/')[-2]
	fname = fin.split('/')[-1]
	date = datetime.datetime.strptime(fname.split('_')[-2], "%m-%d-%Y")

	settings = {'vidLength': vidLength, 'fps': fps, 'width': width, 'height': height, 'gap': gap, 'file': fname, 'date': date,'groupName': groupName} 
	threshold, perc_threshold = best_settings(groupName, settings)
	print(threshold, perc_threshold)
	settings.update({'thresh2perc': thresh2percentile, 'frob_thresh': frob_thresh, 'threshold': threshold, 'perc_thresh': perc_threshold})

	print('Tracking ' + fin + ' with threshold =', threshold)

	### randomly sample frames, calculate background using min filter. 
	frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
	frames = []
	for fid in frameIds:
		cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
		ret, frame = cap.read()
		frames.append(frame)
	cap.release()

	medianFrame = np.percentile(frames, 50, axis=0).astype(dtype=np.uint8)  
	grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

	# adjust so that all frames have same median and deviation
	mad = median_abs_deviation(grayMedianFrame.flatten())
	if mad < 2:
		mad = 2
	alpha = 10/mad
	beta = 50 - alpha*np.median(grayMedianFrame)

	# print(mad, beta)

	### Resample 
	cap = cv2.VideoCapture(fin)
	frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=200)
	frames = []
	for fid in frameIds:
		cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		frames.append(adjusted)
	cap.release()


	# write proper comments
	grayPercentileFrame = np.percentile(frames, perc_threshold, axis=0).astype(dtype=np.uint8)
	grayMedianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

	# Plot maximal projection 
	if plotmaxproj == True:
		maxproj_dir = f'outputs/outputs_{settings["groupName"]}/maxproj/'
		create_directory(maxproj_dir)
		grayMaxFrame = np.percentile(frames, 100, axis=0).astype(dtype=np.uint8)
		fig, ax = plt.subplots()
		ax.imshow(grayMaxFrame, aspect='auto')
		fig.savefig(f'{maxproj_dir}maxproj_{settings["file"].split(".")[0]}.pdf')
		plt.close('all')


	if mode == 'testing':
		fig, ax = plt.subplots(2, 1)
		ax[0].imshow(mask_lighting(grayPercentileFrame, False, settings), aspect='auto')
		ax[1].imshow(grayPercentileFrame, aspect='auto')
		plt.show()
		plt.close()

		# print(np.min(grayMedianFrame), np.max(grayMedianFrame), np.median(grayMedianFrame), median_abs_deviation(grayMedianFrame.flatten()))
		# plt.hist(grayMedianFrame.flatten(), bins=25, alpha = 0.5)
		# plt.show()

		# varFrame =  np.var(frames, axis=0)
		# np.var([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames], axis=0)
		# grayVarianceFrame = np.var([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames], axis=0)
	
	if gap == 7:
		settings['gapL'], settings['gapR'] = 0.482, 0.502
	elif gap == 10:
		settings['gapL'], settings['gapR'] = 0.462, 0.492
	elif gap == 20:
		settings['gapL'], settings['gapR'] = 0.447, 0.507

	# Process actual video
	means, orig_angles, lowDat, cLowDat = process_video(fin, settings, grayPercentileFrame, grayMedianFrame, threshold, thresh2percentile, frob_thresh, alpha, beta, kernel, lowres_flag, mode)

	# Keep original tracking information
	originalTrackingInfo = (means, np.copy(orig_angles), lowDat, cLowDat)

	# Change tracking to real coordinates, fix angles, smooth stuff
	pos, angles, vels, settings = translate_output(np.copy(means), np.copy(orig_angles), settings, lowDat, mode = mode, include = 'default')

	if not isinstance(pos, float):	# If Tracking did worked
		if (cLowDat[5] > vidLength/20) & (np.linalg.norm(pos[0,] - pos[-1,]) < 2):
			print('Fly stays in one spot too long, reprocessing:', settings['file'])
			perc_threshold = 10
			threshold = 50
			grayPercentileFrame = np.percentile(frames, perc_threshold, axis=0).astype(dtype=np.uint8)
			means, orig_angles, lowDat, cLowDat = process_video(fin, settings, grayPercentileFrame, grayMedianFrame, threshold, thresh2percentile, frob_thresh, alpha, beta, kernel, lowres_flag, mode)
			pos, angles, vels, settings = translate_output(np.copy(means), np.copy(orig_angles), settings, lowDat, mode = mode, include = 'default')
			settings.update({'thresh2perc': thresh2percentile, 'frob_thresh': frob_thresh, 'threshold': threshold, 'perc_thresh': perc_threshold})

	# Include all data
	pos2, angles2, vels2, settings2 = translate_output(np.copy(means), np.copy(orig_angles), settings, lowDat, mode = mode, include = 'ends')

	return pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)

def process_video(fin, settings, grayPercentileFrame, grayMedianFrame, threshold, thresh2percentile, frob_thresh, alpha, beta, kernel, lowres_flag, mode):

	cap = cv2.VideoCapture(fin)

	adjustStartFlag = True
	vidLength = settings['vidLength']

	means,angles = np.zeros((vidLength,2)),np.zeros(vidLength)
	oldmeans, oldangle = np.nan, np.nan
	lowDat = [0]*vidLength

	for j in range(int(vidLength)):
		#read in. convert frame to grayscale
		ret, frame = cap.read()

		# frame adjust 
		frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		
		# convert grayscale
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		## calculate difference between frame and bg frame
		dframe = cv2.absdiff(frame, grayPercentileFrame)

		# for videos with early movement of arena
		if (j%(settings['fps']*2)==0) & adjustStartFlag:
			frame_deviation = cv2.absdiff(frame, grayMedianFrame)
			frame_deviation = np.linalg.norm(frame_deviation, ord = 'fro')
			if mode == 'testing':
				print(j/settings['fps'], frame_deviation)
				continue
			if frame_deviation < frob_thresh:
				adjustStartFlag = False
				print('Adjustment time for' , settings['file'], ':', j/settings['fps'])

		# mask for ablated. removes flickering light. 
		dframe = mask_lighting(dframe, adjustStartFlag, settings)

		# threshold picture
		threshold2 = thresh2percentile*threshold
		df2 = dframe.copy()
		th, dframe = cv2.threshold(dframe, threshold, 255, cv2.THRESH_BINARY)

		if mode == 'testing':
			if j == 5000:
				plt.imshow(dframe)
				plt.show()
				plt.close()
    
		nonZs = cv2.findNonZero(dframe)
		nonZs = [] if (nonZs is None) else nonZs
		
		# pass through open morphology
		# if doesnt work, try again on lower threshold and and see if it does
		if len(nonZs) > 3:
			if lowres_flag == False:
				dframe = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
			nonZs = cv2.findNonZero(dframe)
			#lowDat[j] = 1
		else:
			th, dframe = cv2.threshold(df2, threshold2, 255, cv2.THRESH_BINARY)
			if lowres_flag == False:
				dframe = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
			nonZs = cv2.findNonZero(dframe)
			#lowDat[j] = 2


		# checking if distances are too far 
		height = settings['height']
		nonZs = [] if (nonZs is None) else nonZs
		if len(nonZs)>3:
			largest_dist = findFarthestDist(nonZs[:,0,:])
			bigdist = np.round(height/30)
			
			if largest_dist<bigdist:
				mean,angle = getOrientation(nonZs)
				means[j] = mean.squeeze()
				angles[j] = angle
				# if ~np.isnan(oldangle):
				# 	if np.linalg.norm(means[j]-oldmeans)>20:
				# 		# means[j],angles[j] = oldmeans,oldangle
				# 		plt.imshow(dframe)
				# 		plt.show()
				oldmeans,oldangle = means[j],angles[j]
				lowDat[j] = 4
			else:
				nonZs = np.array([z for z in nonZs if np.linalg.norm(oldmeans-np.array(z))< bigdist])
				if len(nonZs)>3:
					mean,angle = getOrientation(nonZs)
					means[j] = mean.squeeze()
					angles[j] = angle
					oldmeans,oldangle = mean.squeeze(),angle
				else:
					means[j] = oldmeans
					angles[j] = oldangle 
				lowDat[j] = 5
				#print('Error! distance is too large: ',largest_dist, 'frame:', j,'\n')
		else:
			#not enough info in the current frame. keep everything as it was. 
			means[j] = oldmeans
			angles[j] = oldangle 
			lowDat[j] = 6

	return means, angles, lowDat, Counter(lowDat)


def translate_output(means, angles, settings, lowDat, mode, include = 'default'):
	# Function translates 

	imgW, imgH = settings['width'], settings['height']
	dt = 1. / settings['fps']

	# Translate information to real length units
	stageW = 35
	scale = imgW / stageW  # how many pixels per cm
	stageH = imgH / scale
	settings['scale'] = scale
	settings['stageH'], settings['stageW'] = stageH, stageW

	firstInd = next((i for i in range(0,means.shape[0]) if ~np.isnan(means[i,0])),None)
	if firstInd == None:
		if include == 'default':
			print('DIDNT FIND ANYTHING FOR ' + settings['file'])
		return np.nan, np.nan, np.nan, settings
	
	settings['firstInd'] = firstInd
	m0 = means[firstInd, :]
	rad = 2 * scale  # startInd is after moving 1 cm from firstInd
	startInd = next((i for i in range(1, means.shape[0])
	                 if np.linalg.norm(means[i, :] - m0) > rad), None)

	if startInd == None:
		startInd = firstInd
	
	skip = skip_settings(settings)*settings['fps']
	# print(skip, startInd)
	startInd = max(skip, startInd)

	if mode == 'middle':
		div = 0.15
		day = int(settings['file'].split('_')[-2].split('-')[-2])
		stopInd = means.shape[0]
		if day == 17:
			settings['flip'] = 1
			means = rotate1([imgW / 2., imgH / 2.], means, np.pi)
		elif day == 26:
			settings['flip'] = 0

	else:
		# flip so all videos start from the left
		div = 0.15
		settings['flip'] = 0
		if m0[0] < imgW / 2.:
			cutLine = imgW * (1 - div)
			stopInd = next(
				(i for i in range(1, means.shape[0]) if means[i, 0] > cutLine),
				means.shape[0])
		elif m0[0] >= imgW / 2.:
			settings['flip'] = 1
			cutLine = imgW * div
			stopInd = next(
				(i for i in range(1, means.shape[0]) if means[i, 0] < cutLine),
				means.shape[0])
			#rotate points by 180 degrees.
			means = rotate1([imgW / 2., imgH / 2.], means, np.pi)
			# no need to rotate angles since it's invariant to rotation pi
			if settings['gap'] > 0:
				settings['gapL'], settings['gapR'] = 1 - settings['gapR'], 1 - settings['gapL']

	settings['startInd'] = startInd
	settings['stopInd'] = stopInd

	if (stopInd - startInd) < 100:
		print('DIDNT FIND ANYTHING FOR ' + settings['file'])
		return np.nan, np.nan, np.nan, settings

	pos = means[firstInd:, :] / scale

	# smoothing 
	# mult = int(settings['fps']/15)
	# smoothX = savgol_filter(pos[:,0], window_length= mult*9, polyorder=3)
	# smoothY = savgol_filter(pos[:,1], window_length= mult*9, polyorder=3)
	# velsX = np.gradient(smoothX, dt)
	# velsY = np.gradient(smoothY, dt)

	# velsY = movingaverage(np.gradient(smoothY, dt), 5)

	# denoisedAngles = fixOrientation(np.copy(angles), movingaverage(velsX,mult*5), movingaverage(velsY,mult*5), mult=mult)
	# angVels = np.gradient(denoisedAngles, dt)

	smoothX, velsX = smooth_and_deriv(pos[:,0], dt)
	smoothY, velsY = smooth_and_deriv(pos[:,1], dt)
	angs = fixOrientation(np.copy(angles), velsX, velsY, lowDat.copy())
	denoisedAngles, angVels = smooth_and_deriv(angs, dt, window_length=15)

	smoothPos = np.vstack((smoothX, smoothY)).T
	smoothVels = np.vstack((velsX, velsY, angVels)).T

	# Track only starting and ending
	if include == 'default':
		denoisedAngles = denoisedAngles[startInd - firstInd : stopInd - firstInd]
		smoothPos = smoothPos[startInd - firstInd : stopInd - firstInd, :]
		smoothVels = smoothVels[startInd - firstInd : stopInd - firstInd, :]
	elif include == 'ends':
		denoisedAngles = denoisedAngles[startInd - firstInd :]
		smoothPos = smoothPos[startInd - firstInd : , :]
		smoothVels = smoothVels[startInd - firstInd : , :]

	# Corresponds to giving all points (include==None)
	return smoothPos, denoisedAngles, smoothVels, settings