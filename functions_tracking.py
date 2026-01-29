# ===== IMPORTS =====

# Standard library
import os
import time
import datetime
from math import atan2, cos, sin, sqrt, pi
from collections import Counter
import pickle

# Numerical and scientific computing
import numpy as np
import cv2
from scipy.signal import medfilt, savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter1d

# Visualization and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Local imports
from utils import *
from functions_tracking_settings import *


# ===== SIGNAL SMOOTHING AND DERIVATIVES =====

def smooth_and_deriv(x, dt, window_length=9, polyorder=3, sigma=2):
	# Smooth a signal using Savitzky-Golay filter and compute its time derivative
	# Parameters:
	#   x: input signal (1D array)
	#   dt: time step between samples
	#   window_length: length of smoothing window (should be odd)
	#   polyorder: polynomial order for the filter
	#   sigma: standard deviation for Gaussian smoothing of derivative
	# Returns: (smoothed_signal, time_derivative)
	
	# Apply Savitzky-Golay filter to smooth the signal
	xhat = savgol_filter(x, window_length=window_length, polyorder=polyorder)
	# Compute first derivative (slope) using Savitzky-Golay filter
	dxdt = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=1) / dt
	# Additional smoothing of the derivative using Gaussian filter
	dxdt = gaussian_filter1d(dxdt, sigma=sigma, mode='nearest')

	return xhat, dxdt


# ===== ORIENTATION AND ROTATION FUNCTIONS =====

def getOrientation(pts):
	# Calculate the mean position and primary orientation of a point cloud
	# Uses PCA to find the principal component (longest axis)
	# Parameters:
	#   pts: array of points with shape (n_points, 1, 2) containing x,y coordinates
	# Returns: (mean_position, angle_in_radians)
	
	X = pts[:, 0, :]
	# Perform Principal Component Analysis (PCA)
	mean = np.mean(X, axis=0)
	Y = X - mean
	eig, evec = np.linalg.eig(Y.T @ Y)
	# First principal component (direction of maximum variance)
	pc1 = evec[:, np.argmax(eig)]
	# Convert to angle in radians
	angle = atan2(pc1[1], pc1[0])
	return mean, angle

def rotate1(origin, points, angles):
	# Rotate points around an origin point by a specified angle
	# Parameters:
	#   origin: center point for rotation (x, y)
	#   points: array of points to rotate, shape (n_points, 2)
	#   angles: rotation angle(s) in radians (scalar or array)
	# Returns: rotated points (modified in place)
	
	# Translate points to origin
	p0 = points - origin
	# Apply 2D rotation matrix
	points[:, 0] = origin[0] + np.cos(angles) * p0[:, 0] - np.sin(angles) * p0[:, 1]
	points[:, 1] = origin[1] + np.sin(angles) * p0[:, 0] + np.cos(angles) * p0[:, 1]
	return points


# ===== GEOMETRIC ANALYSIS FUNCTIONS =====

def findFarthestDist(pts):
	# Find the maximum distance between any two points in a point cloud
	# Uses ConvexHull optimization for large point sets (>1000 points)
	# Parameters:
	#   pts: array of points with shape (n_points, 2)
	# Returns: maximum pairwise distance
	
	if len(pts) < 1000:
		# For small point sets, compute all pairwise distances
		return np.max(cdist(pts, pts))
	else:
		# For large point sets, use convex hull to reduce computation
		hull = ConvexHull(pts)
		# Get only the points on the convex hull boundary
		hpts = pts[hull.vertices]
		# Compute pairwise distances only between hull points
		return np.max(cdist(hpts, hpts))


# ===== VELOCITY DECOMPOSITION =====

def decomposeVelocity(vx_1, vy_1, denoisedThetas):
	# Decompose translational velocity into forward and lateral (slip) components
	# Transforms from (vx, vy) to (forward_velocity, lateral_velocity) in body frame
	# Parameters:
	#   vx_1: x-component of velocity (array)
	#   vy_1: y-component of velocity (array)
	#   denoisedThetas: heading angles in radians (array, same length as vx_1)
	# Returns: (forward_velocity, lateral_velocity)
	
	transV = np.zeros_like(vx_1)
	slipV = np.zeros_like(vx_1)

	for num in range(len(vx_1)):
		# Velocity vector in lab frame
		trueV = np.array([vx_1[num], vy_1[num]])
		# Unit vector in forward direction (aligned with heading)
		cDir = np.array([np.cos(denoisedThetas[num]), np.sin(denoisedThetas[num])])
		# Unit vector in lateral direction (perpendicular to heading)
		cDir_tang = np.array([-np.sin(denoisedThetas[num]), np.cos(denoisedThetas[num])])
		# Project velocity onto forward and lateral directions
		transV[num] = np.dot(trueV, cDir)
		slipV[num] = np.dot(trueV, cDir_tang)

	return (transV, slipV)


# ===== ANGLE FIXING AND UNWRAPPING =====

def generateFlipIntervals(spots):
	# Create intervals from detected flip/reversal points
	# Parameters:
	#   spots: list of indices where direction reversals occur
	# Returns: list of [start, end] intervals between consecutive spots
	
	flipInt = []
	spots = [0] + spots
	for k in range(len(spots) - 1):
		interval = [spots[k], spots[k + 1] - 1]
		flipInt.append(interval)
	return flipInt

def fixFlips(transV, angles, flipInt):
	# Fix directional ambiguity (±π) by checking if forward velocity is positive
	# If average forward velocity is negative in an interval, flip the heading by π
	# Parameters:
	#   transV: forward velocity component for each frame
	#   angles: heading angles (may be off by π)
	#   flipInt: intervals to check [[start, end], ...]
	# Returns: corrected angles with consistent direction
	
	for fI in flipInt:
		transV_interval = transV[fI[0]:fI[1] + 1]
		mean_vel = np.average(transV_interval)
		# If forward velocity is negative, flip heading by π
		if mean_vel < 0:
			angles[fI[0]:fI[1] + 1] = (angles[fI[0]:fI[1] + 1] + np.pi) % (2 * np.pi)
	return angles

def fixOrientation(angles, velsX, velsY, lowDat):
	# Comprehensively fix heading angles which are ambiguous up to π
	# Uses velocity direction and temporal continuity to resolve 180° ambiguities
	# Parameters:
	#   angles: raw heading angles from orientation detection (ambiguous by π)
	#   velsX, velsY: velocity components (for decomposition)
	#   lowDat: array indicating quality of each frame's tracking
	# Returns: corrected and unwrapped heading angles
	
	# Skip initial NaN values
	firstInd = next((i for i in range(0, angles.shape[0]) if ~np.isnan(angles[i])), None)
	angles[firstInd:] = angles[firstInd:] + 2*np.pi * (angles[firstInd:] < 0)
	angs1 = angles[firstInd:]
	angles = angles[firstInd:]
	lowDat = lowDat[firstInd:]

	# Track where direction reversals occur
	switch_spots = []
	for j in range(1, len(angles) - firstInd):
		# Check angular distance between consecutive frames
		angdiff = angle_diff(angs1[j - 1], angles[j])
		
		# If angles are close, accept the angle
		if angdiff < np.pi / 2:
			angs1[j] = angles[j]
		else:
			# If angles are far, try flipping by π
			angs1[j] = (angles[j] - np.pi) % (2 * np.pi)
			angdiff = np.pi - angdiff

		# Mark large jumps as potential flip points
		if angdiff > 0.25 * (np.pi / 2):
			switch_spots.append(j)
			continue
		
		# Mark changes in tracking quality as potential flip points
		if lowDat[j] != lowDat[j - 1]:
			switch_spots.append(j)
	
	# Use velocity decomposition to fix remaining ambiguities
	(transV, slipV) = decomposeVelocity(velsX, velsY, angs1)
	flipInt = generateFlipIntervals(switch_spots)
	angs1 = fixFlips(transV, np.copy(angs1), flipInt)
	# Remove 2π wraps to create continuous angle trajectory
	angs1 = np.unwrap(angs1)

	return angs1


# ===== VIDEO FRAME PROCESSING =====

def track_video(fin, thresh2percentile=0.7, gap=0, frob_thresh=30000, mode='regular', plotmaxproj=True):
	# Main tracking function: processes a video file and extracts trajectory, angles, and velocities
	# Parameters:
	#   fin: file path to input video
	#   thresh2percentile: percentile threshold for secondary binarization
	#   gap: gap width in experimental setup (used to set ROI boundaries)
	#   frob_thresh: Frobenius norm threshold for detecting arena stability
	#   mode: 'regular', 'testing', or 'middle' - affects processing
	#   plotmaxproj: whether to save maximal projection plots
	# Returns: (position, angles, velocities, settings, originalTrackingInfo, (pos2, angles2, vels2))
	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

	cap = cv2.VideoCapture(fin)
	vidLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Auto-detect fps from video resolution (some videos have incorrect metadata)
	if width > 1100:
		fps = 15
		lowres_flag = False
	elif width < 800:
		fps = 30
		lowres_flag = True

	# Extract metadata from filename
	groupName = fin.split('/')[-2]
	fname = fin.split('/')[-1]
	date = datetime.datetime.strptime(fname.split('_')[-2], "%m-%d-%Y")

	settings = {
		'vidLength': vidLength, 'fps': fps, 'width': width, 'height': height,
		'gap': gap, 'file': fname, 'date': date, 'groupName': groupName
	}
	threshold, perc_threshold = best_settings(groupName, settings)
	print(threshold, perc_threshold)
	settings.update({
		'thresh2perc': thresh2percentile, 'frob_thresh': frob_thresh,
		'threshold': threshold, 'perc_thresh': perc_threshold
	})

	print('Tracking ' + fin + ' with threshold =', threshold)

	# ===== BUILD BACKGROUND IMAGE =====
	# Randomly sample frames to estimate background
	frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
	frames = []
	for fid in frameIds:
		cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
		ret, frame = cap.read()
		frames.append(frame)
	cap.release()

	# Compute median frame as background estimate
	medianFrame = np.percentile(frames, 50, axis=0).astype(dtype=np.uint8)
	grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

	# Normalize frame intensities to have consistent median and deviation across video
	mad = median_abs_deviation(grayMedianFrame.flatten())
	if mad < 2:
		mad = 2
	alpha = 10 / mad
	beta = 50 - alpha * np.median(grayMedianFrame)

	# ===== BUILD FOREGROUND THRESHOLD FRAME =====
	# Resample frames to build foreground threshold background
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

	# Build percentile-based foreground and median backgrounds
	grayPercentileFrame = np.percentile(frames, perc_threshold, axis=0).astype(dtype=np.uint8)
	grayMedianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

	# Save maximal projection (all-frames max) for visualization
	if plotmaxproj:
		maxproj_dir = f'outputs/outputs_{settings["groupName"]}/maxproj/'
		create_directory(maxproj_dir)
		grayMaxFrame = np.percentile(frames, 100, axis=0).astype(dtype=np.uint8)
		fig, ax = plt.subplots()
		ax.imshow(grayMaxFrame, aspect='auto')
		fig.savefig(f'{maxproj_dir}maxproj_{settings["file"].split(".")[0]}.pdf')
		plt.close('all')

	# Debug mode: visualize thresholding
	if mode == 'testing':
		fig, ax = plt.subplots(2, 1)
		ax[0].imshow(mask_lighting(grayPercentileFrame, False, settings), aspect='auto')
		ax[1].imshow(grayPercentileFrame, aspect='auto')
		plt.show()
		plt.close()

	# Set gap-specific ROI boundaries
	if gap == 7:
		settings['gapL'], settings['gapR'] = 0.482, 0.502
	elif gap == 10:
		settings['gapL'], settings['gapR'] = 0.462, 0.492
	elif gap == 20:
		settings['gapL'], settings['gapR'] = 0.447, 0.507

	# ===== PROCESS VIDEO =====
	means, orig_angles, lowDat, cLowDat = process_video(
		fin, settings, grayPercentileFrame, grayMedianFrame, threshold,
		thresh2percentile, frob_thresh, alpha, beta, kernel, lowres_flag, mode
	)

	# Keep original tracking information for reference
	originalTrackingInfo = (means, np.copy(orig_angles), lowDat, cLowDat)

	# Translate to real coordinates, fix angles, and smooth
	pos, angles, vels, settings = translate_output(
		np.copy(means), np.copy(orig_angles), settings, lowDat, mode=mode, include='default'
	)

	# Quality check: if fly doesn't move, try with more lenient threshold
	if not isinstance(pos, float):  # If tracking worked
		if (cLowDat[5] > vidLength / 20) & (np.linalg.norm(pos[0,] - pos[-1,]) < 2):
			print('Fly stays in one spot too long, reprocessing:', settings['file'])
			perc_threshold = 10
			threshold = 50
			grayPercentileFrame = np.percentile(frames, perc_threshold, axis=0).astype(dtype=np.uint8)
			means, orig_angles, lowDat, cLowDat = process_video(
				fin, settings, grayPercentileFrame, grayMedianFrame, threshold,
				thresh2percentile, frob_thresh, alpha, beta, kernel, lowres_flag, mode
			)
			pos, angles, vels, settings = translate_output(
				np.copy(means), np.copy(orig_angles), settings, lowDat, mode=mode, include='default'
			)
			settings.update({
				'thresh2perc': thresh2percentile, 'frob_thresh': frob_thresh,
				'threshold': threshold, 'perc_thresh': perc_threshold
			})

	# Include all data (including endpoints that are normally trimmed)
	pos2, angles2, vels2, settings2 = translate_output(
		np.copy(means), np.copy(orig_angles), settings, lowDat, mode=mode, include='ends'
	)

	return pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)


def process_video(fin, settings, grayPercentileFrame, grayMedianFrame, threshold,
                  thresh2percentile, frob_thresh, alpha, beta, kernel, lowres_flag, mode):
	# Process video frame-by-frame: detect centroid and orientation of tracked object
	# Parameters:
	#   fin: input video file path
	#   settings: tracking settings dictionary
	#   grayPercentileFrame, grayMedianFrame: background reference frames
	#   threshold, thresh2percentile: binarization thresholds
	#   frob_thresh: threshold for detecting arena movement
	#   alpha, beta: intensity normalization parameters
	#   kernel: morphological kernel for cleaning
	#   lowres_flag: whether video is low resolution
	#   mode: processing mode ('regular' or 'testing')
	# Returns: (means, angles, lowDat, cLowDat)
	#   means: (n_frames, 2) centroid positions in pixel coordinates
	#   angles: (n_frames,) heading angles in radians
	#   lowDat: quality code for each frame
	#   cLowDat: Counter of quality codes
	
	cap = cv2.VideoCapture(fin)

	adjustStartFlag = True
	vidLength = settings['vidLength']

	means, angles = np.zeros((vidLength, 2)), np.zeros(vidLength)
	oldmeans, oldangle = np.nan, np.nan
	lowDat = [0] * vidLength

	for j in range(int(vidLength)):
		# Read and normalize frame
		ret, frame = cap.read()
		frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Compute difference from reference background
		dframe = cv2.absdiff(frame, grayPercentileFrame)

		# Check if arena is stable (for early movement detection)
		if (j % (settings['fps'] * 2) == 0) & adjustStartFlag:
			frame_deviation = cv2.absdiff(frame, grayMedianFrame)
			frame_deviation = np.linalg.norm(frame_deviation, ord='fro')
			if mode == 'testing':
				print(j / settings['fps'], frame_deviation)
				continue
			if frame_deviation < frob_thresh:
				adjustStartFlag = False
				print('Adjustment time for', settings['file'], ':', j / settings['fps'])

		# Apply lighting mask (for ablated flies)
		dframe = mask_lighting(dframe, adjustStartFlag, settings)

		# Threshold the difference frame to get binary mask
		threshold2 = thresh2percentile * threshold
		df2 = dframe.copy()
		th, dframe = cv2.threshold(dframe, threshold, 255, cv2.THRESH_BINARY)

		if mode == 'testing':
			if j == 5000:
				plt.imshow(dframe)
				plt.show()
				plt.close()

		nonZs = cv2.findNonZero(dframe)
		nonZs = [] if (nonZs is None) else nonZs

		# Apply morphological opening to clean up noise
		if len(nonZs) > 3:
			if lowres_flag == False:
				dframe = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
			nonZs = cv2.findNonZero(dframe)
		else:
			# If signal is weak, try lower threshold
			th, dframe = cv2.threshold(df2, threshold2, 255, cv2.THRESH_BINARY)
			if lowres_flag == False:
				dframe = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
			nonZs = cv2.findNonZero(dframe)

		# Check if object size is reasonable
		height = settings['height']
		nonZs = [] if (nonZs is None) else nonZs
		if len(nonZs) > 3:
			largest_dist = findFarthestDist(nonZs[:, 0, :])
			bigdist = np.round(height / 30)

			if largest_dist < bigdist:
				# Object size is reasonable, compute position and angle
				mean, angle = getOrientation(nonZs)
				means[j] = mean.squeeze()
				angles[j] = angle
				oldmeans, oldangle = means[j], angles[j]
				lowDat[j] = 4
			else:
				# Object too large, filter to nearby pixels
				nonZs = np.array([z for z in nonZs if np.linalg.norm(oldmeans - np.array(z)) < bigdist])
				if len(nonZs) > 3:
					mean, angle = getOrientation(nonZs)
					means[j] = mean.squeeze()
					angles[j] = angle
					oldmeans, oldangle = mean.squeeze(), angle
				else:
					# Can't recover, use previous position
					means[j] = oldmeans
					angles[j] = oldangle
				lowDat[j] = 5
		else:
			# Not enough pixels, interpolate from previous frame
			means[j] = oldmeans
			angles[j] = oldangle
			lowDat[j] = 6

	cap.release()
	return means, angles, lowDat, Counter(lowDat)


# ===== OUTPUT TRANSLATION AND SMOOTHING =====

def translate_output(means, angles, settings, lowDat, mode, include='default'):
	# Convert raw pixel coordinates to real-world units, fix angle ambiguities, and smooth
	# Parameters:
	#   means: raw centroid positions in pixels
	#   angles: raw heading angles (ambiguous up to π)
	#   settings: tracking settings dictionary
	#   lowDat: tracking quality codes for each frame
	#   mode: 'regular' or 'middle' - affects coordinate system
	#   include: 'default' (trim start/end) or 'ends' (keep all frames)
	# Returns: (smoothPos, denoisedAngles, smoothVels, settings)
	#   smoothPos: (n_frames, 2) smoothed position in cm
	#   denoisedAngles: (n_frames,) smoothed heading angles
	#   smoothVels: (n_frames, 3) velocities [vx, vy, angular_velocity]
	#   settings: updated settings dictionary
	
	imgW, imgH = settings['width'], settings['height']
	dt = 1. / settings['fps']

	# ===== COORDINATE SYSTEM SETUP =====
	# Convert from pixel coordinates to real-world coordinates
	stageW = 35  # Stage width in cm
	scale = imgW / stageW  # Pixels per cm
	stageH = imgH / scale
	settings['scale'] = scale
	settings['stageH'], settings['stageW'] = stageH, stageW

	# Find first valid tracking point
	firstInd = next((i for i in range(0, means.shape[0]) if ~np.isnan(means[i, 0])), None)
	if firstInd == None:
		if include == 'default':
			print('DIDNT FIND ANYTHING FOR ' + settings['file'])
		return np.nan, np.nan, np.nan, settings

	settings['firstInd'] = firstInd
	m0 = means[firstInd, :]
	
	# Find start of actual movement (after initial settling, 1 cm from start)
	rad = 2 * scale
	startInd = next((i for i in range(1, means.shape[0])
	                 if np.linalg.norm(means[i, :] - m0) > rad), None)

	if startInd == None:
		startInd = firstInd

	skip = skip_settings(settings) * settings['fps']
	startInd = max(skip, startInd)

	# ===== COORDINATE SYSTEM ORIENTATION =====
	# Flip coordinates so all trials start from the left side
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
		div = 0.15
		settings['flip'] = 0
		if m0[0] < imgW / 2.:
			# Started on left side, no rotation needed
			cutLine = imgW * (1 - div)
			stopInd = next(
				(i for i in range(1, means.shape[0]) if means[i, 0] > cutLine),
				means.shape[0])
		elif m0[0] >= imgW / 2.:
			# Started on right side, rotate by π
			settings['flip'] = 1
			cutLine = imgW * div
			stopInd = next(
				(i for i in range(1, means.shape[0]) if means[i, 0] < cutLine),
				means.shape[0])
			# Rotate spatial coordinates by 180°
			means = rotate1([imgW / 2., imgH / 2.], means, np.pi)
			# Angle is invariant to π rotation
			if settings['gap'] > 0:
				settings['gapL'], settings['gapR'] = 1 - settings['gapR'], 1 - settings['gapL']

	settings['startInd'] = startInd
	settings['stopInd'] = stopInd

	# Quality check: trajectory must have sufficient length
	if (stopInd - startInd) < 100:
		print('DIDNT FIND ANYTHING FOR ' + settings['file'])
		return np.nan, np.nan, np.nan, settings

	# ===== SMOOTH AND COMPUTE DERIVATIVES =====
	# Convert to real-world units (cm)
	pos = means[firstInd:, :] / scale

	# Smooth position and compute velocity using Savitzky-Golay filter
	smoothX, velsX = smooth_and_deriv(pos[:, 0], dt)
	smoothY, velsY = smooth_and_deriv(pos[:, 1], dt)
	
	# Fix heading angle ambiguities (±π)
	angs = fixOrientation(np.copy(angles), velsX, velsY, lowDat.copy())
	# Smooth heading and compute angular velocity
	denoisedAngles, angVels = smooth_and_deriv(angs, dt, window_length=15)

	smoothPos = np.vstack((smoothX, smoothY)).T
	smoothVels = np.vstack((velsX, velsY, angVels)).T

	# ===== TRIM TO VALID RANGE =====
	# Trim to start/stop indices based on mode
	if include == 'default':
		# Keep only the main movement portion
		denoisedAngles = denoisedAngles[startInd - firstInd: stopInd - firstInd]
		smoothPos = smoothPos[startInd - firstInd: stopInd - firstInd, :]
		smoothVels = smoothVels[startInd - firstInd: stopInd - firstInd, :]
	elif include == 'ends':
		# Keep all frames from start to end of video
		denoisedAngles = denoisedAngles[startInd - firstInd:]
		smoothPos = smoothPos[startInd - firstInd:, :]
		smoothVels = smoothVels[startInd - firstInd:, :]

	return smoothPos, denoisedAngles, smoothVels, settings