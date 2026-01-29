import numpy as np
import itertools
import os
import scipy
import pickle
from scipy.stats import norm

# ===== LOAD DATA =====
# Load pre-computed contour data and temperature values from pickle file
# x2: coordinates of contour points
# t0: temperature values at those coordinates
(x2, t0) = pickle.load(open("contour_datBig_gradient.pkl", "rb"), encoding='latin1')


# ===== TRAJECTORY ANALYSIS FUNCTIONS =====

def total_dist(traj):
	# Calculate total path length traveled along a trajectory
	# traj: array of shape (n_points, 2) with x,y coordinates
	distances = np.linalg.norm(traj[1:,] - traj[:-1,], axis=1)
	# Cumulative sum gives distance traveled up to each point
	cumulative_distances = np.cumsum(distances)
	# Return the total distance (last cumulative value)
	return cumulative_distances[-1]

def curvyness(traj):
	# Measure how curved/winding a trajectory is (ratio of path length to straight-line distance)
	# Higher values indicate more curving/wandering
	# traj: array of shape (n_points, 2) with x,y coordinates
	direct_dist = np.linalg.norm(traj[-1,] - traj[0,])
	return total_dist(traj) / direct_dist


# ===== ANGLE/ANGULAR UTILITIES =====

def angle_diff(theta1, theta2):
	# Calculate the shortest angular difference between two angles (in radians)
	# Always returns a value between 0 and π (the shortest rotation angle)
	diff = np.abs(theta1 - theta2) % (2*np.pi)
	# If difference is greater than π, use the shorter path going the other way
	angle_diff = diff + (diff > np.pi) * (2*np.pi - 2*diff)
	return angle_diff

def get_head_loc(x, y, theta, BL):
	# Calculate the head position of an organism given body center and orientation
	# Assumes head is at distance BL/2 (body length/2) from center in direction theta
	# Parameters:
	#   x, y: center body position
	#   theta: heading direction in radians
	#   BL: body length
	# Returns: (head_x, head_y) coordinates
	return x + (BL/2)*np.cos(theta), y + (BL/2)*np.sin(theta)


# ===== CIRCULAR/ANGULAR STATISTICS =====

def circular_ash(data, n_bins, m):
	# Compute circular Average Shifted Histogram (ASH) - a smooth estimate of circular distribution
	# Uses m shifted histograms to create a smooth density estimate
	# Parameters:
	#   data: angular data in radians
	#   n_bins: number of bins around the circle
	#   m: number of shifted histograms for smoothing (higher = smoother)
	# Returns: smoothed circular histogram values
	
	# Initialize array to hold shifted histograms
	arr = np.zeros((n_bins*m, m))
	# Angular spacing between shifted histograms
	delta = (2*np.pi / n_bins) / m

	original_interval = np.array([-np.pi, np.pi])
	
	# Create m shifted versions of the histogram
	for i in range(m):
		interval = original_interval - i * delta
		cast_to = np.pi + i * delta
		# Shift data to align with current bin boundaries
		data_shifted = (data + cast_to) % (2*np.pi) - cast_to
		# Create bin edges
		bins = np.linspace(interval[0], interval[1], num=n_bins+1, endpoint=True)
		# Compute histogram
		hist, _ = np.histogram(data_shifted, bins=bins)
		# Amplify histogram values and roll to correct position
		arr[:, i] = np.roll(amplify_array(hist, m), -i)

	# Average all shifted histograms for smooth result
	return np.average(arr, axis=1)

def circular_weighted_ash(angles, values, n_bins=36, m=10, agg="mean"):
	"""
	Circular Average-Shifted Histogram (ASH) that computes weighted statistics over angular bins.
	Creates a smooth estimate of how values vary around a circle (e.g., heading direction).

	Parameters
	----------
	angles : array-like
		Angular positions in radians (e.g., in [-π, π) or [0, 2π)).
	values : array-like
		Values corresponding to each angle (e.g., speed, turning rate). Same length as angles.
	n_bins : int, default=36
		Number of bins around the circle.
	m : int, default=10
		Number of shifted histograms for smoothing (higher = smoother).
	agg : {'mean', 'median'}, default='mean'
		Aggregation function to apply within each angular bin.

	Returns
	-------
	theta_centers : np.ndarray
		Angular bin centers in radians (length n_bins * m).
	smoothed_values : np.ndarray
		Smoothed aggregated values for each angular bin.
	"""
	angles = np.asarray(angles)
	values = np.asarray(values)
	arr = np.zeros((n_bins*m, m))
	# Angular spacing between shifted histograms
	delta = (2*np.pi / n_bins) / m

	original_interval = np.array([-np.pi, np.pi])

	# Create m shifted versions of the histogram
	for i in range(m):
		interval = original_interval - i * delta
		cast_to = np.pi + i * delta
		# Shift angles to align with current bin boundaries
		angles_shifted = (angles + cast_to) % (2*np.pi) - cast_to

		# Create bin edges and assign each angle to a bin
		bins = np.linspace(interval[0], interval[1], num=n_bins+1, endpoint=True)
		bin_indices = np.digitize(angles_shifted, bins) - 1
		bin_indices = np.clip(bin_indices, 0, n_bins - 1)

		# Aggregate values per bin (collect all values for each angular bin)
		binned_values = [[] for _ in range(n_bins)]
		for b, val in zip(bin_indices, values):
			binned_values[b].append(val)

		# Compute aggregate statistic for each bin
		if agg == "mean":
			stats = np.array([np.mean(v) if v else 0 for v in binned_values])
		elif agg == "median":
			stats = np.array([np.median(v) if v else 0 for v in binned_values])
		else:
			raise ValueError("agg must be 'mean' or 'median'")

		# Amplify and roll for this shifted histogram
		expanded = np.repeat(stats, m)
		arr[:, i] = np.roll(expanded, -i)

	# Average all shifted histograms for smooth result
	smoothed_values = np.mean(arr, axis=1)
	theta_centers = np.linspace(-np.pi, np.pi, n_bins*m, endpoint=False)
	return theta_centers, smoothed_values

def mean_direction(data, p=1, weights=None):
	# Calculate the mean direction (circular mean) and concentration of angular data
	# Uses vector addition in polar coordinates: mean direction is angle of resultant vector
	# Parameters:
	#   data: angular data in radians
	#   p: harmonic order (p=1 for mean direction, p>1 for higher harmonics)
	#   weights: optional weights for each angle (default: uniform weights)
	# Returns: (concentration, mean_angle)
	#   concentration: strength of directionality (0=random, 1=concentrated)
	#   mean_angle: preferred direction in radians
	
	if weights is None:
		# Unweighted case: sum sine and cosine components
		n = len(data)
		sp = np.sum(np.sin(p*data))
		cp = np.sum(np.cos(p*data))
	else:
		# Weighted case: incorporate weights in calculation
		n = np.sum(weights)
		sp = np.sum(weights * np.sin(p*data))
		cp = np.sum(weights * np.cos(p*data))

	# Magnitude of resultant vector (0 to 1, where 1 = all angles identical)
	rp = np.sqrt(sp**2 + cp**2)
	# Angle of resultant vector (mean direction)
	tp = np.arctan2(sp, cp)

	# Normalize by sample size to get concentration parameter
	return rp / n, tp


# ===== ARRAY MANIPULATION FUNCTIONS =====

def amplify_array(arr, repeat_times):
	# Repeat each element in an array a specified number of times
	# Example: amplify_array([1, 2, 3], 2) returns [1, 1, 2, 2, 3, 3]
	amplified_arr = []
	for num in arr:
		amplified_arr.extend([num] * repeat_times)
	return np.array(amplified_arr)

def extend_for_circ(arr):
	# Extend array for circular plotting by appending the first element at the end
	# Ensures smooth wraparound when plotting circular data
	first_element = arr[0]
	extended_array = np.concatenate((arr, [first_element]))
	return extended_array

def indices_grouped_by_condition(array, condition):
	# Group consecutive indices where the condition is True
	# Example: Find all consecutive segments where value > threshold
	# Parameters:
	#   array: input array
	#   condition: lambda function that returns True/False for each element
	# Returns: list of lists, each inner list contains consecutive indices where condition is True
	
	# Pair each value with its index
	enumerated_data = list(enumerate(array))
	# Group consecutive elements by condition result
	grouped_data = itertools.groupby(enumerated_data, key=lambda x: condition(x[1]))
	# Keep only groups where condition was True
	result = [list(indices) for condition_met, indices in grouped_data if condition_met]
	# Extract just the indices (discard the values)
	result = [[x[0] for x in segment] for segment in result]

	return result


# ===== SMOOTHING AND FILTERING FUNCTIONS =====

def movingaverage(interval, window_size):
	# Calculate a smoothed moving average of a signal/trajectory
	# Uses centered window for interior points, adjusts at boundaries
	# Parameters:
	#   interval: 1D array of values to smooth
	#   window_size: width of smoothing window (should be odd for symmetry)
	# Returns: smoothed array with same length as input
	
	if np.size(interval) > 0:
		# Create uniform kernel for convolution
		window = np.ones(int(window_size))
		# Apply convolution with 'same' to maintain original length
		averaged = np.convolve(interval, window, 'same') / float(window_size)
		
		# Adjust left boundary: use progressively wider windows at the start
		for j in range(0, int(np.round((window_size-1)/2))):
			averaged[j] = np.sum(interval[0:2*j+1]) / (2.*j + 1.)
		
		# Adjust right boundary: use progressively wider windows at the end
		for j in range(int(np.round(-(window_size-1)/2-1)), 0):
			averaged[j] = np.sum(interval[2*(j+1)-1:]) / (-2.*(j+1)+1)
	else:
		averaged = []
	
	return averaged

def schmitt_trigger(input, low, high):
	# Apply a Schmitt trigger (hysteresis) filter to binarize a signal
	# Values below 'low' become 0, above 'high' become 1, in-between stay at previous state
	# Reduces noise and prevents rapid switching
	# Parameters:
	#   input: input signal
	#   low: lower threshold
	#   high: upper threshold
	# Returns: binary output signal (0s and 1s)
	
	# Initialize output with 2s (represents "undefined" state)
	output = 2 * np.ones_like(input)
	# Apply thresholds
	output = np.where(input < low, 0, output)
	output = np.where(input > high, 1, output)

	# Find groups of undefined states and fill with previous/next value
	indices = indices_grouped_by_condition(output, lambda x: x == 2)
	for k, idx in enumerate(indices):
		# Edge case: if undefined group is at the start, use value after the group
		if idx[0] == 0:
			output[idx[0]:idx[-1]+1] = output[idx[-1]+1]
			continue
		# Otherwise use the value before the undefined group
		output[idx[0]:idx[-1]+1] = output[idx[0]-1]

	return output


# ===== SPATIAL AND TEMPERATURE FUNCTIONS =====

def shallow_field(x):
	# Interpolate temperature at a given position in the arena
	# Uses pre-loaded temperature field data (x2, t0)
	# Parameters:
	#   x: position(s) to query (will be scaled by 10)
	# Returns: temperature value(s) at the position(s)
	temp = scipy.interpolate.griddata(x2, t0.squeeze(), x*10)
	return temp


# ===== INTERVAL AND VALIDATION FUNCTIONS =====

def is_number_in_interval(array, interval):
	# Check if any value in an array falls within a specified numerical interval
	# Parameters:
	#   array: input array of numerical values
	#   interval: tuple of (lower_bound, upper_bound)
	# Returns: True if any array value is in [lower_bound, upper_bound], False otherwise
	lower_bound, upper_bound = interval
	return np.any((array >= lower_bound) & (array <= upper_bound))


# ===== STATISTICAL FUNCTIONS =====

def proportion_conf(count, nobs, alpha):
	# Calculate a confidence interval for a binomial proportion
	# Uses normal approximation (standard approach for large samples)
	# Parameters:
	#   count: number of successes
	#   nobs: total number of observations
	#   alpha: significance level (e.g., 0.05 for 95% CI)
	# Returns: (proportion, ci_lower, ci_upper)
	
	# Calculate sample proportion
	proportion = count / nobs
	# Standard error of the proportion
	se = np.sqrt(proportion * (1 - proportion) / nobs)
	# Z-score for given confidence level
	z = norm.ppf(1 - alpha/2)
	# Confidence interval bounds
	ci_low = proportion - z * se
	ci_upp = proportion + z * se

	return proportion, ci_low, ci_upp

def p_diff_or_not(p, ornot=False):
	# Convert a probability to a preference difference metric
	# If ornot=True: returns difference from chance (p - (1-p)) ranging from -1 to 1
	# If ornot=False: returns p unchanged
	# Useful for quantifying preference/bias when p is probability of choosing one option
	if ornot:
		return p - (1 - p)
	else:
		return p


# ===== FILE SYSTEM FUNCTIONS =====

def create_directory(outputDir):
	# Create a directory if it doesn't already exist
	# Prints confirmation message when directory is created
	CHECK_FOLDER = os.path.isdir(outputDir)
	if not CHECK_FOLDER:
		os.makedirs(outputDir)
		print("Created folder: ", outputDir)