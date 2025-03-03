import numpy as np
import itertools
import os
import scipy
import pickle
from scipy.stats import norm

(x2,t0) = pickle.load(open("contour_datBig_gradient.pkl","rb"),encoding='latin1')

def angle_diff(theta1, theta2):
	diff = np.abs(theta1 - theta2)%(2*np.pi)

	angle_diff = diff + (diff>np.pi)*(2*np.pi - 2*diff)
	# angle_diff = np.pi - np.abs(np.pi-diff)

	return angle_diff

def total_dist(traj):
	distances = np.linalg.norm(traj[1:,] - traj[:-1,], axis=1)
	cumulative_distances = np.cumsum(distances)
	return cumulative_distances[-1]

def curvyness(traj):
	direct_dist = np.linalg.norm(traj[-1,] - traj[0,])
	return total_dist(traj)/direct_dist

def indices_grouped_by_condition(array, condition):

	# Using enumerate to get indices and values
	enumerated_data = list(enumerate(array))
	# Using groupby to group consecutive elements based on the condition
	grouped_data = itertools.groupby(enumerated_data, key=lambda x: condition(x[1]))
	# Filtering out groups where the condition is not met and extracting element & indices
	result = [list(indices) for condition_met, indices in grouped_data if condition_met]
	# Filtering out indices only since each element is (item, index)
	result = [[x[0] for x in segment] for segment in result]

	return result

def create_directory(outputDir):
    CHECK_FOLDER = os.path.isdir(outputDir)
    if not CHECK_FOLDER:
        os.makedirs(outputDir)
        print("Created folder: ", outputDir)
    
def movingaverage(interval, window_size):
	#function to calculate a moving average of a trajectory. 
	#Note: Should only be used with odd window sizes. 
	if np.size(interval)>0:
		window= np.ones(int(window_size))
		averaged = np.convolve(interval, window, 'same')/float(window_size)
		for j in range(0,int(np.round((window_size-1)/2))):
			# print j
			averaged[j] = np.sum(interval[0:2*j+1])/(2.*j + 1.)
		for j in range(int(np.round(-(window_size-1)/2-1)),0):
			# print j,averaged[j:]
			averaged[j] = np.sum(interval[2*(j+1)-1:])/(-2.*(j+1)+1)
	else:
		averaged = []
	return averaged

def is_number_in_interval(array, interval):
    # Check if any number in the array falls within the specified interval.
    # Parameters:
    #     array (numpy.ndarray): Input array of numerical values.
    #     interval (tuple): Tuple containing the lower and upper bounds of the interval.
    # Returns:
    #     bool: True if any number falls within the interval, False otherwise.
    lower_bound, upper_bound = interval
    return np.any((array >= lower_bound) & (array <= upper_bound))

def get_head_loc(x, y, theta, BL):
	# Purpose: returns head locations 
	return x + (BL/2)*np.cos(theta), y + (BL/2)*np.sin(theta)

def shallow_field(x):
	# Purpose: Returns arena temp at given position
	# only depends on x position 
	temp = scipy.interpolate.griddata(x2,t0.squeeze(),x*10)
	return temp

def amplify_array(arr, repeat_times):
	amplified_arr = []
	for num in arr:
		amplified_arr.extend([num] * repeat_times)
	return np.array(amplified_arr)

def circular_ash(data, n_bins, m):

	arr = np.zeros((n_bins*m, m))
	delta = (2*np.pi/n_bins)/m

	original_interval = np.array([-np.pi, np.pi])
	for i in range(m):
		interval = original_interval - i*delta
		cast_to = np.pi + i*delta
		data_shifted = (data + cast_to)%(2*np.pi) - cast_to
		bins = np.linspace(interval[0], interval[1], num = n_bins+1, endpoint=True)
		hist, _ = np.histogram(data_shifted, bins=bins)
		arr[:,i] = np.roll(amplify_array(hist, m), -i)

	return np.average(arr, axis = 1)

def extend_for_circ(arr):
    # Get the first element of the array
    first_element = arr[0]
    # Append the first element to the array
    extended_array = np.concatenate((arr, [first_element]))
    return extended_array

def schmitt_trigger(input, low, high):
	output = 2*np.ones_like(input)
	output = np.where(input < low, 0, output)
	output = np.where(input > high, 1, output)

	indices = indices_grouped_by_condition(output, lambda x: x==2)
	for k,idx in enumerate(indices):
		if idx[0] == 0:
			output[idx[0]:idx[-1]+1] = output[idx[-1]+1]
			continue
		output[idx[0]:idx[-1]+1] = output[idx[0]-1]

	return output

def proportion_conf(count, nobs, alpha):

	proportion = count / nobs
	# Calculate the standard error
	se = np.sqrt(proportion * (1 - proportion) / nobs)
	# Calculate the z-score for the given confidence level
	z = norm.ppf(1 - alpha/2)
	# Calculate the confidence interval
	ci_low = proportion - z * se
	ci_upp = proportion + z * se

	return proportion, ci_low, ci_upp

def mean_direction(data, p=1, weights = None):
    
	if weights == None:
		n = len(data)
		sp = np.sum(np.sin(p*data))
		cp = np.sum(np.cos(p*data))
	else:
		n = np.sum(weights)
		sp = np.sum(weights*np.sin(p*data))
		cp = np.sum(weights*np.cos(p*data))

	rp = np.sqrt(sp**2 + cp**2)
	tp = np.arctan2(sp, cp)

	return rp/n, tp
		
def p_diff_or_not(p, ornot = False):
	if ornot:
		return p - (1-p)
	else:
		return p