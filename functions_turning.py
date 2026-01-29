# ===== IMPORTS =====

# Local utilities
from utils import *
from allfitdist import *

# Data handling
import pickle
import pandas as pd

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

# Signal processing and statistics
from scipy.signal import find_peaks
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# Other
import joypy
import numpy as np
import os


# ===== GLOBAL SETTINGS =====

# Default border length (in cm) - excludes turning events this close to arena walls
bl_default = 2


# ===== TURN DETECTION =====

def find_turn_indices(angVels, ht=np.pi/3, eps=0):
	# Detect turning events from angular velocity signal
	# A turn is defined as a continuous period where |angVels| > eps that includes at least one peak > ht
	# Parameters:
	#   angVels: angular velocity time series (radians/second)
	#   ht: angular velocity threshold for detecting prominent turns (radians/second)
	#   eps: baseline threshold for defining "turning period" (lower sensitivity)
	# Returns: (peaks, turn_idxs)
	#   peaks: list of frame indices at peak angular velocity of each turn
	#   turn_idxs: list of (start_frame, end_frame) tuples for each turn

	turn_idxs = []
	peaks = []

	# ===== DETECT RIGHTWARD TURNS (angVels > ht) =====
	exceed_indices = np.where(angVels > ht)[0]
	# Find continuous segments where angVels > eps (loose threshold defines turn window)
	result_segments = indices_grouped_by_condition(angVels, lambda x: x > eps)
	
	for seg in result_segments:
		# Skip very short segments (noise)
		if len(seg) <= 2:
			continue
		turn = (seg[0], seg[-1])
		# Only count as turn if this segment contains a peak > ht
		if is_number_in_interval(exceed_indices, turn):
			turn_idxs.append(turn)
			# Peak is at midpoint of turn
			peaks.append(int((turn[0] + turn[1]) / 2))

	# ===== DETECT LEFTWARD TURNS (angVels < -ht) =====
	exceed_indices = np.where(angVels < -ht)[0]
	# Find continuous segments where angVels < -eps
	result_segments = indices_grouped_by_condition(angVels, lambda x: x < -eps)
	
	for seg in result_segments:
		if len(seg) <= 2:
			continue
		turn = (seg[0], seg[-1])
		if is_number_in_interval(exceed_indices, turn):
			turn_idxs.append(turn)
			peaks.append(int((turn[0] + turn[1]) / 2))

	# ===== SORT TURNS BY TIME =====
	# Pair peaks with turn indices and sort by peak time
	paired_arrays = list(zip(peaks, turn_idxs))
	sorted_paired_arrays = sorted(paired_arrays, key=lambda x: x[0])

	if len(paired_arrays) == 0:
		return [], []
	
	# Unzip sorted arrays
	peaks, turn_idxs = map(list, zip(*sorted_paired_arrays))

	return peaks, turn_idxs


# ===== TURN DATA EXTRACTION =====

def get_turns(groupName, ht=np.pi/3, speed_threshold=None, angle_threshold=None, ignore=None, save_data=False):
	# Extract and characterize all detected turns from a fly group
	# Calculates turn angles, sizes, temperatures, and curvature metrics
	# Parameters:
	#   groupName: fly genotype name
	#   ht: angular velocity threshold for turn detection
	#   speed_threshold: optional (min_speed, max_speed) filter
	#   angle_threshold: minimum turn angle (degrees) to include
	#   ignore: if True, return only turn counts (fast mode for exploratory analysis)
	#   save_data: whether to save results to CSV
	# Returns:
	#   If ignore=True: (turn_counts, in_box_times_in_seconds)
	#   Otherwise: (turns_dataframe, in_box_angles_array, fps)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	
	# Initialize storage for turn metrics
	all_turns = []
	all_peak_times = []
	angle1 = []  # Heading at turn start
	angle2 = []  # Heading at turn end
	turn_lengths = []  # Duration of turn
	in_box_angles = []  # All headings during time spent in arena
	all_temp_diffs = []  # Temperature change during turn
	all_curve = []  # Head trajectory curvature
	all_casts = []  # Casting behavior flag (unused)
	all_fnames = []  # Source filenames

	# ===== PROCESS EACH VIDEO =====
	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		# Load tracking data
		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		speed = np.sqrt((vels[:, 0])**2 + (vels[:, 1])**2)
		angVels = vels[:, 2]

		# ===== SPATIAL FILTERING =====
		# Exclude points near walls (less reliable tracking, arena-specific behavior)
		bl = bl_default
		stageW, stageH = settings['stageW'], settings['stageH']
		inds = (pos[:, 1] > bl) & (pos[:, 1] < (stageH - bl)) & \
		       (pos[:, 0] > bl) & (pos[:, 0] < (stageW - bl))
		
		# Optional: filter by speed range (e.g., "turns during walking")
		if speed_threshold is not None:
			inds = inds & (speed > speed_threshold) & (speed < 5)
		
		actual_inds = np.where(inds)[0]

		# ===== TURN DETECTION =====
		peaks, turn_idxs = find_turn_indices(angVels, ht=ht)
		# Limit to turns away from borders
		turn_idxs = [t for k, t in enumerate(turn_idxs) if peaks[k] in actual_inds]
		peaks = [p for p in peaks if p in actual_inds]

		# ===== HEADING NORMALIZATION =====
		# Reprocess angles to [-π, π] range for circular statistics
		reprocess_angles = (angles + np.pi) % (2 * np.pi) - np.pi
		
		# Fast mode: return only turn counts
		if ignore == True:
			all_turns.append(peaks)
			in_box_angles.append(len(reprocess_angles[inds]) / settings['fps'])
			continue
		
		# Store all headings during time in arena (for normalization)
		in_box_angles.append(reprocess_angles[inds])

		# Casting behavior (currently unused - placeholder for future use)
		cast = np.zeros_like(peaks)

		# ===== EXTRACT TURN DETAILS =====
		for k, turn_idx in enumerate(turn_idxs):
			turn0, turn1 = turn_idx[0], turn_idx[1]
			
			# ===== TURN ANGLE =====
			# Change in heading angle from start to end of turn
			turn_angle = angles[turn1] - angles[turn0]

			# Optional: skip small turns below angle threshold
			if angle_threshold is not None:
				abs_turn_angle = np.abs(turn_angle)
				if abs_turn_angle < angle_threshold / (180 / np.pi):
					continue

			# ===== TEMPERATURE CONTEXT =====
			# Sample temperatures along head trajectory during turn
			X, Y = get_head_loc(pos[turn0:turn1, 0], pos[turn0:turn1, 1], 
			                     angles[turn0:turn1], BL=0.3)
			temps = shallow_field(X)
			temp_diff = temps[-1] - temps[0]

			# ===== HEAD TRAJECTORY CURVATURE =====
			# Measure "crookedness" of head path during turn
			head_loc = np.vstack((X, Y)).T
			curve = curvyness(head_loc)

			# ===== STORE TURN DATA =====
			all_turns.append(turn_angle)
			all_peak_times.append(peaks[k] / settings['fps'])
			angle1.append(reprocess_angles[turn0])
			angle2.append(reprocess_angles[turn1])
			turn_lengths.append((turn1 - turn0) / settings['fps'])
			all_temp_diffs.append(temp_diff)
			all_curve.append(curve)
			all_casts.append(cast[k])
			all_fnames.append(settings['file'])

	# ===== COMPILE DATAFRAME =====
	turns_df = pd.DataFrame({
		'all_turns': all_turns,
		'all_peak_times': all_peak_times,
		'angle1': angle1,
		'angle2': angle2,
		'turn_lengths': turn_lengths,
		'all_temp_diffs': all_temp_diffs,
		'all_curve': all_curve,
		'all_casts': all_casts,
		'all_fnames': all_fnames
	})

	# ===== OPTIONAL: SAVE DATA =====
	if save_data:
		dataDir = f'data/{groupName}/'
		create_directory(dataDir)
		turns_df.to_csv(f"{dataDir}turns_{groupName}.csv", index=False)
	
	if ignore == True:
		return all_turns, in_box_angles

	return turns_df, in_box_angles, settings['fps']


# ===== TURN DIRECTIONALITY ANALYSIS =====

def polar_turns(groupName, ht=np.pi/3, speed_threshold=None, angle_threshold=None, 
                mode=None, casting=False, return_data=None, plot_dir='explore', nbins=4):
	# Analyze directionality of turns: what heading angles lead to left vs right turns?
	# Polar bar plot showing proportion of left/right turns by incoming heading direction
	# Parameters:
	#   groupName: fly genotype name
	#   ht: angular velocity threshold
	#   speed_threshold: optional speed filter
	#   angle_threshold: minimum turn magnitude to include
	#   mode: 'stratify' compares early vs late periods, None shows all data
	#   return_data: if True, return raw proportions without plotting
	#   nbins: 4 or 6 (directional bins: 4=cardinal, 6=30° intervals)
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, 
	                                          angle_threshold=angle_threshold, save_data=True)

	angle1 = np.array(turn_df.angle1)

	# ===== BIN SETUP =====
	min_alpha = 0.2

	if nbins == 4:
		# Cardinal directions (shifted for centering)
		bins = np.linspace(-np.pi/4, 2*np.pi - np.pi/4, nbins + 1)
		angle1 = (angle1 + np.pi/4) % (2*np.pi) - np.pi/4
		alphas = [min_alpha, 1, min_alpha, 1]  # Highlight E-W
	elif nbins == 6:
		# 30° intervals
		bins = np.linspace(0, 2*np.pi, nbins + 1)
		angle1 = (angle1) % (2*np.pi)
		alphas = [min_alpha, 1, min_alpha, min_alpha, 1, min_alpha]  # Highlight N, S

	mid_angles = (bins[1:] + bins[:-1]) / 2
	inds = np.digitize(angle1, bins)

	# ===== DATA PREPARATION =====
	df = turn_df.rename(columns={
	    'angle1': 'Angle1',
	    'all_turns': 'Angle_diff',
	    'all_peak_times': 'peaks'
	})
	df['Angle_diff'] = (180 / np.pi) * df.Angle_diff
	df['inds_Angle1'] = inds

	# ===== OPTIONAL: STRATIFY BY TIME =====
	if mode is None:
		dfs = [df]
		fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
		ax = [ax]
	elif mode == 'stratify':
		bdry = 30  # Boundary time in seconds
		dfs = [df[df.peaks < bdry], df[df.peaks >= bdry]]
		fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))
		ax[0].set_title(f'First {bdry} seconds')
		ax[1].set_title(f'After {bdry} seconds')

	# ===== ANALYSIS FOR EACH SUBPLOT =====
	for i, df in enumerate(dfs):
		pvals = []
		Ns = []
		lCountNorm, rCountNorm = np.zeros(nbins), np.zeros(nbins)
		
		# For each directional bin, count left vs right turns
		for k in range(nbins):
			df_subset = df[df.inds_Angle1 == (k + 1)]
			# Positive angles = left turns, negative = right turns
			pos = df_subset[df_subset['Angle_diff'] > 0].shape[0]
			neg = df_subset[df_subset['Angle_diff'] < 0].shape[0]
			Ns.append([neg, pos + neg])
			
			# Test if proportion differs from 0.5 (equal left/right)
			stat, pval = proportions_ztest(pos, pos + neg, 0.5)
			pvals.append(pval)
			
			# Normalize counts to proportions
			lCountNorm[k] = pos / (pos + neg) if (pos + neg) > 0 else 0
			rCountNorm[k] = neg / (pos + neg) if (pos + neg) > 0 else 0

		if return_data == True:
			plt.close(fig)
			return Ns

		# ===== VISUALIZATION =====
		# Stacked bars: bottom=right (purple), top=left (green)
		p1 = ax[i].bar(mid_angles, rCountNorm, width=(2 * np.pi / nbins), 
		               color=list(zip(['purple'] * nbins, alphas)),
		               edgecolor=list(zip(['black'] * nbins, alphas)), linewidth=1.5)
		p3 = ax[i].bar(mid_angles, lCountNorm, width=(2 * np.pi / nbins),
		               color=list(zip(['green'] * nbins, alphas)), bottom=rCountNorm,
		               edgecolor=list(zip(['black'] * nbins, alphas)), linewidth=1.5)

		# Add sample size labels
		for bar, height, N in zip(mid_angles, pvals, Ns):
			label = rf'$N_R = \frac{{{N[0]}}}{{{N[1]}}}$'
			ax[i].text(bar + 2*np.pi/nbins/4, 1.3, label, ha='center', va='bottom')

		ax[i].legend((p1[0], p3[0]), ('Right', 'Left'), loc='upper left', bbox_to_anchor=(0.85, 1.1))
		ax[i].set_rorigin(-1.0)
		ax[i].set_ylim([0, 1])
		ax[i].set_aspect('equal')
		ax[i].tick_params(axis='y', which='major', labelsize=6.5)

	fig.suptitle(groupName)
	fig.tight_layout()

	# ===== SAVE =====
	default = 'polar_turns_'
	if mode is not None:
		default = default + mode + '_'

	if speed_threshold is not None:
		fig.savefig(outputDir + default + 'speed_thresh_' + groupName + '.pdf', transparent=True)
		fig.savefig(outputDir_png + default + 'speed_thresh_' + groupName + '.png')
	else:
		fig.savefig(outputDir + default + groupName + '.pdf', transparent=True)
		fig.savefig(outputDir_png + default + groupName + '.png')


def polar_ash(groupName, ht=np.pi/3, speed_threshold=None, angle_threshold=None, 
              mode=None, casting=False, plot_dir='explore'):
	# Smoothed version of polar_turns using Average Shifted Histogram
	# Creates smoother estimate of turn directionality across all heading angles
	# Parameters:
	#   groupName: fly genotype name
	#   ht: angular velocity threshold
	#   speed_threshold: optional speed filter
	#   angle_threshold: minimum turn magnitude
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, 
	                                          angle_threshold=angle_threshold)

	df = turn_df.rename(columns={
	    'angle1': 'Angle1',
	    'all_turns': 'Angle_diff',
	})
	df['Angle_diff'] = (180 / np.pi) * df['Angle_diff']

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

	# ===== SEPARATE LEFT AND RIGHT TURNS =====
	pos = np.array(df[df['Angle_diff'] > 0]['Angle1'])  # Left turns
	neg = np.array(df[df['Angle_diff'] < 0]['Angle1'])  # Right turns

	# ===== SMOOTH WITH ASH =====
	n_bins = 8
	m = 8
	pos = circular_ash(pos, n_bins=n_bins, m=m)
	neg = circular_ash(neg, n_bins=n_bins, m=m)
	
	# Normalize to proportion
	rCountNorm = neg / (pos + neg)
	rCountNorm = extend_for_circ(rCountNorm)

	# ===== ANGLE ARRAY =====
	bins = np.linspace(-np.pi, np.pi, n_bins * m + 1)
	mid_angles = (bins[1:] + bins[:-1]) / 2
	mid_angles = extend_for_circ(mid_angles)
	
	# Normalize total height for visual reference
	test = (neg + pos) / np.max(neg + pos)
	test = extend_for_circ(test)

	# ===== VISUALIZATION =====
	# Proportion line
	ax.plot(mid_angles, rCountNorm, color='black', linewidth=1.5)
	# Normalized height line
	ax.plot(mid_angles, test, color='red', linewidth=1.5)
	# Fill regions
	p1 = ax.fill_between(mid_angles, rCountNorm, np.zeros_like(mid_angles), color='purple')
	p2 = ax.fill_between(mid_angles, rCountNorm, np.ones_like(mid_angles), color='green')

	ax.legend((p1, p2), ('Right', 'Left'), loc='upper right')
	ax.set_xlabel('Incoming angle')
	ax.set_rorigin(-1.0)
	ax.set_ylim([0, 1])
	ax.set_yticks(np.arange(5) / 4)
	ax.xaxis.grid(False)
	ax.set_aspect('equal')

	fig.suptitle(groupName)
	fig.tight_layout()

	# ===== SAVE =====
	default = 'polar_ash_'
	if speed_threshold is not None:
		fig.savefig(outputDir + default + 'speed_thresh_' + groupName + '.pdf', transparent=True)
		fig.savefig(outputDir_png + default + 'speed_thresh_' + groupName + '.png')
	else:
		fig.savefig(outputDir + default + groupName + '.pdf', transparent=True)
		fig.savefig(outputDir_png + default + groupName + '.png')


# ===== TURN FREQUENCY ANALYSIS =====

def num_turns(groupName, ht=np.pi/3, speed_threshold=None, angle_threshold=None, 
              mode=None, return_data=None, plot_dir='explore'):
	# Analyze turning frequency normalized by time spent facing each direction
	# Shows if flies turn more frequently in certain orientations
	# Parameters:
	#   groupName: fly genotype name
	#   ht: angular velocity threshold
	#   speed_threshold: optional speed filter
	#   angle_threshold: minimum turn magnitude
	#   mode: 'ash' for smoothed estimate, None for binned
	#   return_data: if True, return data without plotting
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, 
	                                          angle_threshold=angle_threshold)

	angle1 = turn_df.angle1
	in_box_angles = np.concatenate(in_box_angles)

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	if mode == 'ash':
		# ===== SMOOTHED ESTIMATE =====
		n_bins = 6
		m = 6
		# Smooth turn distribution
		angle1_smooth = circular_ash(angle1, n_bins=n_bins, m=m)
		# Smooth time distribution and convert to seconds
		in_box_angles_smooth = circular_ash(in_box_angles, n_bins=n_bins, m=m) / fps
		
		# Turning frequency = turns per second spent in each direction
		ratio = angle1_smooth / in_box_angles_smooth
		ratio = extend_for_circ(ratio)

		bins = np.linspace(-np.pi, np.pi, n_bins * m + 1)
		mid_angles = (bins[1:] + bins[:-1]) / 2
		mid_angles = extend_for_circ(mid_angles)

		if return_data == True:
			plt.close(fig)
			return mid_angles, ratio

		# Plot: turns per minute
		ax.plot(mid_angles, 60 * ratio, color='black', linewidth=1.5)
		ax.set_ylim([0, 50])
		ax.set_title(f'{groupName}\n Number of turns per minute')

	elif mode == None:
		# ===== BINNED ESTIMATE =====
		num_bins = 6
		colors = ['green', 'grey', 'brown', 'brown', 'grey', 'green']
		angle1 = angle1 % (2 * np.pi)
		in_box_angles = in_box_angles % (2 * np.pi)

		# Create bins and digitize
		hist, bins = np.histogram(angle1, bins=num_bins, range=(0, 2 * np.pi))
		inds = np.digitize(angle1, bins)
		inds2 = np.digitize(in_box_angles, bins)
		
		# Calculate turning frequency per bin
		mag = np.array([np.sum(angle1[inds == (i + 1)]) / np.sum(in_box_angles[inds2 == (i + 1)] / fps) 
		                for i in range(num_bins)])

		barbins = bins[:-1] + np.pi / num_bins

		if return_data == True:
			plt.close(fig)
			return barbins, mag

		# Plot: turns per minute
		ax.bar(barbins, 60 * mag, width=0.75 * (2 * np.pi / num_bins), 
		       align="center", color=colors, edgecolor='k')
		ax.set_ylim([0, 50])
		ax.set_title(f'{groupName}\n Number of turns per minute spent moving in each bin')

	# ===== SAVE =====
	message = 'num_turns_'
	if speed_threshold is not None:
		message += 'speed_thresh_'
	if mode is not None:
		message += 'ash_'
	
	fig.tight_layout()
	fig.savefig(outputDir + message + groupName + '.pdf', transparent=True)
	fig.savefig(outputDir_png + message + groupName + '.png')
	fig.clf()
	plt.close(fig)


# ===== TURN MAGNITUDE ANALYSIS =====

def turn_distribution(groupName, ht=np.pi/3, mode='angle', side=False, 
                      speed_threshold=None, angle_threshold=None, plot_dir='explore', return_data=None):
	# Analyze distribution of turn magnitudes (how much do flies turn?)
	# Can show absolute turn sizes or directional (left vs right) turns
	# Parameters:
	#   groupName: fly genotype name
	#   ht: angular velocity threshold
	#   mode: 'angle' (distribution), 'turn_size_binned' (by direction), 'turn_ash' (smooth by direction)
	#   side: if False, show absolute turn magnitudes; if True, show signed turns (left/right)
	#   speed_threshold: optional speed filter
	#   angle_threshold: minimum turn magnitude
	#   return_data: if True, return data without plotting
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, 
	                                          angle_threshold=angle_threshold)

	all_turns = turn_df.all_turns
	angle1 = turn_df.angle1
	
	sns.set_theme(style="ticks", font="Arial", font_scale=1.4)

	# Convert to degrees
	all_turns = (180 / np.pi) * np.array(all_turns)

	# Optionally show absolute magnitude
	if side == False:
		all_turns = np.abs(all_turns)

	# ===== MODE 1: HISTOGRAM WITH DISTRIBUTION FITS =====
	if mode == 'angle':
		fig, ax = plt.subplots()

		# Plot histogram
		sns.histplot(x=all_turns, stat="density", ax=ax, bins=50)

		# Fit multiple distributions and overlay best fits
		fits_df = allfitdist(all_turns, common_cont_dist_names, sortby='BIC')
		x = np.linspace(0, max(all_turns), 1000)

		# Plot top 10 distribution fits
		for _, row in fits_df.head(10).iterrows():
			dist_name = row['Distribution']
			dist = getattr(stats, dist_name)
			fitted_dist = dist(*row['Params'])
			pdf_fitted = fitted_dist.pdf(x)

			# Highlight lognormal fit
			if dist_name == 'lognorm':
				ax.plot(x, pdf_fitted, '-', label=f'{dist_name} distribution fit', 
				        color="dimgray", lw=1.2, alpha=0.8)

		if return_data == True:
			kde_plot = ax.lines[0]
			kde_x = kde_plot.get_xdata()
			kde_y = kde_plot.get_ydata()
			plt.close('all')
			return kde_x, kde_y

		# ===== FORMATTING =====
		ax.set_xlabel('Angle (deg)')
		xlim_max = 360
		if side == False:
			ax.set_xticks(30 * np.arange(13))
			ax.set_xlim([0, xlim_max])
		else:
			ax.set_xticks(30 * (np.arange(11) - 5))
			ax.set_xlim([-xlim_max, xlim_max])

		plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

		# Add median line
		median_val = int(np.median(all_turns))
		ax.axvline(median_val, color="black", linestyle="--", lw=1.5, 
		          label=f"Median = {median_val}°")

		title = f'Angular velocity threshold = {int((180/np.pi)*ht)}°/s'
		ax.set_title(title)
		ax.legend()

	# ===== MODE 2: TURN SIZE BY INCOMING DIRECTION (BINNED) =====
	elif mode == 'turn_size_binned':
		colors = ['green', 'grey', 'brown', 'brown', 'grey', 'green']
		min_alpha = 0.2
		nbins = 6
		bins = np.linspace(0, 2 * np.pi, nbins + 1)
		angle1 = (angle1) % (2 * np.pi)
		alphas = [min_alpha, 1, min_alpha, min_alpha, 1, min_alpha]

		mid_angles = (bins[1:] + bins[:-1]) / 2
		inds = np.digitize(angle1, bins)

		# Median turn size for each incoming direction
		mag = np.array([np.median(all_turns[inds == (i + 1)]) for i in range(nbins)])

		fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
		ax.bar(mid_angles, mag, width=1 * (2 * np.pi / nbins), align="center", color=colors, edgecolor='k')
		ax.set_title('Median turn size per bin')

	# ===== MODE 3: TURN SIZE BY INCOMING DIRECTION (SMOOTHED ASH) =====
	elif mode == 'turn_ash':
		nbins = 6
		m = 6

		# Compute median turn size for each angle using smoothed ASH
		mid_angles, mag = circular_weighted_ash(angle1, all_turns, n_bins=nbins, m=m, agg='median')
		mag = extend_for_circ(mag)
		mid_angles = extend_for_circ(mid_angles)

		if return_data == True:
			return mid_angles, mag

		# ===== PUBLICATION-QUALITY POLAR PLOT =====
		fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
		ax.plot(mid_angles, mag, color='black', linewidth=2.0)
		
		# Subtle fill under curve
		ax.fill_between(mid_angles, 0, mag, color='gray', alpha=0.15)
		
		# ===== STYLING =====
		ax.set_theta_zero_location('E')  # 0° at right
		ax.set_theta_direction(1)  # Clockwise
		ax.set_thetagrids(range(0, 360, 45), 
		                  labels=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
		ax.set_rlabel_position(225)
		
		# Radial grid
		ax.set_rticks(np.linspace(0, 135, 4))
		ax.tick_params(labelsize=10)
		ax.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
		ax.spines['polar'].set_visible(False)

		fig.suptitle(groupName)
		fig.tight_layout()

	# ===== SAVE =====
	if speed_threshold is not None:
		fig.savefig(outputDir + 'turn_distribution_' + mode + '_speed_thresh_' + groupName + '.pdf', transparent=True)
		fig.savefig(outputDir_png + 'turn_distribution_' + mode + '_speed_thresh_' + groupName + '.png')
	else:
		fig.savefig(outputDir + 'turn_distribution_' + mode + '_' + groupName + '.pdf', transparent=True)
		fig.savefig(outputDir_png + 'turn_distribution_' + mode + '_' + groupName + '.png')

	fig.clf()
	plt.close('all')
	sns.reset_orig()
