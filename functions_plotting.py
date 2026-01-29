# ===== IMPORTS =====

# Local utilities
from utils import *
from functions_tracking import *
from functions_turning import *

# Data handling and plotting
import pickle
import os
import pandas as pd
import itertools

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Signal processing and analysis
from scipy.signal import find_peaks, welch, bessel, filtfilt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ===== GLOBAL SETTINGS =====

# Default body length parameter for head position calculations
bl_default = 2

# Load and prepare colormap from file (represents thermal gradient)
C = np.loadtxt('cmap.txt', dtype='int').astype('float')
cm1 = C / 255.0
cm1 = mpl.colors.ListedColormap(cm1)


# ===== TRAJECTORY VISUALIZATION FUNCTIONS =====

def plot_final(fin, groupName, mode='translational', speed_threshold=None, ht=np.pi/2):
	# Generate a finalized trajectory plot with thermal gradient background (no turn overlays)
	# Useful for clean visualizations of spatial distribution and speed
	# Parameters:
	#   fin: path to pickle file containing tracking data
	#   groupName: fly genotype name (used for output directory)
	#   mode: visualization mode ('translational' shows speed coloring)
	#   speed_threshold: optional filter to show only speeds within range
	#   ht: angular velocity threshold for turn detection (unused in this function)
	
	outputDir = 'tracks_final/' + groupName
	create_directory(outputDir)
	
	# Load tracking data from pickle
	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
	f1.close()

	print('Plotting', fin)

	fig, ax = plt.subplots()
	stageH, stageW = settings['stageH'], settings['stageW']

	# Calculate speed from velocity components
	speed = np.sqrt((vels[:, 0])**2 + (vels[:, 1])**2)
	angVels = vels[:, 2]

	# ===== FILTER BY SPATIAL REGION =====
	# Exclude points near walls to focus on open-field behavior
	bl = bl_default  # Border length in cm
	inds = (pos[:, 1] > bl) & (pos[:, 1] < (stageH - bl)) & (pos[:, 0] > bl) & (pos[:, 0] < (stageW - bl))
	
	# Optional: filter by speed range
	if speed_threshold is not None:
		inds = inds & (speed > speed_threshold) & (speed < 5)

	# ===== RENDER VISUALIZATION =====
	# Display thermal gradient as background
	ax.imshow(t0, extent=[0, stageW, 0, stageH], cmap=cm1, vmin=25, vmax=40.)

	if mode == 'translational':
		# Color points by speed
		scatter = ax.scatter(pos[:, 0], pos[:, 1],
		           c=speed,
		           vmin=0, vmax=2,
		           s=6,
		           cmap='coolwarm', zorder=5)

	# Add vertical lines marking thermal zones
	for i in range(4):
		ax.axvline(7 * (i + 1), linewidth=0.2, zorder=1, color='grey')

	ax.set_xlim([0, stageW])
	ax.set_ylim([0, stageH])
	ax.set_aspect('equal')
	ax.invert_yaxis()
	ax.set_xticks([])
	ax.set_yticks([])

	fig.tight_layout()
	fig.savefig(outputDir + '/' + fin.split('/')[-1].split('.')[0] + '.pdf', transparent=True)
	fig.clf()
	plt.close(fig)


def plot_track(fin, groupName, mode='translational', speed_threshold=None, ht=np.pi/2):
	# Generate detailed trajectory plot with detected turns highlighted and color-coded
	# Left turns shown in yellow, right turns in cyan
	# Parameters:
	#   fin: path to pickle file containing tracking data
	#   groupName: fly genotype name (used for output directory)
	#   mode: visualization mode ('translational' for speed coloring)
	#   speed_threshold: optional filter for speed range
	#   ht: angular velocity threshold for turn detection (radians)
	
	outputDir = 'tracks/' + groupName
	create_directory(outputDir)
	outputDir_png = 'tracks_png/' + groupName
	create_directory(outputDir_png)
	
	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
	f1.close()

	print('Plotting', fin)

	fig, ax = plt.subplots()
	stageH, stageW = settings['stageH'], settings['stageW']

	speed = np.sqrt((vels[:, 0])**2 + (vels[:, 1])**2)
	angVels = vels[:, 2]

	# ===== SPATIAL FILTERING =====
	bl = bl_default
	inds = (pos[:, 1] > bl) & (pos[:, 1] < (stageH - bl)) & (pos[:, 0] > bl) & (pos[:, 0] < (stageW - bl))
	
	if speed_threshold is not None:
		inds = inds & (speed > speed_threshold) & (speed < 5)

	actual_inds = np.where(inds)[0]

	# ===== DETECT AND COLOR-CODE TURNS =====
	# Find peaks in angular velocity (turning events)
	peaks, turn_idxs = find_turn_indices(angVels, ht=ht)
	# Limit to turns away from borders
	turn_idxs = [t for k, t in enumerate(turn_idxs) if peaks[k] in actual_inds]
	peaks = [p for p in peaks if p in actual_inds]

	# Calculate turn angles to determine direction (left vs right)
	turn_angles = np.array([angles[t[1]] - angles[t[0]] for t in turn_idxs])

	# Plot each detected turn with color-coding and head position
	for k, turn_idx in enumerate(turn_idxs):
		turn0, turn1 = turn_idx[0], turn_idx[1]
		pos_segment = pos[turn0:turn1, :]
		angle_segment = angles[turn0:turn1]
		# Calculate head location during turn
		X, Y = get_head_loc(pos_segment[:, 0], pos_segment[:, 1], angle_segment, BL=0.3)
		head_loc = np.vstack((X, Y)).T

		# Color by turn direction
		if turn_angles[k] >= 0:
			ax.plot(pos_segment[:, 0], pos_segment[:, 1], linewidth=0.3, zorder=20, color='yellow')
		else:
			ax.plot(pos_segment[:, 0], pos_segment[:, 1], linewidth=0.3, zorder=20, color='cyan')

		# Draw head trajectory
		ax.plot(head_loc[:, 0], head_loc[:, 1], linewidth=0.1, zorder=20, color='white')
		# Draw body-to-head connections
		for j in range(len(pos_segment)):
			ax.plot([pos_segment[j, 0], head_loc[j, 0]], [pos_segment[j, 1], head_loc[j, 1]],
			        linewidth=0.1, color='white', zorder=20)

	# Sample trajectory for visualization
	skips = pos[::10 * settings['fps'], ]

	# ===== CUSTOM COLORMAP FOR SPEED =====
	custom_colors = plt.cm.tab10.colors[:2] + tuple([plt.cm.tab10.colors[2]]) * 2 + \
	                tuple([plt.cm.tab10.colors[4]]) * 2 + tuple([plt.cm.tab10.colors[3]]) * 2
	custom_cmap = ListedColormap(custom_colors)

	# ===== RENDER VISUALIZATION =====
	# Thermal gradient background
	ax.imshow(t0, extent=[0, stageW, 0, stageH], cmap=cm1, vmin=25, vmax=40.)

	if mode == 'translational':
		scatter = ax.scatter(pos[:, 0], pos[:, 1],
		           c=speed,
		           vmin=0, vmax=2,
		           s=0.8,
		           cmap=custom_cmap)
		
		cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label('speed (cm/s)', rotation=270)

		# Mark sampled positions
		ax.scatter(skips[:, 0], skips[:, 1], color='black', s=1)
		
		# Draw full trajectory path
		ax.plot(pos[:, 0], pos[:, 1], linewidth=0.2, zorder=10, color='black')

	# ===== MARK EXPERIMENTAL GAP (IF PRESENT) =====
	if settings['gap'] > 0:
		gapL, gapR = settings['gapL'], settings['gapR']
		ax.axvline(x=gapL * stageW, linestyle='--', color='black', zorder=1000, dashes=(3, 3), linewidth=1)
		ax.axvline(x=gapR * stageW, linestyle='--', color='black', zorder=1000, dashes=(3, 3), linewidth=1)

	ax.set_xlim([0, stageW])
	ax.set_ylim([0, stageH])
	ax.set_aspect('equal')
	ax.invert_yaxis()
	ax.set_title(f'Time cutoff: {int(settings["startInd"]/settings["fps"])} seconds, Info: {originalTrackingInfo[3]}')

	fig.tight_layout()
	fig.savefig(outputDir + '/' + fin.split('/')[-1].split('.')[0] + '.pdf', transparent=True)
	fig.savefig(outputDir_png + '/' + fin.split('/')[-1].split('.')[0] + '.png', transparent=True)
	fig.clf()
	plt.close(fig)


def plot_track_segmented(fin, groupName, mode='translational', speed_threshold=None, ht=np.pi/2):
	# Generate trajectory plot segmented by behavioral states (walking, turning, stopped, wall-touching)
	# Each behavioral state is color-coded
	# Parameters:
	#   fin: path to pickle file containing tracking data
	#   groupName: fly genotype name
	#   mode: visualization mode
	#   speed_threshold: threshold distinguishing walking from stopping
	#   ht: angular velocity threshold for turn detection
	
	outputDir = 'tracks_test/' + groupName
	create_directory(outputDir)
	outputDir_png = 'tracks_png/' + groupName
	create_directory(outputDir_png)
	
	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
	f1.close()

	fig, ax = plt.subplots()
	stageH, stageW = settings['stageH'], settings['stageW']

	speed = np.sqrt((vels[:, 0])**2 + (vels[:, 1])**2)
	angVels = vels[:, 2]

	# ===== IDENTIFY BEHAVIORAL STATES =====
	# Segment trajectory into walking (1), stopping (0), turning (2), wall-touching (3)
	stopseq = get_stop_seq(speed, 1 / settings['fps'])

	bl = bl_default
	inds = (pos[:, 1] > bl) & (pos[:, 1] < (stageH - bl)) & (pos[:, 0] > bl) & (pos[:, 0] < (stageW - bl))
	
	action_seqs = np.copy(stopseq)
	action_seqs[~inds] = 3  # Mark wall-touching regions

	include_inds = np.where((action_seqs == 0) | (action_seqs == 3), False, True)
	include_inds = np.where(include_inds)[0]

	# ===== DETECT AND MARK TURNS =====
	peaks, turn_idxs = find_turn_indices(angVels, ht=ht)
	turn_idxs = [t for k, t in enumerate(turn_idxs) if peaks[k] in include_inds]
	peaks = [p for p in peaks if p in include_inds]

	# Mark turn regions in action sequence
	for turn_idx in turn_idxs:
		turn0, turn1 = turn_idx[0], turn_idx[1]
		action_seqs[turn0:turn1] = 2

	skips = pos[::10 * settings['fps'], ]
	custom_cmap = ListedColormap(plt.cm.tab10.colors[:4])

	# ===== RENDER VISUALIZATION =====
	ax.imshow(t0, extent=[0, stageW, 0, stageH], cmap=cm1, vmin=25, vmax=40.)

	if mode == 'translational':
		scatter = ax.scatter(pos[:, 0], pos[:, 1],
		           c=action_seqs,
		           s=0.8, vmin=-0.5, vmax=3.5,
		           cmap=custom_cmap)
		
		cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label('behavioral state', rotation=270)
		
		ax.plot(pos[:, 0], pos[:, 1], linewidth=0.2, zorder=10, color='black')

	# ===== MARK EXPERIMENTAL GAP =====
	if settings['gap'] > 0:
		gapL, gapR = settings['gapL'], settings['gapR']
		ax.axvline(x=gapL * stageW, linestyle='--', color='black', zorder=1000, dashes=(3, 3), linewidth=1)
		ax.axvline(x=gapR * stageW, linestyle='--', color='black', zorder=1000, dashes=(3, 3), linewidth=1)

	ax.set_xlim([0, stageW])
	ax.set_ylim([0, stageH])
	ax.set_aspect('equal')
	ax.invert_yaxis()
	ax.set_title(f'Time cutoff: {int(settings["startInd"]/settings["fps"])} seconds, Info: {originalTrackingInfo[3]}')

	fig.tight_layout()
	fig.savefig(outputDir + '/' + fin.split('/')[-1].split('.')[0] + '.pdf', transparent=True)
	fig.clf()
	plt.close(fig)


# ===== TIME SERIES ANALYSIS AND PLOTTING =====

def plot_scalar(fin, groupName, ht=np.pi/3, speed_threshold=0.25, limit=60):
	# Generate multi-panel time series plot showing speed, translational velocity, angular velocity, and heading
	# Each panel shows one 60-second segment; multiple panels tile the entire recording
	# Parameters:
	#   fin: path to pickle file
	#   groupName: fly genotype name
	#   ht: angular velocity threshold for turn detection
	#   speed_threshold: minimum speed for "walking" behavior
	#   limit: segment duration in seconds
	
	outputDir = 'tracks/' + groupName + '/scalars'
	create_directory(outputDir)
	outputDir_png = 'tracks_png/' + groupName + '/scalars'
	create_directory(outputDir_png)
	
	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
	f1.close()

	print('Plotting', fin)

	stageH, stageW = settings['stageH'], settings['stageW']
	speed = np.sqrt((vels[:, 0])**2 + (vels[:, 1])**2)
	angVels = vels[:, 2]

	# Decompose velocity into forward and lateral components
	(transV, slipV) = decomposeVelocity(vels[:, 0], vels[:, 1], angles)

	time = np.arange(len(speed)) / settings['fps']
	reprocess_angles = (angles + np.pi) % (2 * np.pi) - np.pi

	# ===== SPATIAL FILTERING =====
	bl = bl_default
	inds = (pos[:, 1] > bl) & (pos[:, 1] < (stageH - bl)) & (pos[:, 0] > bl) & (pos[:, 0] < (stageW - bl))
	plot_inds = [0 if d == False else None for d in inds]
	actual_inds = np.where(inds)[0]

	# ===== TURN DETECTION =====
	peaks, turn_idxs = find_turn_indices(angVels, ht=ht)
	turn_idxs = [t for k, t in enumerate(turn_idxs) if peaks[k] in actual_inds]
	peaks = [p for p in peaks if p in actual_inds]

	# ===== SEGMENTATION INTO TIME WINDOWS =====
	limit_f = limit * settings['fps']
	time_f = len(speed)
	num_levels = int(time_f // limit_f) + 1 * ((time_f % limit_f) > 0)

	fig, ax = plt.subplots(num_levels, 4, figsize=(12, num_levels * 2))

	if num_levels == 1:
		ax = [ax]

	# ===== PLOT EACH TIME SEGMENT =====
	for i in range(num_levels):
		start = i * limit_f
		end = (i + 1) * limit_f
		time_interval = time[start:end]

		# Plot four behavioral measures
		ax[i][0].plot(time_interval, speed[start:end], linewidth=0.5)
		ax[i][1].plot(time_interval, transV[start:end], linewidth=0.5)
		ax[i][2].plot(time_interval, (180 / np.pi) * angVels[start:end], linewidth=0.5)
		ax[i][3].plot(time_interval, (180 / np.pi) * reprocess_angles[start:end], linewidth=0.5)

		# Highlight wall-touching times (red)
		for j in range(4):
			ax[i][j].plot(time_interval, plot_inds[start:end], 'r', linewidth=0.5)
		
		# ===== FORMATTING =====
		for j in range(4):
			ax[i][j].set_xlabel('time (s)')

		ax[i][0].set_ylabel('speed (cm/s)')
		ax[i][1].set_ylabel('trans vel (cm/s)')
		ax[i][2].set_ylabel('angvel (deg/s)')
		ax[i][3].set_ylabel('angle (deg)')

		# Set axis limits
		for j in range(4):
			ax[i][j].set_xlim([time_interval[0], time_interval[-1]])

		ax[i][0].set_ylim([-0.1, 5])
		ax[i][1].set_ylim([-0.1, 5])
		ax[i][2].set_ylim([-360, 360])
		ax[i][3].set_ylim([-180, 180])
		ax[i][3].set_yticks([-180, -90, 0, 90, 180])

		# Draw detection thresholds as dashed lines
		ax[i][0].axhline(y=speed_threshold, color='r', linestyle='--', linewidth=0.5)
		ax[i][2].axhline(y=(180 / np.pi) * ht, color='r', linestyle='--', linewidth=0.5)
		ax[i][2].axhline(y=-(180 / np.pi) * ht, color='r', linestyle='--', linewidth=0.5)

		# Mark detected turns with X markers
		temp_turn_stuff = [(k, p) for k, p in enumerate(peaks) if start <= p <= end]
		if len(temp_turn_stuff) > 0:
			idx_interval, peaks_interval = zip(*temp_turn_stuff)
			peaks_interval = np.array(peaks_interval)

			ax[i][0].plot(peaks_interval / settings['fps'], speed[peaks_interval], "x", markersize=1.5)
			ax[i][1].plot(peaks_interval / settings['fps'], transV[peaks_interval], "x", markersize=1.5)
			ax[i][2].plot(peaks_interval / settings['fps'], (180 / np.pi) * angVels[peaks_interval], "x", markersize=3)
			ax[i][3].plot(peaks_interval / settings['fps'], (180 / np.pi) * reprocess_angles[peaks_interval], "x", markersize=1.5)
			
			# Highlight turn windows in heading trace
			for k in idx_interval:
				turn0, turn1 = turn_idxs[k][0], turn_idxs[k][1]
				ax[i][3].plot(time[turn0:turn1], (180 / np.pi) * reprocess_angles[turn0:turn1],
				             linewidth=0.5, zorder=20, color='red')

	fig.tight_layout()
	fig.savefig(outputDir + '/' + fin.split('/')[-1].split('.')[0] + '_scalar.pdf', transparent=True)
	fig.savefig(outputDir_png + '/' + fin.split('/')[-1].split('.')[0] + '_scalar.png', transparent=True)
	fig.clf()
	plt.close(fig)


# ===== POPULATION-LEVEL ANALYSIS FUNCTIONS =====

def fly_progression_plot(groupName, plot_dir='explore'):
	# Plot spatial position over time for all flies in a genotype group
	# Useful for visualizing consistency/variability in thermotaxis behavior
	# Parameters:
	#   groupName: fly genotype name
	#   plot_dir: subdirectory for output organization
	
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	fig, ax = plt.subplots()
	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		# Normalize position to 0-1 range across stage width
		normalized_x = pos[:, 0] / settings['stageW']
		ax.plot(np.arange(settings['stopInd'] - settings['startInd']) / settings['fps'], normalized_x)

	ax.set_xlabel('Time(s)')
	ax.set_ylabel('Position(normalized)')

	fig.tight_layout()
	fig.savefig(outputDir + 'fly_progression_' + groupName + '.pdf', transparent=True)
	fig.savefig(outputDir_png + 'fly_progression_' + groupName + '.png', transparent=True)
	fig.clf()
	plt.close(fig)


def distance_reached_plot(groupName, mode=None, plot_dir='explore'):
	# Analyze how far flies traveled to reach specific positions along the gradient
	# Returns box plot of distances with proportion of flies reaching each position
	# Parameters:
	#   groupName: fly genotype name
	#   mode: 'ten' restricts analysis to first 10 minutes
	#   plot_dir: subdirectory for output organization
	
	inputDir = f'outputs/outputs_{groupName}/'
	dirs = os.listdir(inputDir)

	outputDir = f'plots/{groupName}/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = f'plots_png/{groupName}/{plot_dir}/'
	create_directory(outputDir_png)

	sns.set_theme(style="ticks", font="Arial", font_scale=1.4)

	# Define positions along gradient to analyze
	lineDists = np.linspace(0.2, 0.8, 4)
	allLineInds, allCumDists = [], []
	last_quad_df = []

	# ===== PROCESS ALL FLIES =====
	for file in dirs:
		if 'output' not in file:
			continue
		with open(os.path.join(inputDir, file), 'rb') as f:
			pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f)

		# Optional: truncate to 10 minutes
		if mode == 'ten':
			tenMin = int(600 * settings['fps'])
			pos = pos[:tenMin, :]

		# Find when fly crosses each position threshold
		normalized_x = pos[:, 0] / settings['stageW']
		lineInds = [next((i for i, x in enumerate(normalized_x) if x > l), None) for l in lineDists]

		# Calculate cumulative distance traveled
		distances = np.linalg.norm(np.diff(pos, axis=0), axis=1)
		cumulative_distances = np.cumsum(distances)
		lineFirstHitDist = [
			cumulative_distances[i - 1] if i is not None else cumulative_distances[-1]
			for i in lineInds
		]

		last_quad_df.append({"fname": file, "last_qad": lineFirstHitDist[-1]})
		allLineInds.append(lineInds)
		allCumDists.append(lineFirstHitDist)

	# ===== AGGREGATE STATISTICS =====
	percent_reached, distances_reached, med_dist_reached = [], [], []
	numFiles = len(allLineInds)

	for i in range(len(lineDists)):
		count = sum(1 for j in range(numFiles) if allLineInds[j][i] is not None)
		dists = [allCumDists[j][i] for j in range(numFiles)]
		percent_reached.append(count / numFiles)
		distances_reached.append(dists)
		med_dist_reached.append(np.median(dists))

	# ===== PUBLICATION-QUALITY VISUALIZATION =====
	fig, ax1 = plt.subplots(figsize=(5, 4))

	# Primary axis: box plot of distances
	c1 = "#1f77b4"
	sns.boxplot(data=distances_reached, ax=ax1, width=0.4, showfliers=False,
				boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.2),
				medianprops=dict(color='black', linewidth=1.5))
	sns.swarmplot(data=distances_reached, ax=ax1, color=c1, size=4, alpha=0.8)

	ax1.set_xticks(range(len(lineDists)))
	ax1.set_xticklabels([f"{d:.1f}" for d in lineDists])
	ax1.set_xlabel("Normalized position along gradient", labelpad=8)
	ax1.set_ylabel("Distance walked (cm)", color=c1, labelpad=8)
	ax1.tick_params(axis='y', colors=c1)
	ax1.set_ylim(0, 306)
	ax1.spines['top'].set_visible(False)
	ax1.yaxis.grid(True, which='major', color='0.85', linewidth=0.8)
	ax1.xaxis.grid(False)

	# Secondary axis: proportion reaching each position
	c2 = "#d62728"
	ax2 = ax1.twinx()
	ax2.plot(range(len(lineDists)), percent_reached,
			'-o', color=c2, lw=2, markersize=6, alpha=0.9)
	ax2.set_ylabel("Proportion reached", color=c2, labelpad=8)
	ax2.tick_params(axis='y', colors=c2)
	ax2.spines['top'].set_visible(False)
	ax2.grid(False)
	ax2.set_ylim(0, 1.02)

	sns.despine(ax=ax1, right=False)
	fig.suptitle(groupName, fontweight='bold', fontsize=14, y=1.03)
	fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.88)

	# ===== SAVE =====
	prefix = 'dist_reached_tenMin_' if mode == 'ten' else 'dist_reached_'
	fig.savefig(os.path.join(outputDir, f"{prefix}{groupName}.pdf"),
				transparent=True, bbox_inches='tight')
	fig.savefig(os.path.join(outputDir_png, f"{prefix}{groupName}.png"),
				dpi=300, bbox_inches='tight')

	plt.close(fig)
	sns.reset_orig()

	# ===== SAVE DATA FOR STATISTICS =====
	dataDir = f'data/{groupName}/'
	create_directory(dataDir)
	last_quad_df = pd.DataFrame(last_quad_df)
	last_quad_df.to_csv(f"{dataDir}last_quad_{groupName}.csv", index=False)


def heading_index(groupName, speed_threshold=0.25, mode='orientation', plot_dir='explore', return_data=None):
	# Calculate heading index: a measure of directional preference/thermotaxis
	# Higher values indicate stronger preference for moving toward warm/cold zones
	# Parameters:
	#   groupName: fly genotype name
	#   speed_threshold: minimum speed to include in analysis
	#   mode: 'orientation' (cosine of heading), 'angvel' (x-velocity / speed), 'new' (sum-based)
	#   plot_dir: output subdirectory
	#   return_data: if set, return data instead of plotting
	
	inputDir = f'outputs/outputs_{groupName}/'
	dirs = os.listdir(inputDir)

	outputDir = f'plots/{groupName}/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = f'plots_png/{groupName}/{plot_dir}/'
	create_directory(outputDir_png)
	
	dataDir = f'data/{groupName}/'
	create_directory(dataDir)

	hi_list = []
	hi_df = []

	# ===== CALCULATE HEADING INDEX FOR EACH FLY =====
	for file in dirs:
		if 'output' not in file:
			continue

		with open(os.path.join(inputDir, file), 'rb') as f:
			pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f)

		speed = np.sqrt((vels[:, 0])**2 + (vels[:, 1])**2)
		idx = speed > speed_threshold

		if mode == 'orientation':
			# Average of cosine of heading angle (biased toward 0°/360°)
			filtered_angles = angles[idx]
			hi = np.average(np.cos(filtered_angles))
		elif mode == 'angvel':
			# Average of x-velocity normalized by speed
			velsX_filtered = vels[:, 0][idx]
			speed_filtered = speed[idx]
			hi = np.average(velsX_filtered / speed_filtered)
		elif mode == 'new':
			# Ratio of total x-displacement to total distance
			hi = np.sum(vels[:, 0]) / np.sum(speed)

		if np.isnan(hi):
			print('One of the heading indices is nan:', file)
			continue

		hi_list.append(hi)
		hi_df.append({"fname": file, "hi": hi})

	# ===== SAVE DATA =====
	hi_df = pd.DataFrame(hi_df)
	hi_df.to_csv(f"{dataDir}hi_{groupName}.csv", index=False)

	if return_data is not None:
		return hi_list

	# ===== PUBLICATION-QUALITY PLOT =====
	sns.set_theme(style="ticks", font="Arial", font_scale=1.4)

	fig, ax = plt.subplots(figsize=(3, 4))

	# Box + swarm plot
	sns.boxplot(data=[hi_list], ax=ax, width=0.4, showfliers=False,
				boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
				medianprops=dict(color='black', linewidth=2))
	sns.swarmplot(data=[hi_list], ax=ax, color="#ff7f0e", size=6, alpha=0.8)

	ax.set_ylim([-0.5, 1])
	ax.set_xticks([0])
	ax.set_xticklabels([''])
	ax.yaxis.grid(True, color='0.85', linewidth=0.8)
	ax.xaxis.grid(False)

	fig.suptitle(groupName, fontweight='bold', fontsize=14, y=1.03)
	fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.88)

	# Save
	fig.savefig(os.path.join(outputDir, f'heading_index_{groupName}.pdf'), transparent=True, bbox_inches='tight')
	fig.savefig(os.path.join(outputDir_png, f'heading_index_{groupName}.png'), dpi=300, bbox_inches='tight')

	sns.reset_orig()
	plt.close(fig)


# ===== FREQUENCY DOMAIN ANALYSIS =====

def psd(groupName, plot_dir='explore', nperseg=256, detrend=False, mode='arithmetic'):
	# Compute and plot power spectral density of heading angles across all flies
	# Shows dominant oscillation frequencies in turning behavior
	# Parameters:
	#   groupName: fly genotype name
	#   plot_dir: output subdirectory
	#   nperseg: FFT segment length (affects frequency resolution)
	#   detrend: whether to detrend signal before computing PSD
	#   mode: 'arithmetic' or 'geometric' averaging across flies
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	weights = []
	powers = []

	# ===== COMPUTE PSD FOR EACH FLY =====
	for k, file in enumerate(dirs):
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		# Compute Welch power spectral density
		f, pxx = welch(angles, fs=settings['fps'], window='boxcar', nperseg=nperseg, detrend=detrend)
		weights.append(angles.shape[0])
		powers.append(pxx)

	# ===== AGGREGATE PSD ACROSS FLIES =====
	if mode == 'arithmetic':
		# Weighted arithmetic mean
		psd = np.sum(np.array(weights) * np.array(powers).T, axis=1) / np.sum(weights)
	elif mode == 'geometric':
		# Weighted geometric mean (better for log-scale distributions)
		psd = np.exp(np.sum(np.array(weights) * np.log(np.array(powers).T), axis=1) / np.sum(weights))

	# ===== PLOT =====
	fig, ax = plt.subplots()
	ax.plot(f, psd)

	ax.set_yscale('log')
	ax.set_ylim([10**(-4), 10**5])
	ax.set_xlabel('Freq (Hz)')
	ax.set_ylabel('PSD')

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + 'psd_' + groupName + '.pdf', transparent=True)
	fig.savefig(outputDir_png + 'psd_' + groupName + '.png', transparent=True)
	fig.clf()
	plt.close(fig)


def acf(groupName, plot_dir='explore', out_to=10):
	# Compute autocorrelation of angular velocity across flies
	# Identifies temporal structure and characteristic timescale of turning behavior
	# Parameters:
	#   groupName: fly genotype name
	#   plot_dir: output subdirectory
	#   out_to: maximum lag to compute (in seconds)
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	all_angvels = []
	settings_fps = None
	
	# ===== COLLECT ANGULAR VELOCITIES =====
	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		with open(inputDir + file, 'rb') as f:
			pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f)
		all_angvels.append(vels[:, 2])
		if settings_fps is None:
			settings_fps = settings['fps']

	# ===== COMPUTE AUTOCORRELATION =====
	num = int(np.round(settings_fps * out_to))
	corrs = [1]  # Autocorrelation at lag 0 is always 1

	for t in range(1, num + 1):
		p1 = []
		p2 = []
		# Concatenate all lags from all flies
		for angvel in all_angvels:
			p1.extend(list(angvel[:-t]))
			p2.extend(list(angvel[t:]))
		p = np.corrcoef(p1, p2)[0, 1]
		corrs.append(p)

	# Find characteristic timescale (1/e decay)
	corrs_array = np.array(corrs)
	try:
		tau_idx = np.where(corrs_array <= 1 / np.e)[0][0]
		tau_sec = tau_idx / settings_fps
	except IndexError:
		tau_sec = None  # Never reaches 1/e

	# ===== PLOT =====
	fig, ax = plt.subplots()
	lags_sec = np.arange(num + 1) / settings_fps
	ax.plot(lags_sec, corrs, label='Autocorrelation')
	ax.axhline(0, color='red', linewidth=1)
	if tau_sec is not None:
		ax.axvline(tau_sec, color='green', linestyle='--', label=f'1/e decay ≈ {tau_sec:.2f}s')
	ax.set_xlabel('Lags (seconds)')
	ax.set_ylabel('Autocorrelation')
	ax.set_title(groupName)
	ax.legend()
	fig.tight_layout()
	fig.savefig(outputDir + 'autocorr_' + groupName + '.pdf', transparent=True)
	fig.savefig(outputDir_png + 'autocorr_' + groupName + '.png', transparent=True)
	plt.close(fig)