from utils import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joypy
from scipy.signal import find_peaks
from matplotlib.cm import get_cmap
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from allfitdist import *

bl_default = 2

def peaks_verification(groupName, speed_threshold = None, plot_dir = 'explore'):
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	hts = (np.pi/3)*(np.arange(6)+1)
	timewindows = np.arange(7) + 1
	cmap = get_cmap('viridis')

	fig, ax = plt.subplots(1,len(hts), figsize = (len(hts)*5,4))
	
	for i in range(len(hts)):
		data = []
		for file in dirs:
			if 'output' not in file.split('/')[-1]:
				continue

			f1 = open(inputDir + file, 'rb')
			pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
			f1.close()

			speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
			angVels = vels[:, 2]

			# only keep turns not touching wall
			bl = bl_default	# border length
			stageW, stageH = settings['stageW'], settings['stageH']
			inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl))
			if speed_threshold is not None:
				inds = inds & (speed>speed_threshold) & (speed<5)
			actual_inds = np.where(inds)[0]

			peaks, _ = find_peaks(np.abs(angVels), height=hts[i])
			peaks =[p for p in peaks if p in actual_inds]
			for j in range(len(timewindows)):
				tm = timewindows[j]
				for p in peaks:
					if (p-tm < 0) or p+tm >= len(angles) :
						continue
					turn = angles[p+tm] - angles[p-tm]
					data.append([(180/np.pi)*turn, tm])

		df = pd.DataFrame(data, columns=['Turn', 'TimeWindow'])
		ax[i].set_title(f'>{np.round((180/np.pi)*hts[i])} deg/s')
		for k in timewindows:
			sns.kdeplot(data=df[df['TimeWindow']==k], x="Turn", ax = ax[i], color=cmap(k/ len(timewindows)) )

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + 'peaks_verification_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'peaks_verification_'+groupName+'.png')
	fig.clf()
	plt.close(fig)

def find_turn_indices(angVels, ht=np.pi/3, eps=0):

	turn_idxs = []
	peaks = []

	exceed_indices = np.where(angVels > ht)[0]
	result_segments = indices_grouped_by_condition(angVels, lambda x: x > eps)
	for seg in result_segments:
		if len(seg)<=2:
			continue
		turn = (seg[0], seg[-1])
		if is_number_in_interval(exceed_indices, turn):
			turn_idxs.append(turn)
			peaks.append(int((turn[0]+turn[1])/2))

	exceed_indices = np.where(angVels < -ht)[0]
	result_segments = indices_grouped_by_condition(angVels, lambda x: x < -eps)
	for seg in result_segments:
		if len(seg)<=2:
			continue
		turn = (seg[0], seg[-1])
		if is_number_in_interval(exceed_indices, turn):
			turn_idxs.append(turn)
			peaks.append(int((turn[0]+turn[1])/2))

	# Zip the arrays together
	paired_arrays = list(zip(peaks, turn_idxs))
	
	# Sort the paired arrays based on the first array
	sorted_paired_arrays = sorted(paired_arrays, key=lambda x: x[0])

	if len(paired_arrays) == 0:
		return [], []
	# Unzip the sorted paired arrays
	peaks, turn_idxs = map(list, zip(*sorted_paired_arrays))

	return peaks, turn_idxs

def get_turns(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, ignore = None, save_data = False):
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	
	all_turns = []
	all_peak_times = []
	angle1 = []
	angle2 = []
	turn_lengths = []
	in_box_angles = []
	all_temp_diffs = []
	all_curve = []
	all_casts = []
	all_fnames = []

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angVels = vels[:, 2]

		# only keep turns not touching wall
		bl = bl_default	# border length
		stageW, stageH = settings['stageW'], settings['stageH']
		inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl))
		if speed_threshold is not None:
			inds = inds & (speed>speed_threshold) & (speed<5)
		actual_inds = np.where(inds)[0]		

		# Find turns using angVels
		peaks, turn_idxs = find_turn_indices(angVels, ht = ht)
		# Limit to peaks that are not too close to border
		turn_idxs = [t for k,t in enumerate(turn_idxs) if peaks[k] in actual_inds]
		peaks = [p for p in peaks if p in actual_inds]

		reprocess_angles = (angles + np.pi)%(2*np.pi) - np.pi
		if ignore == True:
			all_turns.append(peaks)
			in_box_angles.append(len(reprocess_angles[inds])/settings['fps'])
			continue
		# To be used for normalizing to amount of time spent facing each direction
		in_box_angles.append(reprocess_angles[inds])

		# casting
		# thresh = np.pi/6
		# turn_angles = np.array([angles[t[1]] - angles[t[0]] for t in turn_idxs])
		# cast1 = list(np.abs(turn_angles[:-1] + turn_angles[1:]) < thresh) + [False]
		# cast2 = [False] + list(np.abs(turn_angles[:-1] + turn_angles[1:]) < thresh)
		# cast3 = np.abs(turn_angles) < np.pi/3
		# cast = (np.array(cast1) | np.array(cast2)) & cast3
		cast = np.zeros_like(peaks)

		for k,turn_idx in enumerate(turn_idxs):
			# conditional to make sure indices dont go out
			turn0, turn1 = turn_idx[0], turn_idx[1]
			
			# Get difference of ingoing vs outgoing angle
			turn_angle = angles[turn1] - angles[turn0]

			# temp diff stuff
			X, Y = get_head_loc(pos[turn0:turn1,0], pos[turn0:turn1,1], angles[turn0:turn1], BL=0.3)
			temps = shallow_field(X)

			# curvyness stuff
			head_loc = np.vstack((X, Y)).T
			curve = curvyness(head_loc)

			if angle_threshold is not None:
				abs_turn_angle = np.abs(turn_angle)
				if abs_turn_angle < angle_threshold/(180/np.pi):
					continue
			
			all_turns.append(turn_angle)
			all_peak_times.append(peaks[k]/settings['fps'])
			angle1.append(reprocess_angles[turn0])
			angle2.append(reprocess_angles[turn1])
			turn_lengths.append((turn1-turn0)/settings['fps'])
			all_temp_diffs.append(temps[-1]-temps[0])
			all_curve.append(curve)
			all_casts.append(cast[k])
			all_fnames.append(settings['file'])
   
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

	# Store data for statistical testing
	if save_data:
		dataDir = f'data/{groupName}/'
		create_directory(dataDir)
		turns_df.to_csv(f"{dataDir}turns_{groupName}.csv", index=False)
	
	if ignore == True:
		return all_turns, in_box_angles

	return turns_df, in_box_angles, settings['fps']

def before_turns(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, casting=False, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	angle1s = []
	turn_diffs = []
	before_lengths = []
	before_times = []
	before_temp_changes = []

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angVels = vels[:, 2]

		# only keep turns not touching wall
		bl = bl_default	# border length
		stageW, stageH = settings['stageW'], settings['stageH']
		inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl))
		if speed_threshold is not None:
			inds = inds & (speed>speed_threshold) & (speed<5)
		actual_inds = np.where(inds)[0]

		reprocess_angles = (angles + np.pi)%(2*np.pi) - np.pi

		# Find turns using angVels
		peaks, turn_idxs = find_turn_indices(angVels, ht = ht)
		if len(peaks) == 0:
			continue

		if casting == True:
			# casting
			thresh = np.pi/6
			turn_angles = np.array([angles[t[1]] - angles[t[0]] for t in turn_idxs])
			cast1 = list(np.abs(turn_angles[:-1] + turn_angles[1:]) < thresh) + [False]
			cast2 = [False] + list(np.abs(turn_angles[:-1] + turn_angles[1:]) < thresh)
			cast3 = np.abs(turn_angles) < np.pi/3
			cast = (np.array(cast1) | np.array(cast2)) & cast3

			# Limit to peaks that are not too close to border
			turn_idxs = [t for k,t in enumerate(turn_idxs) if peaks[k] in actual_inds]
			cast = [c for k,c in enumerate(cast) if peaks[k] in actual_inds]
			peaks = [p for p in peaks if p in actual_inds]
			
			# Limit to cast == False
			turn_idxs = [t for k,t in enumerate(turn_idxs) if cast[k]==False]
			peaks = [p for k,p in enumerate(peaks) if cast[k]==False]

			# Get all "preturns"
			before_turn_idxs = [0] + [t[1] for t in turn_idxs[:-1]]

		else:
			# Get all "preturns"
			before_turn_idxs = [0] + [t[1] for t in turn_idxs[:-1]]

			# Limit to peaks that are not too close to border
			turn_idxs = [t for k,t in enumerate(turn_idxs) if peaks[k] in actual_inds]
			before_turn_idxs = [t for k,t in enumerate(before_turn_idxs) if peaks[k] in actual_inds]
			peaks = [p for p in peaks if p in actual_inds]

		for k,turn_idx in enumerate(turn_idxs):
			# conditional to make sure indices dont go out
			turn0, turn1 = turn_idx[0], turn_idx[1]
			before = before_turn_idxs[k]

			if turn0 - before < 15:
				# print('Close', before, turn0)
				continue

			# Get difference of ingoing vs outgoing angle
			turn_angle = angles[turn1] - angles[turn0]

			# temp diff stuff
			X, Y = get_head_loc(pos[before:turn0,0], pos[before:turn0,1], angles[before:turn0], BL=0.3)
			temps = shallow_field(X)
			temp_change = temps[-1] - temps[0]

			# time & distance 
			time_spent = (turn0 - before)/settings['fps']

			pos_segment = pos[before:turn0,:]
			dist_before = total_dist(pos_segment)

			angle1s.append(reprocess_angles[turn0])
			turn_diffs.append(turn_angle)
			before_lengths.append(dist_before)
			before_times.append(time_spent)
			before_temp_changes.append(temp_change)

	# restrict to angles where no temp gradient
	inds = ((np.pi/4) < np.abs(angle1s)) &  ((3*np.pi/4) > np.abs(angle1s))
	# apply to each variables
	angle1s = np.array(angle1s)[inds]
	turn_diffs = np.array(turn_diffs)[inds]
	before_lengths = np.array(before_lengths)[inds]
	before_times = np.array(before_times)[inds]
	before_temp_changes = np.array(before_temp_changes)[inds]
	# Indices which are the "correct" turn
	inds2 = (angle1s*turn_diffs) < 0 
	# Create dataframe to use for seaborn plotting
	df = pd.DataFrame({'Angle1': angle1s, 'Angle_diff': turn_diffs , 'Distance': before_lengths, 'TimeSpent': before_times, 'TempChange': before_temp_changes, 'Corrective':inds2})

	fig, ax = plt.subplots(3,1)	
	modes = ['TempChange', 'TimeSpent', 'Distance']
	for i in range(3):
		sns.histplot(df, x=modes[i], hue="Corrective", element="step",stat="density", common_norm=False, kde = True, ax = ax[i])

		# Calculate the median for each hue category
		meds = [np.median(df[df['Corrective'] == True][modes[i]]), np.median(df[df['Corrective'] == False][modes[i]])]
		colors = ['orange', 'blue']

		# Add median lines for each category
		for j in range(2):
			ax[i].axvline(meds[j], linestyle='--', color=colors[j])

	ax[0].set_xlim(-1,1)
	ax[1].set_xlim(0,20)
	ax[2].set_xlim(0,3)

	fig.suptitle(groupName)
	fig.tight_layout()

	fig.savefig(outputDir + 'before_turn_distribution_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'before_turn_distribution_'+groupName+'.png')
	fig.clf()
	plt.close(fig)
 
def turn_distribution(groupName, ht=np.pi/3, mode ='angle', side=False, speed_threshold = None, angle_threshold = None, plot_dir = 'explore', return_data = None):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold)
 
	all_turns = turn_df.all_turns
	angle1 = turn_df.angle1
	turn_lengths = turn_df.turn_lengths
	all_temp_diffs = turn_df.all_temp_diffs
	all_curve = turn_df.all_curve
 
	fig, ax = plt.subplots()

	if mode ==  'angle':
		q1, q2 = 25, 75
		# Conversion to deg
		all_turns = (180/np.pi)*np.array(all_turns)

		# Plot the distribution of "turns"
		if side == False:
			all_turns = np.abs(all_turns)

		# sns.histplot(x=all_turns, stat="density", kde=True, ax=ax, label='KDE Estimate')
		sns.histplot(x=all_turns, stat="density", ax=ax)


		fits_df = allfitdist(all_turns, common_cont_dist_names, sortby = 'BIC')
		x = np.linspace(0, max(all_turns), 1000) 

		for _, row in fits_df.head(3).iterrows():
			dist_name = row['Distribution']
			bic = row['BIC']
			# print(row['Params'])
			dist = getattr(stats, dist_name)
			fitted_dist = dist(*row['Params'])
			pdf_fitted = fitted_dist.pdf(x)
			ax.plot(x, pdf_fitted, '-', label=f'{dist_name} distribution fit, BIC = {bic:.1f}')
   
		# # Fit a lognormal distribution to the data
		# shape, loc, scale = stats.lognorm.fit(all_turns, floc=0)  # floc=0 fixes the location to 0
		#  
		# pdf_fitted = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
		# ax.plot(x, pdf_fitted, 'r-', label='Lognormal Fit')

		if return_data == True:
			kde_plot = ax.lines[0]  # Assuming the KDE line is the first line in the plot
			kde_x = kde_plot.get_xdata()
			kde_y = kde_plot.get_ydata()
			plt.close('all')
			return kde_x, kde_y

		ax.set_xlabel('Angle (deg)')
		xlim_max = 360
		if side == False:
			ax.set_xticks(30*(np.arange(13)))
			ax.set_xlim([0, xlim_max])
		else:
			ax.set_xticks(30*(np.arange(11)-5))
			ax.set_xlim([-xlim_max, xlim_max])

		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_turns, q1))}, {q2}%: {np.round(np.percentile(all_turns, q2))}'
	
	elif mode == 'time':
		q1, q2 = 25, 75
		sns.histplot(x=turn_lengths, stat="density", kde=True, ax=ax)
		ax.set_xlabel('Time spent in turn (seconds)')
		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(turn_lengths, q1),2)}, {q2}%: {np.round(np.percentile(turn_lengths, q2), 2)} seconds'

	elif mode == 'temp':
		q1, q2 = 10, 90

		sns.histplot(x=all_temp_diffs, stat="density", kde=True, ax=ax)
		ax.set_xlabel('Difference in temp at start vs end (C)')
		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_temp_diffs, q1),2)}, {q2}%: {np.round(np.percentile(all_temp_diffs, q2),2)} C'

	elif mode == 'temp_restrict':
		q1, q2 = 10, 90

		inds = ((np.pi/4) < np.abs(angle1)) &  ((3*np.pi/4) > np.abs(angle1))
		all_temp_diffs = np.array(all_temp_diffs)[inds]

		sns.histplot(x=all_temp_diffs, stat="density", kde=True, ax=ax)
		ax.set_xlabel('Difference in temp at start vs end (C)')
		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_temp_diffs, q1),2)}, {q2}%: {np.round(np.percentile(all_temp_diffs, q2),2)} C'

	elif mode == 'curve':
		q1, q2 = 10, 90

		inds = ((np.pi/4) < np.abs(angle1)) &  ((3*np.pi/4) > np.abs(angle1))
		all_curve = np.array(all_curve)[inds]
		angle1 = np.array(angle1)[inds]
		all_turns = np.array(all_turns)[inds]
		# Indices which are the "correct" turn
		inds2 = (angle1*all_turns) < 0 
		# Create dataframe to use for seaborn plotting
		df = pd.DataFrame({'Curve': all_curve, 'Angle1': angle1, 'Angle_diff': all_turns, 'Corrective':inds2})

		sns.histplot(df, x='Curve', hue="Corrective", element="step",stat="density", common_norm=False, kde = True, ax = ax)

		# Calculate the median for each hue category
		meds = [np.median(df[df['Corrective'] == True]['Curve']), np.median(df[df['Corrective'] == False]['Curve'])]
		colors = ['orange', 'blue']

		# Add median lines for each category
		for j in range(2):
			ax.axvline(meds[j], linestyle='--', color=colors[j])

		ax.set_xlim(1,5)

		# ax.set_xlabel('Difference in temp at start vs end (C)')
		# title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_temp_diffs, q1),2)}, {q2}%: {np.round(np.percentile(all_temp_diffs, q2),2)} C'
		title = 'Curve'
	
	elif mode == 'joint':
		all_turns = (180/np.pi)*np.array(all_turns)
		all_turns = np.abs(all_turns)

		sns.regplot(x=all_turns, y=turn_lengths, ax=ax)
		ax.set_xlabel('Turn angle (deg)')
		ax.set_ylabel('Time in turn (seconds)')

		title = 'Time in turn vs turn angle joint regplot'

	ax.set_title(title)
	ax.legend()


	fig.suptitle(groupName)
	fig.tight_layout()

	if speed_threshold is not None:
		fig.savefig(outputDir + 'turn_distribution_'+mode +'_speed_thresh_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + 'turn_distribution_'+mode +'_speed_thresh_'+groupName+'.png')
	else:
		fig.savefig(outputDir + 'turn_distribution_'+mode+'_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + 'turn_distribution_'+mode+'_'+groupName+'.png')
	fig.clf()
	plt.close(fig)

def polar_turns(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, mode = None, casting = False, return_data = None, plot_dir = 'explore', nbins = 4):
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold,save_data=True)

	angle1 = np.array(turn_df.angle1)

	min_alpha = 0.2

	if nbins == 4:
		bins = np.linspace(-np.pi/4, 2*np.pi-np.pi/4, nbins+1)
		angle1 = (angle1+ np.pi/4)%(2*np.pi) - np.pi/4
		alphas = [min_alpha, 1, min_alpha, 1]
	elif nbins == 6:
		bins = np.linspace(0, 2*np.pi, nbins+1)
		angle1 = (angle1)%(2*np.pi)
		alphas = [min_alpha, 1, min_alpha,min_alpha, 1,min_alpha]
 
	mid_angles = (bins[1:] + bins[:-1])/2
	inds = np.digitize(angle1, bins)
 
	df = turn_df.rename(columns={
    'angle1': 'Angle1',
    'all_turns': 'Angle_diff',
    'all_peak_times': 'peaks'
    })
	df['Angle_diff'] = (180/np.pi)*df.Angle_diff
	df['inds_Angle1'] = inds
	
	# Old
 
	# angle_diff_reprocess = (180/np.pi)*np.array(all_turns)
	# df = pd.DataFrame({'Angle1': angle1, 'Angle_diff': angle_diff_reprocess , 'inds_Angle1': inds, 'peaks': all_peak_times, 'Casts': all_casts})

	# RECOMMENT
 
	# pos = np.array(df[df['Angle_diff'] > 0]['Angle1'])
	# neg = np.array(df[df['Angle_diff'] < 0]['Angle1'])

	# pos = stats.vonmises(loc= 0.5 * np.pi, kappa=0.5).rvs(100)
	# neg = stats.vonmises(loc= 1.5 * np.pi, kappa=0.5).rvs(100)%(2*np.pi) - np.pi

	# print(stats.ks_2samp(pos, neg))

	# sorted_data = np.sort(pos)
	# cdf = np.linspace(0, 1, len(sorted_data))
	# plt.plot(sorted_data, cdf)
	# sorted_data = np.sort(neg)
	# cdf = np.linspace(0, 1, len(sorted_data))
	# plt.plot(sorted_data, cdf)
	# plt.hist(pos, bins=30, density=True, alpha=0.7)
	# plt.hist(neg, bins=30, density=True, alpha=0.7)
	# plt.show()
	# plt.close()

	# if casting == True:
	# 	df = df[df.Casts==False]

	if mode is None:
		dfs = [df]
		fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
		ax = [ax]

	elif mode  == 'stratify':
		bdry = 30
		dfs = [df[df.peaks < bdry], df[df.peaks >= bdry]]
		fig, ax = plt.subplots(1,2,subplot_kw={'projection': 'polar'}, figsize=(12,6))
		ax[0].set_title(f'First {bdry} seconds')
		ax[1].set_title(f'After {bdry} seconds')

	for i,df in enumerate(dfs):
		# print(len(df))
		pvals = []
		Ns = []
		lCountNorm, rCountNorm = np.zeros(nbins), np.zeros(nbins)
		for k in range(nbins):
			df_subset = df[df.inds_Angle1 == (k+1)]
			pos = df_subset[df_subset['Angle_diff'] > 0].shape[0]
			neg = df_subset[df_subset['Angle_diff'] < 0].shape[0]
			Ns.append([neg,pos+neg])
			stat, pval = proportions_ztest(pos, pos+neg, 0.5)
			pvals.append(pval)
			lCountNorm[k] = pos/(pos+neg)
			rCountNorm[k] = neg/(pos+neg)

		if return_data == True:
			plt.close(fig)
			return Ns

		p1 = ax[i].bar(mid_angles,rCountNorm,width = (2 * np.pi / nbins), color=list(zip(['purple']*nbins, alphas)),edgecolor=list(zip(['black']*nbins, alphas)),linewidth=1.5)
		p3 = ax[i].bar(mid_angles,lCountNorm,width = (2 * np.pi / nbins), color=list(zip(['green']*nbins, alphas)), bottom =rCountNorm, edgecolor=list(zip(['black']*nbins, alphas)),linewidth=1.5)

		for bar, height, N in zip(mid_angles, pvals, Ns):
			# print(height)
			ax[i].text(bar + 2*np.pi/nbins/4, 1.3, f'N={N[0]}/{N[1]}, p={height:.4f}', ha='center', va='bottom')

		ax[i].legend((p1[0],p3[0]),('Right','Left'),loc='upper right')
		ax[i].set_xlabel('Incoming angle')
		ax[i].set_rorigin(-1.0)
		ax[i].set_ylim([0,1])
		ax[i].xaxis.grid(False)
		ax[i].set_aspect('equal')

	fig.suptitle(groupName)
	fig.tight_layout()

	default = 'polar_turns_'
	if mode is not None:
		default = default + mode + '_'

	if speed_threshold is not None:
		fig.savefig(outputDir + default + 'speed_thresh_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png +  default + 'speed_thresh_'+groupName+'.png')
	else:
		fig.savefig(outputDir + default +groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + default +groupName+'.png')

def polar_ash(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, mode = None, casting = False, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold)
 
	df = turn_df.rename(columns={
    'angle1': 'Angle1',
    'all_turns': 'Angle_diff',
    })
	df['Angle_diff'] = (180/np.pi)*df['Angle_diff']
  

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))

	pos = np.array(df[df['Angle_diff'] > 0]['Angle1'])
	neg = np.array(df[df['Angle_diff'] < 0]['Angle1'])

	n_bins = 8
	m = 8
	pos = circular_ash(pos, n_bins=n_bins, m=m)
	neg = circular_ash(neg, n_bins=n_bins, m=m)
	rCountNorm = neg/(pos+neg)
	# print(neg+pos)
	rCountNorm = extend_for_circ(rCountNorm)

	bins = np.linspace(-np.pi, np.pi, n_bins*m+1)
	mid_angles = (bins[1:] + bins[:-1])/2
	mid_angles = extend_for_circ(mid_angles)
	test = (neg+pos)/np.max(neg+pos)
	test = extend_for_circ(test)

	# Plot the data
	ax.plot(mid_angles, rCountNorm, color='black',linewidth=1.5)
	ax.plot(mid_angles, test, color='red',linewidth=1.5)
	# Fill the area between 
	p1 = ax.fill_between(mid_angles, rCountNorm, np.zeros_like(mid_angles), color='purple')
	p2 = ax.fill_between(mid_angles, rCountNorm, np.ones_like(mid_angles), color='green')

	ax.legend((p1,p2),('Right','Left'),loc='upper right')
	ax.set_xlabel('Incoming angle')
	ax.set_rorigin(-1.0)
	ax.set_ylim([0,1])
	ax.set_yticks(np.arange(5)/4)
	ax.xaxis.grid(False)
	ax.set_aspect('equal')

	fig.suptitle(groupName)
	fig.tight_layout()

	default = 'polar_ash_'

	if speed_threshold is not None:
		fig.savefig(outputDir + default + 'speed_thresh_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png +  default + 'speed_thresh_'+groupName+'.png')
	else:
		fig.savefig(outputDir + default +groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + default +groupName+'.png')

def num_turns(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, mode = None, return_data = None, plot_dir = 'explore'):
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold)

	angle1 = turn_df.angle1
 
	# Plot number of turns
	in_box_angles = np.concatenate(in_box_angles)


	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	if mode == 'ash':
		n_bins = 8
		m = 16
		angle1 = circular_ash(angle1, n_bins=n_bins, m=m)
		in_box_angles = circular_ash(in_box_angles, n_bins=n_bins, m=m)/fps
		ratio = angle1/in_box_angles
		ratio = extend_for_circ(ratio)

		bins = np.linspace(-np.pi, np.pi, n_bins*m+1)
		mid_angles = (bins[1:] + bins[:-1])/2
		mid_angles = extend_for_circ(mid_angles)

		if return_data == True:
			plt.close(fig)
			return mid_angles, ratio

		ax.plot(mid_angles, 60*ratio, color='black',linewidth=1.5)

		ax.set_ylim([0,50])
		ax.set_title(f'{groupName}\n Number of turns per minute')

	elif mode == None:
		num_bins = 6
		colors = ['green','grey','brown','brown','grey','green']
		angle1 = angle1%(2*np.pi)
		in_box_angles = in_box_angles%(2*np.pi)

		hist, bins = np.histogram(angle1, bins=num_bins, range=(0, 2*np.pi))
		inds = np.digitize(angle1, bins)
		inds2 = np.digitize(in_box_angles, bins)
		
		mag = np.array([np.sum(angle1[inds==(i+1)])/np.sum(in_box_angles[inds2==(i+1)]/fps) for i in range(num_bins)])

		if return_data == True:
			plt.close(fig)
			return mag

		barbins = bins[:-1] + np.pi / num_bins
		ax.bar(barbins, 60*mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')

		ax.set_ylim([0,50])
		# ax.set_yticks(np.linspace(0,speed_threshold[1]/2,6))
		ax.set_title(f'{groupName}\n Number of turns per minute spent moving in each bin')

	message = 'num_turns_'
	if speed_threshold is not None:
		message += 'speed_thresh_'
	if mode is not None:
		message += 'ash_'
	
	# Save plot
	fig.tight_layout()
	fig.savefig(outputDir + message + groupName + '.pdf', transparent=True)
	fig.savefig(outputDir_png + message + groupName + '.png')

	fig.clf()
	plt.close(fig)





### DEPRECATED

# def distribution_in_out(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, plot_dir = 'explore'):

# 	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir_png)

# 	all_turns, all_peak_times, angle1, angle2, turn_lengths, all_temp_diffs, all_curve, all_casts, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold)
			
# 	# Plot to see conditional distribution
# 	nbins = 6
# 	bins = np.linspace(-np.pi, np.pi, nbins+1)

# 	hist, _ = np.histogram(angle1, bins=bins)
# 	p1 = hist/len(angle1)
# 	hist, _ = np.histogram(angle2, bins=bins)
# 	p2 = hist/len(angle2)

# 	hist, _, _ = np.histogram2d(angle1, angle2, bins=bins)
# 	p12 = hist/len(angle1)
	
# 	# Calculate conditional distribution P(angle2 | angle1)
# 	p_angle2_given_angle1 = p12 / p1[:, None]

# 	fig, ax = plt.subplots()
# 	imshow_plot = ax.imshow(p_angle2_given_angle1)

# 	# Add numbers on the imshow plot
# 	for i in range(nbins):
# 		for j in range(nbins):
# 			ax.text(j, i, f'{p_angle2_given_angle1[i, j]:.2f}', ha='center', va='center', color='white')

# 	# Set xticks and labels
# 	ticks = np.linspace(-0.5, nbins-1+0.5, nbins+1)
# 	ticklabels = np.round(np.linspace(-180, 180, nbins+1))	# cast angles to deg
# 	# ticklabels = np.round(np.linspace(-np.pi, np.pi, nbins+1), 2)

# 	# Add lines to middle
# 	for y in ticks:
# 		ax.axhline(y, color='red', linestyle='--')

# 	fig.gca().invert_yaxis()
# 	ax.set(xticks=ticks, xticklabels=ticklabels)
# 	ax.set(yticks=ticks, yticklabels=ticklabels)
# 	ax.set(xlabel='angle_next', ylabel='angle_prev')
# 	ax.set(title = 'P(angle_next | angle_prev)')

# 	# Add a colorbar to the plot
# 	colorbar = plt.colorbar(imshow_plot)
# 	imshow_plot.set_clim(vmin=0, vmax=0.5)

# 	fig.suptitle(groupName)
# 	fig.tight_layout()

# 	# Save plot
# 	if speed_threshold is not None:
# 		fig.savefig(outputDir + 'turn_conditionals_speed_thresh_'+groupName+'.pdf', transparent=True)
# 		fig.savefig(outputDir_png + 'turn_conditionals_speed_thresh_'+groupName+'.png')
# 	else:
# 		fig.savefig(outputDir + 'turn_conditionals_'+groupName+'.pdf', transparent=True)
# 		fig.savefig(outputDir_png + 'turn_conditionals_'+groupName+'.png')
	
# 	fig.clf()
# 	plt.close(fig)

# 	# Plot to see joint distribution of in and out angle
# 	# g = sns.jointplot(x=angle1, y=angle2, kind="hist", bins=8)
# 	# g.set_axis_labels(xlabel='angle in', ylabel='angle out')
# 	# plt.tight_layout()
# 	# plt.show()
# 	# plt.close()

# def joyplot_polar(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, plot_dir = 'explore'):
	
# 	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir_png)

# 	all_turns, all_peak_times, angle1, angle2, turn_lengths, all_temp_diffs, all_curve, all_casts, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold)

# 	angle1 = np.array(angle1)
# 	# angle2_reprocess = (angle2 - mid_angles[inds-1] + np.pi)%(2*np.pi) - np.pi
# 	# angle_diff = (angle2 - angle1 + np.pi)%(2*np.pi) - np.pi
# 	# angle_diff_reprocess = (180/np.pi)*angle_diff # convert to deg
# 	angle_diff_reprocess = (180/np.pi)*np.array(all_turns)

			
# 	# Plot to see conditional distribution
# 	# nbins = 6
# 	# bins = np.linspace(-np.pi, np.pi, nbins+1)
# 	# mid_angles = (bins[1:] + bins[:-1])/2
# 	# inds = np.digitize(angle1, bins)

# 	nbins = 4
# 	bins = np.linspace(-np.pi/4, 2*np.pi-np.pi/4, nbins+1)
# 	mid_angles = (bins[1:] + bins[:-1])/2
# 	angle1 = (angle1+ np.pi/4)%(2*np.pi) - np.pi/4
# 	inds = np.digitize(angle1, bins)
# 	# alphas = [0.2, 1, 0.2, 1]

# 	df = pd.DataFrame({'Angle1': angle1, 'Angle_diff': angle_diff_reprocess , 'inds_Angle1': inds})

# 	# fig, axs = joypy.joyplot(df, by='inds_Angle1', column='Angle2', overlap=0, figsize=(10, 6), hist=True, density=True, bins=np.linspace(-180, 180, nbins*10+1))
# 	# fig, axs = joypy.joyplot(df, by='inds_Angle1', column='Angle2', overlap=0, figsize=(10, 6))

# 	fig, axs = plt.subplots(nbins, 2, figsize=(10,6))

# 	for k in range(len(axs)):
# 		df_subset = df[df.inds_Angle1 == (nbins-k)]
# 		sns.histplot(data=df_subset, x="Angle_diff", element='step', stat='density', bins=np.linspace(-180, 180, nbins*4+1), ax = axs[k][1])
# 		sns.kdeplot(data=df_subset, x="Angle_diff", fill=True, ax = axs[k][0])
# 		ang2 = int(np.round((180/np.pi)*(bins[nbins-k])))
# 		ang1 = int(np.round((180/np.pi)*(bins[nbins-k-1])))

# 		axs[k][0].set_ylabel(f'[{ang1},{ang2}]')
# 		axs[k][1].set_ylabel('')
		
# 		for i in range(2):
# 			axs[k][i].set_xlabel('')
# 			axs[k][i].set_xlim([-180, 180])

# 			axs[k][i].spines['top'].set_visible(False)
# 			axs[k][i].spines['right'].set_visible(False)
# 			axs[k][i].spines['left'].set_visible(False)

# 	max_value = max(ax.get_ylim()[1] for ax in [axs[k][0] for k in range(nbins)])
# 	max_value2 = max(ax.get_ylim()[1] for ax in [axs[k][1] for k in range(nbins)])
# 	for k in range(len(axs)):
# 		axs[k][0].set_ylim([0, max_value])
# 		axs[k][1].set_ylim([0, max_value2])

# 	fig.suptitle(groupName)
# 	fig.tight_layout()

# 	if speed_threshold is not None:
# 		fig.savefig(outputDir + 'turn_joyplot_speed_thresh_'+groupName+'.pdf', transparent=True)
# 		fig.savefig(outputDir_png + 'turn_joyplot_speed_thresh_'+groupName+'.png')
# 	else:
# 		fig.savefig(outputDir + 'turn_joyplot_'+groupName+'.pdf', transparent=True)
# 		fig.savefig(outputDir_png + 'turn_joyplot_'+groupName+'.png')
		
# 	fig.clf()
# 	plt.close('all')