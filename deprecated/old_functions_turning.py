### DEPRECATED

# def peaks_verification(groupName, speed_threshold = None, plot_dir = 'explore'):
# 	inputDir = 'outputs/outputs_' + groupName + '/'
# 	dirs = os.listdir(inputDir)

# 	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir_png)

# 	hts = (np.pi/3)*(np.arange(6)+1)
# 	timewindows = np.arange(7) + 1
# 	cmap = get_cmap('viridis')

# 	fig, ax = plt.subplots(1,len(hts), figsize = (len(hts)*5,4))
	
# 	for i in range(len(hts)):
# 		data = []
# 		for file in dirs:
# 			if 'output' not in file.split('/')[-1]:
# 				continue

# 			f1 = open(inputDir + file, 'rb')
# 			pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
# 			f1.close()

# 			speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
# 			angVels = vels[:, 2]

# 			# only keep turns not touching wall
# 			bl = bl_default	# border length
# 			stageW, stageH = settings['stageW'], settings['stageH']
# 			inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl))
# 			if speed_threshold is not None:
# 				inds = inds & (speed>speed_threshold) & (speed<5)
# 			actual_inds = np.where(inds)[0]

# 			peaks, _ = find_peaks(np.abs(angVels), height=hts[i])
# 			peaks =[p for p in peaks if p in actual_inds]
# 			for j in range(len(timewindows)):
# 				tm = timewindows[j]
# 				for p in peaks:
# 					if (p-tm < 0) or p+tm >= len(angles) :
# 						continue
# 					turn = angles[p+tm] - angles[p-tm]
# 					data.append([(180/np.pi)*turn, tm])

# 		df = pd.DataFrame(data, columns=['Turn', 'TimeWindow'])
# 		ax[i].set_title(f'>{np.round((180/np.pi)*hts[i])} deg/s')
# 		for k in timewindows:
# 			sns.kdeplot(data=df[df['TimeWindow']==k], x="Turn", ax = ax[i], color=cmap(k/ len(timewindows)) )

# 	fig.suptitle(groupName)
# 	fig.tight_layout()
# 	fig.savefig(outputDir + 'peaks_verification_'+groupName+'.pdf', transparent=True)
# 	fig.savefig(outputDir_png + 'peaks_verification_'+groupName+'.png')
# 	fig.clf()
# 	plt.close(fig)

# def before_turns(groupName, ht=np.pi/3, speed_threshold = None, angle_threshold = None, casting=False, plot_dir = 'explore'):

# 	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir_png)

# 	inputDir = 'outputs/outputs_' + groupName + '/'
# 	dirs = os.listdir(inputDir)

# 	angle1s = []
# 	turn_diffs = []
# 	before_lengths = []
# 	before_times = []
# 	before_temp_changes = []

# 	for file in dirs:
# 		if 'output' not in file.split('/')[-1]:
# 			continue

# 		f1 = open(inputDir + file, 'rb')
# 		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
# 		f1.close()

# 		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
# 		angVels = vels[:, 2]

# 		# only keep turns not touching wall
# 		bl = bl_default	# border length
# 		stageW, stageH = settings['stageW'], settings['stageH']
# 		inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl))
# 		if speed_threshold is not None:
# 			inds = inds & (speed>speed_threshold) & (speed<5)
# 		actual_inds = np.where(inds)[0]

# 		reprocess_angles = (angles + np.pi)%(2*np.pi) - np.pi

# 		# Find turns using angVels
# 		peaks, turn_idxs = find_turn_indices(angVels, ht = ht)
# 		if len(peaks) == 0:
# 			continue

# 		if casting == True:
# 			# casting
# 			thresh = np.pi/6
# 			turn_angles = np.array([angles[t[1]] - angles[t[0]] for t in turn_idxs])
# 			cast1 = list(np.abs(turn_angles[:-1] + turn_angles[1:]) < thresh) + [False]
# 			cast2 = [False] + list(np.abs(turn_angles[:-1] + turn_angles[1:]) < thresh)
# 			cast3 = np.abs(turn_angles) < np.pi/3
# 			cast = (np.array(cast1) | np.array(cast2)) & cast3

# 			# Limit to peaks that are not too close to border
# 			turn_idxs = [t for k,t in enumerate(turn_idxs) if peaks[k] in actual_inds]
# 			cast = [c for k,c in enumerate(cast) if peaks[k] in actual_inds]
# 			peaks = [p for p in peaks if p in actual_inds]
			
# 			# Limit to cast == False
# 			turn_idxs = [t for k,t in enumerate(turn_idxs) if cast[k]==False]
# 			peaks = [p for k,p in enumerate(peaks) if cast[k]==False]

# 			# Get all "preturns"
# 			before_turn_idxs = [0] + [t[1] for t in turn_idxs[:-1]]

# 		else:
# 			# Get all "preturns"
# 			before_turn_idxs = [0] + [t[1] for t in turn_idxs[:-1]]

# 			# Limit to peaks that are not too close to border
# 			turn_idxs = [t for k,t in enumerate(turn_idxs) if peaks[k] in actual_inds]
# 			before_turn_idxs = [t for k,t in enumerate(before_turn_idxs) if peaks[k] in actual_inds]
# 			peaks = [p for p in peaks if p in actual_inds]

# 		for k,turn_idx in enumerate(turn_idxs):
# 			# conditional to make sure indices dont go out
# 			turn0, turn1 = turn_idx[0], turn_idx[1]
# 			before = before_turn_idxs[k]

# 			if turn0 - before < 15:
# 				# print('Close', before, turn0)
# 				continue

# 			# Get difference of ingoing vs outgoing angle
# 			turn_angle = angles[turn1] - angles[turn0]

# 			# temp diff stuff
# 			X, Y = get_head_loc(pos[before:turn0,0], pos[before:turn0,1], angles[before:turn0], BL=0.3)
# 			temps = shallow_field(X)
# 			temp_change = temps[-1] - temps[0]

# 			# time & distance 
# 			time_spent = (turn0 - before)/settings['fps']

# 			pos_segment = pos[before:turn0,:]
# 			dist_before = total_dist(pos_segment)

# 			angle1s.append(reprocess_angles[turn0])
# 			turn_diffs.append(turn_angle)
# 			before_lengths.append(dist_before)
# 			before_times.append(time_spent)
# 			before_temp_changes.append(temp_change)

# 	# restrict to angles where no temp gradient
# 	inds = ((np.pi/4) < np.abs(angle1s)) &  ((3*np.pi/4) > np.abs(angle1s))
# 	# apply to each variables
# 	angle1s = np.array(angle1s)[inds]
# 	turn_diffs = np.array(turn_diffs)[inds]
# 	before_lengths = np.array(before_lengths)[inds]
# 	before_times = np.array(before_times)[inds]
# 	before_temp_changes = np.array(before_temp_changes)[inds]
# 	# Indices which are the "correct" turn
# 	inds2 = (angle1s*turn_diffs) < 0 
# 	# Create dataframe to use for seaborn plotting
# 	df = pd.DataFrame({'Angle1': angle1s, 'Angle_diff': turn_diffs , 'Distance': before_lengths, 'TimeSpent': before_times, 'TempChange': before_temp_changes, 'Corrective':inds2})

# 	fig, ax = plt.subplots(3,1)	
# 	modes = ['TempChange', 'TimeSpent', 'Distance']
# 	for i in range(3):
# 		sns.histplot(df, x=modes[i], hue="Corrective", element="step",stat="density", common_norm=False, kde = True, ax = ax[i])

# 		# Calculate the median for each hue category
# 		meds = [np.median(df[df['Corrective'] == True][modes[i]]), np.median(df[df['Corrective'] == False][modes[i]])]
# 		colors = ['orange', 'blue']

# 		# Add median lines for each category
# 		for j in range(2):
# 			ax[i].axvline(meds[j], linestyle='--', color=colors[j])

# 	ax[0].set_xlim(-1,1)
# 	ax[1].set_xlim(0,20)
# 	ax[2].set_xlim(0,3)

# 	fig.suptitle(groupName)
# 	fig.tight_layout()

# 	fig.savefig(outputDir + 'before_turn_distribution_'+groupName+'.pdf', transparent=True)
# 	fig.savefig(outputDir_png + 'before_turn_distribution_'+groupName+'.png')
# 	fig.clf()
# 	plt.close(fig)

# def turn_distribution(groupName, ht=np.pi/3, mode ='angle', side=False, speed_threshold = None, angle_threshold = None, plot_dir = 'explore', return_data = None):

# 	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
# 	create_directory(outputDir_png)

# 	turn_df, in_box_angles, fps = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold)
 
# 	all_turns = turn_df.all_turns
# 	angle1 = turn_df.angle1
# 	turn_lengths = turn_df.turn_lengths
# 	all_temp_diffs = turn_df.all_temp_diffs
# 	all_curve = turn_df.all_curve
 
# 	fig, ax = plt.subplots()

# 	if mode ==  'angle':
# 		q1, q2 = 25, 75
# 		# Conversion to deg
# 		all_turns = (180/np.pi)*np.array(all_turns)

# 		# Plot the distribution of "turns"
# 		if side == False:
# 			all_turns = np.abs(all_turns)

# 		# sns.histplot(x=all_turns, stat="density", kde=True, ax=ax, label='KDE Estimate')
# 		sns.histplot(x=all_turns, stat="density", ax=ax)


# 		fits_df = allfitdist(all_turns, common_cont_dist_names, sortby = 'BIC')
# 		x = np.linspace(0, max(all_turns), 1000) 

# 		for _, row in fits_df.head(3).iterrows():
# 			dist_name = row['Distribution']
# 			bic = row['BIC']
# 			# print(row['Params'])
# 			dist = getattr(stats, dist_name)
# 			fitted_dist = dist(*row['Params'])
# 			pdf_fitted = fitted_dist.pdf(x)

# 			# if dist_name == 'lognormal':
# 			ax.plot(x, pdf_fitted, '-', label=f'{dist_name} distribution fit, BIC = {bic:.1f}')

   
# 		# # Fit a lognormal distribution to the data
# 		# shape, loc, scale = stats.lognorm.fit(all_turns, floc=0)  # floc=0 fixes the location to 0
# 		#  
# 		# pdf_fitted = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
# 		# ax.plot(x, pdf_fitted, 'r-', label='Lognormal Fit')

# 		if return_data == True:
# 			kde_plot = ax.lines[0]  # Assuming the KDE line is the first line in the plot
# 			kde_x = kde_plot.get_xdata()
# 			kde_y = kde_plot.get_ydata()
# 			plt.close('all')
# 			return kde_x, kde_y

# 		ax.set_xlabel('Angle (deg)')
# 		xlim_max = 360
# 		if side == False:
# 			ax.set_xticks(30*(np.arange(13)))
# 			ax.set_xlim([0, xlim_max])
# 		else:
# 			ax.set_xticks(30*(np.arange(11)-5))
# 			ax.set_xlim([-xlim_max, xlim_max])

# 		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_turns, q1))}, {q2}%: {np.round(np.percentile(all_turns, q2))}'
	
# 	elif mode == 'time':
# 		q1, q2 = 25, 75
# 		sns.histplot(x=turn_lengths, stat="density", kde=True, ax=ax)
# 		ax.set_xlabel('Time spent in turn (seconds)')
# 		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(turn_lengths, q1),2)}, {q2}%: {np.round(np.percentile(turn_lengths, q2), 2)} seconds'

# 	elif mode == 'temp':
# 		q1, q2 = 10, 90

# 		sns.histplot(x=all_temp_diffs, stat="density", kde=True, ax=ax)
# 		ax.set_xlabel('Difference in temp at start vs end (C)')
# 		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_temp_diffs, q1),2)}, {q2}%: {np.round(np.percentile(all_temp_diffs, q2),2)} C'

# 	elif mode == 'temp_restrict':
# 		q1, q2 = 10, 90

# 		inds = ((np.pi/4) < np.abs(angle1)) &  ((3*np.pi/4) > np.abs(angle1))
# 		all_temp_diffs = np.array(all_temp_diffs)[inds]

# 		sns.histplot(x=all_temp_diffs, stat="density", kde=True, ax=ax)
# 		ax.set_xlabel('Difference in temp at start vs end (C)')
# 		title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_temp_diffs, q1),2)}, {q2}%: {np.round(np.percentile(all_temp_diffs, q2),2)} C'

# 	elif mode == 'curve':
# 		q1, q2 = 10, 90

# 		inds = ((np.pi/4) < np.abs(angle1)) &  ((3*np.pi/4) > np.abs(angle1))
# 		all_curve = np.array(all_curve)[inds]
# 		angle1 = np.array(angle1)[inds]
# 		all_turns = np.array(all_turns)[inds]
# 		# Indices which are the "correct" turn
# 		inds2 = (angle1*all_turns) < 0 
# 		# Create dataframe to use for seaborn plotting
# 		df = pd.DataFrame({'Curve': all_curve, 'Angle1': angle1, 'Angle_diff': all_turns, 'Corrective':inds2})s

# 		sns.histplot(df, x='Curve', hue="Corrective", element="step",stat="density", common_norm=False, kde = True, ax = ax)

# 		# Calculate the median for each hue category
# 		meds = [np.median(df[df['Corrective'] == True]['Curve']), np.median(df[df['Corrective'] == False]['Curve'])]
# 		colors = ['orange', 'blue']

# 		# Add median lines for each category
# 		for j in range(2):
# 			ax.axvline(meds[j], linestyle='--', color=colors[j])

# 		ax.set_xlim(1,5)

# 		# ax.set_xlabel('Difference in temp at start vs end (C)')
# 		# title = f'AngVel threshold: {np.round((180/np.pi)*ht)} deg/s, Percentiles are {q1}%: {np.round(np.percentile(all_temp_diffs, q1),2)}, {q2}%: {np.round(np.percentile(all_temp_diffs, q2),2)} C'
# 		title = 'Curve'
	
# 	elif mode == 'joint':
# 		all_turns = (180/np.pi)*np.array(all_turns)
# 		all_turns = np.abs(all_turns)

# 		sns.regplot(x=all_turns, y=turn_lengths, ax=ax)
# 		ax.set_xlabel('Turn angle (deg)')
# 		ax.set_ylabel('Time in turn (seconds)')

# 		title = 'Time in turn vs turn angle joint regplot'

# 	ax.set_title(title)
# 	ax.legend()

# 	fig.suptitle(groupName)
# 	fig.tight_layout()

# 	if speed_threshold is not None:

# 		fig.savefig(outputDir + 'turn_distribution_'+mode +'_speed_thresh_'+groupName+'.pdf', transparent=True)
# 		fig.savefig(outputDir_png + 'turn_distribution_'+mode +'_speed_thresh_'+groupName+'.png')

# 	else:

# 		fig.savefig(outputDir + 'turn_distribution_'+mode+'_'+groupName+'.pdf', transparent=True)
# 		fig.savefig(outputDir_png + 'turn_distribution_'+mode+'_'+groupName+'.png')
  
# 	fig.clf()
# 	plt.close(fig)


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