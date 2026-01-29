


# Deprecated

def get_angvelpeak(fin, ht=np.pi/3):
	
	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _  = pickle.load(f1)
	f1.close()

	stageH, stageW = settings['stageH'], settings['stageW']

	speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
	angVels = vels[:,2]

	peaks, turn_idxs = find_turn_indices(angVels, ht = ht)

	return peaks 


def border_touch(groupName, plot_dir = 'explore'):
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	all_last_points = []
	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		stageW, stageH = settings['stageW'], settings['stageH']

		# check only middle ones
		bl = bl_default	# border length
		inds = ~((pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) 
			& (pos[:,0]>bl) & (pos[:,0]<(stageW-bl)))
		
		# adjust so that it accounts for end
		first_touch_ind = np.argmax(inds)
		if np.max(inds): # if nothing outside boundary, use the last point
			lp = pos[first_touch_ind,0:2]
		else:
			lp = pos[-1,0:2]

		all_last_points.append(lp)
	all_last_points = np.array(all_last_points)
	
	fig, ax = plt.subplots()
	ax.scatter(all_last_points[:,0], all_last_points[:,1], color='red')
	ax.set_xlim([0, stageW])
	ax.set_ylim([0, stageH])

	ax.set_xlabel('boundary first touch points')

	fig.tight_layout()
	fig.savefig(outputDir + 'boundary_touch_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'boundary_touch_'+groupName+'.png', transparent=True)
	fig.clf()
	plt.close(fig)

def efficiency_plot(groupName, plot_dir = 'explore', return_data = None):
    
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)
 
	effx_list = []
 
	# Store data for statistical testing
	dataDir = f'data/{groupName}/'
	create_directory(dataDir)
	reachOrNot_df = []
	effx_df = []
	
	count_reached = 0
	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _  = pickle.load(f1)
		f1.close()

		normalized_x = pos[:,0]/settings['stageW']
  
		distances = np.linalg.norm(pos[1:,] - pos[:-1,], axis=1)
		cumulative_distances = np.cumsum(distances)

		goal = 0.8

		# ind = next((i for i in range(len(pos)) if normalized_x[i] > limiter), np.argmax(normalized_x))
  
		ind = next((i for i in range(len(pos)) if normalized_x[i] > goal), len(pos)-1)

		travel_to_max = cumulative_distances[ind-1]/settings['stageW']
		invade_dist = min(np.max(normalized_x), goal) - np.min(normalized_x)
  
		effx = invade_dist/travel_to_max
		effx_list.append(effx)
		effx_df.append({"fname": file, "effx": effx})

		if np.max(normalized_x) > goal:
			count_reached += 1
			reachOrNot_df.append({"fname": file, "reachOrNot": 1})
		else:
			reachOrNot_df.append({"fname": file, "reachOrNot": 0})

	# Convert list of dicts into df and store into data storage
	reachOrNot_df = pd.DataFrame(reachOrNot_df)
	effx_df = pd.DataFrame(effx_df)
	reachOrNot_df.to_csv(f"{dataDir}reachOrNot_{groupName}.csv", index=False)
	effx_df.to_csv(f"{dataDir}effx_{groupName}.csv", index=False)

	if return_data is not None:
		return effx_list, count_reached
	
	fig, ax = plt.subplots()
 
	ax.boxplot(effx_list)
	plt.scatter([1]*len(effx_list), effx_list)
 
	# ax.set_xticklabels([round(p, 2) for p in percent_reached])
	ax.set_ylabel('Efficiency')
	ax.set_ylim([0, 1])

	fig.suptitle(groupName)

	fig.tight_layout()
	fig.savefig(outputDir + 'efficiency_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'efficiency_'+groupName+'.png')

	fig.clf()
	plt.close(fig)
 

def heading_index(groupName, speed_threshold = 0.25, mode = 'orientation', plot_dir = 'explore', return_data = None):
    
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)
 
	hi_list = []
	hi_df = []
 
	# Store data for statistical testing
	dataDir = f'data/{groupName}/'
	create_directory(dataDir)

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _  = pickle.load(f1)
		f1.close()

		# Find all indices where speed greater than threshold
		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		idx = speed > speed_threshold
		# Limit threshold by 5? maybe set inframe?
  
		if mode == 'orientation':
			filtered_angles = angles[idx]
			hi = np.average(np.cos(filtered_angles))
			# print(len(idx), np.isnan(hi), file)
		elif mode == 'angvel':
			# Takes the normalized X coordinate 
			# corresponding to cosine
			velsX_filtered = vels[:,0][idx]
			speed_filtered = speed[idx]
			hi = np.average(velsX_filtered/speed_filtered)

		elif mode == 'new':
			hi = np.sum(vels[:,0])/np.sum(speed)
		
		if np.isnan(hi):
			print('One of the heading indices is nan: ', file)
			continue
  
		hi_list.append(hi)
		hi_df.append({"fname": file, "hi": hi})

	# Convert list of dicts into df and store into data storage
	hi_df = pd.DataFrame(hi_df)

	hi_df.to_csv(f"{dataDir}hi_{groupName}.csv", index=False)

	if return_data is not None:
		return hi_list
	
	fig, ax = plt.subplots()
 
	ax.boxplot(hi_list)
	# print(hi_list, np.median(hi_list))
	ax.scatter([1]*len(hi_list), hi_list)
	# print(hi_list)
	# ax.set_xticklabels([round(p, 2) for p in percent_reached])
	ax.set_ylabel('Heading Index')
	ax.set_ylim([-1, 1])

	fig.suptitle(groupName)

	fig.tight_layout()
	fig.savefig(outputDir + 'heading_index_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'heading_index_'+groupName+'.png')

	fig.clf()
	plt.close(fig)

def acf(groupName, plot_dir = 'explore', out_to=10):
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)

	all_angvels = []
	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()
		all_angvels.append(vels[:,2])

	num = int(np.round(settings['fps']*out_to))
	corrs = [1]

	for t in range(1, num+1):

		p1 = []
		p2 = []
		for angvel in all_angvels:
			p1.extend(list(angvel[:-t]))
			p2.extend(list(angvel[t:]))
		
		p = np.corrcoef(p1, p2)[0,1]
		# ax.scatter(p1, p2)
		# print(p)
		corrs.append(p)
	
	fig, ax = plt.subplots()

	ax.plot(np.arange(num+1)/settings['fps'], corrs)
	ax.axhline(0, color='red', linewidth=1)

	ax.set_xlabel('Lags (seconds)')
	ax.set_ylabel('Autocorrelation')

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + 'autocorr_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'autocorr_'+groupName+'.png', transparent=True)
	fig.clf()
	plt.close(fig)

def get_stop_seq(speed, dt, min_stop_dt = 0.2):
	min_stop_t = int(np.round(min_stop_dt/dt))
	stopseq = schmitt_trigger(speed, 0.1, 0.25)
	segments = indices_grouped_by_condition(stopseq, lambda x: x == 0)

	for seg in segments:
		if len(seg)<=min_stop_t:
			stopseq[seg[0]:seg[-1]+1] = 1
		else:
			continue

	return stopseq

def get_stops(fin):

	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
	f1.close()

	speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)

	stopseq = get_stop_seq(speed, 1/settings['fps'])
	result_segments = indices_grouped_by_condition(stopseq, lambda x: x == 0)

	stop_inds = []
	stop_lengths = []

	for seg in result_segments:
		start = seg[0] 
		end = seg[-1]
		stopdist = np.linalg.norm(pos[start,:]-pos[end,:])
		if stopdist>2:
			print('Large stop distance in file:', fin)
			# continue
		middle = int(np.round((start+end)/2))
		stop_inds.append(middle)
		stop_lengths.append((end-start+1)/settings['fps'])

	return stop_inds, stop_lengths

def get_stops2(fin, speed_threshold = 0.25):

	f1 = open(fin, 'rb')
	pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
	f1.close()

	speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)

	condition = lambda x: x < speed_threshold
	result_segments = indices_grouped_by_condition(speed, condition)
	stop_inds = []
	stop_lengths = []

	for seg in result_segments:
		if len(seg)<=int(1*settings['fps']):
			# print('small stop')
			continue
		start = seg[0] 
		end = seg[-1]
		stopdist = np.linalg.norm(pos[start,:]-pos[end,:])
		if stopdist>2:
			print('Large stop distance in file:', fin)
			continue
		middle = int(np.round((start+end)/2))
		stop_inds.append(middle)
		stop_lengths.append((end-start+1)/settings['fps'])

	return stop_inds, stop_lengths

def avp_statistics(groupName, speed_threshold=None, plot_dir = 'explore'):
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	
	all_context_angles = []
	all_context_lengths = []
	all_context_pairs = []
	all_context_pair_lengths = []

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue
		
		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _  = pickle.load(f1)
		f1.close()
		stageH, stageW = settings['stageH'], settings['stageW']

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		
		# Need this to tell what's too close to boundaries
		bl = bl_default
		if speed_threshold is not None:
			inframe = lambda idx: (pos[idx,1]>bl) & (pos[idx,1]<(stageH-bl)) & (pos[idx,0]>bl) & (pos[idx,0]<(stageW-bl)) & (speed[idx]<5) & (speed[idx]>speed_threshold) 
		else:
			inframe = lambda idx: (pos[idx,1]>bl) & (pos[idx,1]<(stageH-bl)) & (pos[idx,0]>bl) & (pos[idx,0]<(stageW-bl)) & (speed[idx]<5)
		
		peaks = get_angvelpeak(inputDir + file)
		peaks = np.array([0] + peaks + [-1])
		
		# First we get pairs of points. This lets us create a direction. 
		pairs = []
		context_angles = []
		context_lengths = []
		for i in range(len(peaks)-1):
			start = peaks[i]
			end = peaks[i+1]
			pos1 = pos[start,:]
			pos2 = pos[end,:]
			if (inframe(start) and inframe(end)):
				pairs.append((pos1, pos2))
				vector = pos2 - pos1
				context_angles.append(np.arctan2(vector[1], vector[0]) % (2*np.pi))
				context_lengths.append(np.linalg.norm(vector))
			else:
				pairs.append(None)
				context_angles.append(None)
				context_lengths.append(None)
		all_context_angles.append(context_angles)
		all_context_lengths.append(context_lengths)
		# This is to see what directions follow up what other directions
		context_pairs = []
		context_pair_lengths = []
		for i in range(len(pairs)-1):
			if pairs[i] is not None and pairs[i+1] is not None:
				context_pairs.append((context_angles[i], context_angles[i+1]))
				context_pair_lengths.append((context_lengths[i], context_lengths[i+1]))
		all_context_pairs.append(context_pairs)
		all_context_pair_lengths.append(context_pair_lengths)
	
	# Processing lists into something usable
	all_context_angles = list(itertools.chain(*all_context_angles))
	all_context_pairs = list(itertools.chain(*all_context_pairs))
	all_context_pair_lengths = list(itertools.chain(*all_context_pair_lengths))
	all_context_lengths = list(itertools.chain(*all_context_lengths))

	all_context_angles = [ctx for ctx in all_context_angles if ctx is not None]
	all_context_lengths = [ctx for ctx in all_context_lengths if ctx is not None]
	all_context_pairs = np.array(all_context_pairs)
	all_context_pair_lengths = np.array(all_context_pair_lengths)

	# For plotting all context angles
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	num_bins = 6
	colors = ['cyan','grey','pink','pink','grey','cyan']
	hist, bins = np.histogram(all_context_angles, bins=num_bins, range=(0, 2 * np.pi))
	hist = hist/len(all_context_angles)

	barbins = bins[:-1] + np.pi / num_bins
	ax.bar(barbins, hist, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
	ax.set_yticklabels([])
	ax.set_yticks([0,0.1,0.2,0.3])
	ax.set_ylim([0,0.4])
	ax.set_title('Percentage angle traveled when moving')
	
	# fig.savefig(outputDir + 'hinge_angles_'+groupName+'.pdf', transparent=True)
	# fig.savefig(outputDir_png + 'hinge_angles_'+groupName+'.png', transparent=True)
	fig.clf()
	plt.close(fig)

	# Plotting context pairs
	ctx1 = all_context_pairs[:,0]
	ctx2 = all_context_pairs[:,1]
	inds1 = np.digitize(ctx1, bins)
	inds2 = np.digitize(ctx2, bins)
	# Essentially this will map bins 123 to 123 and 456 to 321
	inds1 = inds1 + (inds1>3)*(7 - 2*inds1)
	inds2 = inds2 + (inds2>3)*(7 - 2*inds2)
	
	# Testing here
	fig, ax = plt.subplots()
	test1 = all_context_pair_lengths[inds1==2,0]
	test2 = -inds2[inds1==2]+2
	sns.regplot( x=test1, y=test2, ax = ax)

	fig.tight_layout()
	fig.savefig(outputDir + 'context_lengths_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'context_lengths_'+groupName+'.png')
	fig.clf()
	plt.close(fig)


	# for i in range(3):
	# 	t = inds2[inds1==(i+1)]
	# 	# Use Counter to count occurrences
	# 	counter = Counter(t)

	# 	# Calculate probabilities
	# 	total_occurrences = sum(counter.values())
	# 	probabilities = {key: count / total_occurrences for key, count in counter.items()}

	# 	print(probabilities)

	# Plot to see conditional distribution
	nbins = 3
	bins = np.linspace(0.5, 3.5, nbins+1)

	hist, _ = np.histogram(inds1, bins=bins)
	p1 = hist/len(inds1)
	hist, _ = np.histogram(inds2, bins=bins)
	p2 = hist/len(inds2)
	hist, _, _ = np.histogram2d(inds1, inds2, bins=bins)
	p12 = hist/len(inds1)
	
	# Calculate conditional distribution P(angle2 | angle1)
	p_angle2_given_angle1 = p12 / p1[:, None]

	fig, ax = plt.subplots()
	imshow_plot = ax.imshow(p_angle2_given_angle1)

	# Add numbers on the imshow plot
	for i in range(nbins):
		for j in range(nbins):
			ax.text(j, i, f'{p_angle2_given_angle1[i, j]:.2f}', ha='center', va='center', color='white')

	# Set xticks and labels
	ticks = np.linspace(0, nbins-1, nbins)
	ticklabels = ['Good', 'Neutral', 'Bad']

	# Add lines to middle to differentiate
	for y in (ticks[:-1]+0.5):
		ax.axhline(y, color='red', linestyle='--')

	fig.gca().invert_yaxis()
	ax.set(xticks=ticks, xticklabels=ticklabels)
	ax.set(yticks=ticks, yticklabels=ticklabels)
	ax.set(xlabel='Context_after', ylabel='Context_before')
	ax.set(title = 'P(Context_after | Context_prev)')
	fig.suptitle(groupName)

	# Add a colorbar to the plot
	colorbar = plt.colorbar(imshow_plot)
	imshow_plot.set_clim(vmin=0, vmax=0.8)

	# Save plot
	fig.tight_layout()
	fig.savefig(outputDir + 'context_transitions_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'context_transitions_'+groupName+'.png')
	fig.clf()
	plt.close(fig)

def stop_statistics(groupName, speed_threshold = 0.25, plot_dir = 'explore', mode = 'time', return_data = None):
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	
	all_stop_lengths = []
	all_stop_x = []
	num_windows = 3
	win_dict = {i: [] for i in range(1, num_windows+1)}

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue
		
		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()
		stageH, stageW = settings['stageH'], settings['stageW']
		
		# Need this to tell what's too close to boundaries
		bl = bl_default
		inframe = lambda pos: (pos[1]>bl) & (pos[1]<(stageH-bl)) & (pos[0]>0.2*stageW) & (pos[0]<0.8*stageW)

		# This is used to divide up the frame into thirds (or whatever you want)
		win_bins = np.linspace(0.2*stageW, 0.8*stageW, 3+1)	# imposed technical limit

		# Get stops indices and lengths
		stop_inds, stop_lengths = get_stops(inputDir + file)

		# Iterate through stops to get only within frame ones. 
		for i in range(len(stop_inds)):
			stop = pos[stop_inds[i], :]
			if inframe(stop):	# check in frame
				all_stop_lengths.append(stop_lengths[i])
				all_stop_x.append(stop[0])
				# Also seperate out stop lengths based on where in border
				window_number = np.digitize(stop[0], win_bins)
				win_dict[window_number].append(stop_lengths[i])

	if return_data == True:
		pass
		# sns.histplot(x=all_stop_lengths, ax = ax, stat="density", kde=True, bins=10)

	if mode == 'time':
		# First plot, just overall distribution of stops
		fig, ax = plt.subplots()
		ax.set_xscale('log')
		sns.histplot(x=all_stop_lengths, ax = ax, stat="density", kde=True, bins=10)

		if return_data == True:
			kde_plot = ax.lines[0] 
			kde_x = kde_plot.get_xdata()
			kde_y = kde_plot.get_ydata()
			plt.close('all')
			return kde_x, kde_y

		ax.set_xlim([0.2, 300])
		ax.axvline(np.median(all_stop_lengths), color='red', label='Median')
		ax.set_xlabel('Time (s)')
		ax.legend()
		ax.set_title('Distribution of stop durations')

		fig.tight_layout()
		fig.savefig(outputDir + 'stop_distribution_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + 'stop_distribution_'+groupName+'.png', transparent=True)
		fig.clf()
		plt.close(fig)

	elif mode == 'stratify':
		# Second plot, distribution seperated into windows
		fig, ax = plt.subplots(3,1, figsize=(4,9))
		for i in range(num_windows):
			ax[i].set_xscale('log')
			sns.histplot(x=win_dict[i+1], ax = ax[i], stat="density", kde=True, bins=10)
			ax[i].axvline(np.median(win_dict[i+1]), color='red', label='Median')
			ax[i].set_xlabel('Time (s)')
			ax[i].legend()
			ax[i].set_title(f'Window {i+1}, Number of stops: {len(win_dict[i+1])}')

		# Flexible axis limit
		min_val_x = min(a.get_xlim()[0] for a in ax)
		max_val_x = max(a.get_xlim()[1] for a in ax)
		min_val_y = min(a.get_ylim()[0] for a in ax)
		max_val_y = max(a.get_ylim()[1] for a in ax)
		for a in ax:
			a.set_xlim(min_val_x, max_val_x)
			a.set_ylim(min_val_y, max_val_y)

		fig.suptitle('Distribution of stop lengths when divided into segments')

		fig.tight_layout()
		fig.savefig(outputDir + 'stop_distribution_windowed_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + 'stop_distribution_windowed_'+groupName+'.png', transparent=True)
		fig.clf()
		plt.close(fig)

	elif mode == 'location':
		# First plot, just overall distribution of stops
		fig, ax = plt.subplots()
		sns.histplot(x=all_stop_x, ax = ax, stat="density", kde=True, bins=20)
		if return_data == True:
			kde_plot = ax.lines[0] 
			kde_x = kde_plot.get_xdata()
			kde_y = kde_plot.get_ydata()
			plt.close('all')
			return kde_x, kde_y
		ax.axvline(np.median(all_stop_x), color='red', label='Median')
		ax.set_xlabel('Location (x)')
		ax.legend()
		ax.set_title('Distribution of stop x locations')

		fig.tight_layout()
		fig.savefig(outputDir + 'stop_locations_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + 'stop_locations_'+groupName+'.png', transparent=True)
		fig.clf()
		plt.close(fig)
	
	elif mode == 'joint':
		# Generate random data
		y = all_stop_lengths
		x = all_stop_x

		# Create logarithmic bins for the x-axis
		y_bins = np.logspace(np.log10(0.2), np.log10(300), 10+1)
		# Linear bins for the y-axis
		x_bins = np.linspace(7, 28, 7+1)

		# Compute 2D histogram
		hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

		fig, ax = plt.subplots(figsize=(8, 6))
		im = ax.imshow(hist.T, extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()], aspect='auto', origin='lower', cmap='viridis')
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_label('Frequency')
		ax.set_xlabel('Stop Location')
		ax.set_ylabel('Stop Duration')
		ax.set_title('Joint Distribution')

		# Set yticks to linspace
		ax.set_xticks(x_bins)
  		# Set y-axis ticks at every power of 10
		y_tick_positions = np.linspace(y_bins.min(), y_bins.max(), 10+1)
		ax.set_yticks(y_tick_positions)
		# Set y-axis tick labels as powers of 10
		y_tick_labels = ['$10^{{{:.2f}}}$'.format(np.log10(pos)) for pos in y_tick_positions]
		ax.set_yticklabels(y_tick_labels)

		fig.tight_layout()
		fig.savefig(outputDir + 'joint_stops_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + 'joint_stops_'+groupName+'.png', transparent=True)
		fig.clf()
		plt.close(fig)
