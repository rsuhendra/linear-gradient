from functions_plotting import *
from functions_plotting2 import *
from sklearn.linear_model import LinearRegression
from statsmodels.stats.proportion import proportions_ztest


def stop_threshold_stats(groupNames, thresh = 0.25):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)

	for groupName in groupNames:
		overall_speed = extract_quantity_from_region(groupName=groupName, region='all', mode='speed', speed_threshold=None)
		speed_over = extract_quantity_from_region(groupName=groupName, region='all', mode='speed', speed_threshold=thresh)
		speed_under = extract_quantity_from_region(groupName=groupName, region='all', mode='speed', speed_threshold=thresh, invert=True)

		overall_angvel = extract_quantity_from_region(groupName=groupName, region='all', mode='angvel', speed_threshold=None)
		angvel_over = extract_quantity_from_region(groupName=groupName, region='all', mode='angvel', speed_threshold=thresh)
		angvel_under = extract_quantity_from_region(groupName=groupName, region='all', mode='angvel', speed_threshold=thresh, invert=True)

		data = {'GroupName': groupName, 'OverallSpeed': np.mean(overall_speed), 'SpeedUnder': np.mean(speed_under), 'SpeedOver': np.mean(speed_over), 'ProportionTimeUnder': len(speed_under)/len(overall_angvel), 'OverallAngvel': np.mean(overall_speed), 'AngvelUnder': np.mean(angvel_under), 'AngvelOver': np.mean(angvel_over)}

		# print(data)

def predict_taxis(groupNames):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)

	df = pd.DataFrame(columns=['Genotype', 'speed_g', 'speed_n', 'speed_b', 'rotvel_g', 'rotvel_n', 'rotvel_b', 'speed_prop', 'rotvel_prop', 'reached', 'med_dist'])

	for k, groupName in enumerate(groupNames):
		inputDir = 'outputs/outputs_' + groupName + '/'
		dirs = os.listdir(inputDir)

		barbins, mag = velocity_plot(groupName, region='all', mode='vel', plot=False)
		# good, neutral, bad context
		tot = np.sum(mag)
		gc, nc, bc = (mag[0]+mag[5])/tot, (mag[1]+mag[4])/tot, (mag[2]+mag[3])/tot

		barbins, mag = angvels_plot(groupName, region='all', plot=False)
		tot = np.sum(mag)
		gc2, nc2, bc2 = (mag[0]+mag[5])/tot, (mag[1]+mag[4])/tot, (mag[2]+mag[3])/tot

		reached = 0
		total = 0
		cumDist = []

		for file in dirs:
			if 'output' not in file.split('/')[-1]:
				continue

			f1 = open(inputDir + file, 'rb')
			pos, angles, vels, settings, originalTrackingInfo, _  = pickle.load(f1)
			f1.close()

			# use only first 10 mins of data
			tenMin = 600*settings['fps']
			if len(pos)>tenMin:
				pos = pos[:tenMin]

			lineDist = 0.8
			normalized_x = pos[:,0]/settings['stageW']
			ind = next((i for i in range(len(pos)) if normalized_x[i] > lineDist), None)
			reached = reached + 1 if ind is not None else reached
			total += 1
			
			distances = np.linalg.norm(pos[1:,] - pos[:-1,], axis=1)
			cumulative_distances = np.cumsum(distances)
			lineFirstHitDist = cumulative_distances[ind-1] if ind is not None else  cumulative_distances[-1]
			cumDist.append(lineFirstHitDist)
		

		row = pd.DataFrame([{'Genotype': groupName, 'speed_g': gc, 'speed_n': nc, 'speed_b': bc, 'rotvel_g': gc2, 'rotvel_n': nc2, 'rotvel_b': bc2, 'speed_prop':gc/bc, 'rotvel_prop':gc2/bc2, 'reached': reached/total, 'med_dist': np.median(cumDist)}])
		
		df = pd.concat([df, row], ignore_index=True)
	
	print(df.sort_values(by='reached', ascending=False))

	# Split features and target variable
	X = df[['speed_g', 'speed_b', 'rotvel_g', 'rotvel_b']]  # Multiple predictor variables
	# X = df[['speed_prop', 'rotvel_prop']]  # Multiple predictor variables
	y = df['reached']
	# y = df['med_dist']

	# Initialize and fit the linear regression model
	model = LinearRegression()
	model.fit(X, y)
	r_squared = model.score(X, y)

	# Print the intercept and coefficients
	print('Intercept:', model.intercept_)
	print('Coefficients:', model.coef_)
	print('R-squared:', r_squared)

def all_reach_plot(groupNames, mode='distance', tenMin = False):

	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)
	
	mydict = {}
	all_percs = []

	for k, groupName in enumerate(groupNames):
		inputDir = 'outputs/outputs_' + groupName + '/'
		dirs = os.listdir(inputDir)

		lineDist = 0.8
		time_spent = []
		cumDist = []
		reached = 0

		fig, ax = plt.subplots()
		for file in dirs:
			if 'output' not in file.split('/')[-1]:
				continue

			f1 = open(inputDir + file, 'rb')
			pos, angles, vels, settings, originalTrackingInfo, _  = pickle.load(f1)
			f1.close()

			# use only first 10 mins of data
			if tenMin == True:
				pos = pos[:600*settings['fps']]

			normalized_x = pos[:,0]/settings['stageW']
			ind = next((i for i in range(len(pos)) if normalized_x[i] > lineDist), None)
			reached = reached+1 if ind is not None else reached
			time_spent.append(ind/settings['fps'] if ind is not None else len(pos)/settings['fps'])
			
			distances = np.linalg.norm(pos[1:,] - pos[:-1,], axis=1)
			cumulative_distances = np.cumsum(distances)
			lineFirstHitDist = cumulative_distances[ind-1] if ind is not None else  cumulative_distances[-1]
			cumDist.append(lineFirstHitDist)

		numFiles = len(cumDist)
		dists = [d for d in cumDist]
		percent_reached = reached/numFiles
		all_percs.append(percent_reached)

		title = f'{groupName}\n{np.round(percent_reached, 2)}'

		if mode == 'distance':
			mydict[title] = dists
		if mode == 'time':
			mydict[title] = time_spent
		

	# Create a list of dictionaries, each containing the group name and corresponding value
	data = [{'Group': key, 'Value': value} for key, values in mydict.items() for value in values]

	# Convert the list of dictionaries to a DataFrame
	df = pd.DataFrame(data)

	
	# Create the boxplot using Seaborn
	fig, ax = plt.subplots(figsize = (15,5))

	sns.boxplot(data=df, x='Group', y='Value', hue='Group', showfliers=False, ax=ax, palette = plt.cm.viridis(np.array(all_percs)).tolist())
	sns.swarmplot(data=df, x='Group', y='Value', color='k', ax=ax)

	# Add colorbar
	cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax)
	cbar.set_label('Color Scale')

	if mode == 'distance':
		ax.set_ylim([0, 400])
		ax.set_ylabel('Distance travelled (cm)')
	elif mode == 'time':
		ax.set_ylabel('Time spent (seconds)')

	ax.tick_params(axis='x', rotation=45)

	message = 'dist_reached_group_'
	if tenMin == True:
		message = message + 'tenMin_'

	fig.tight_layout()
	fig.savefig(f'{outputDir}{message}{mode}.pdf', transparent=True)
	fig.savefig(f'{outputDir_png}{message}{mode}.png')
	
	fig.clf()
	plt.close('all')


def all_stop_distribution(groupNames, mode = 'time'):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)

	fig, ax = plt.subplots()

	for k, groupName in enumerate(groupNames):
		controlGroups = ['WT', 'Kir-+', 'SS98+', 'SS90+', 'SS00096-+', 'SS408+']
		kde_x, kde_y = stop_statistics(groupName, mode = mode, return_data=True)
		if groupName in controlGroups:
			ax.plot(kde_x, kde_y, label = groupName, color='black')
		else:
			ax.plot(kde_x, kde_y, label = groupName)

	ax.legend()
	if mode == 'time':
		ax.set_xscale('log')
		ax.set_xlim([0.2, 300])
		ax.set_xlabel('Time (s)')
		ax.set_title('Distribution of stop durations')
	elif mode == 'location':
		ax.set_xlim([5, 30])
		ax.set_xlabel('Location (x)')
		ax.set_title('Distribution of stop x locations')

	fig.tight_layout()
	fig.savefig(outputDir + f'all_stop_{mode}.pdf', transparent=True)
	fig.savefig(outputDir_png + f'all_stop_{mode}.png')

	fig.clf()
	plt.close('all')

def all_num_turns(groupNames, ht=np.pi/3, speed_threshold=None):
	sns.reset_orig()
	sns.set_theme(style="ticks", font="Arial", font_scale=1.4)
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))

	# --- Matching color palette with the other function ---
	colors = sns.color_palette("Dark2", n_colors=len(groupNames))


	controlGroups = ['Kir-+', 'SS98+', 'SS90+', 'SS00096-+', 'SS408+']

	for idx, groupName in enumerate(groupNames):

		mid_angles, ratio = num_turns(
			groupName,
			ht=ht,
			speed_threshold=speed_threshold,
			mode='ash',
			return_data=True
		)

		# --- Plot style consistent with all_turn_distribution_ash ---
		if groupName in controlGroups:
			ax.plot(
				mid_angles,
				60 * ratio,
				linewidth=2.2,
				color='black',
				label=groupName,
				alpha=0.9
			)
		else:
			ax.plot(
				mid_angles,
				60 * ratio,
				linewidth=2.2,
				color=colors[idx],
				label=groupName,
				alpha=0.9
			)

	# --- Style tweaks identical to the other function ----------------

	ax.set_theta_zero_location('E')   # 0° at E (matching)
	ax.set_theta_direction(1)         # clockwise

	ax.set_thetagrids(
		range(0, 360, 45),
		labels=['0°','45°','90°','135°','180°','225°','270°','315°']
	)

	# Y-axis limits (based on your logic)
	ax.set_ylim([0, 60])

	# Radial ticks cleaner
	ax.set_rticks([15, 30, 45, 60])
	ax.tick_params(labelsize=10)

	# Grid style consistent
	ax.grid(color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

	# Legend consistency
	ax.legend(
		loc='upper right',
		bbox_to_anchor=(1.35, 1.1),
		frameon=False,
		fontsize=10
	)

	fig.suptitle('Number of turns per minute', fontsize=14)

	fig.tight_layout()

	fig.savefig(outputDir + 'all_num_turns_ash.pdf', transparent=True)
	fig.savefig(outputDir_png + 'all_num_turns_ash.png', dpi=300)

	plt.close(fig)
	sns.reset_orig()



def all_polar_ratios(groupNames, ht = np.pi/2, speed_threshold = None):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)

	fig, ax = plt.subplots()
	counts, nobs = [], []
	for k, groupName in enumerate(groupNames):
		Ns = polar_turns(groupName, ht = ht, speed_threshold = speed_threshold, return_data=True)
		# print(Ns)
		counts.append(Ns[1][0] + (Ns[3][1]-Ns[3][0]))
		nobs.append(Ns[1][1] + Ns[3][1])

	counts = np.array(counts)
	nobs = np.array(nobs)
	alpha = 0.05
	prop, ci_low, ci_upp = proportion_conf(counts, nobs, alpha)
	bar_width = 0.7

	ax.axhline(0.5, color='red')
	bars = ax.bar(groupNames, prop, bar_width, yerr=[prop - ci_low, ci_upp - prop], capsize=5, color='green', edgecolor='black')
	ax.bar(groupNames, 1-prop, bar_width, bottom =prop, capsize=5, color='purple', edgecolor='black')

	for bar in bars:
		height = bar.get_height()
		ax.annotate(f'{height:.2f}',  # Format to 2 decimal places
			xy=(bar.get_x() + bar.get_width() / 2, 0.35),  # X and Y position
			ha='center', va='bottom', fontsize = 8)  # Horizontal and vertical alignment

	# ax.legend()
	ax.set_ylim([0,1])
	ax.tick_params(axis='x', rotation=45)
	ax.set_title(f'Ratio of correct turns')

	fig.tight_layout()
	fig.savefig(outputDir + f'all_polar_ratios.pdf', transparent=True)
	fig.savefig(outputDir_png + f'all_polar_ratios.png')

	fig.clf()
	plt.close('all')


def all_turn_boxplots(groupNames, ht=np.pi/2, speed_threshold = None, angle_threshold = None, mode = 'average', plot_dir = 'explore'):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)
	
	data = []
	for k, groupName in enumerate(groupNames):

		all_turns, in_box_angles = get_turns(groupName, ht=ht, speed_threshold=speed_threshold, angle_threshold=angle_threshold, ignore = True)

		for i in range(len(all_turns)):
			num_turns = len(all_turns[i])
			duration = in_box_angles[i]
			if duration < 10:
				continue
			data.append({'Group': groupName, 'Average_Turns': 60*num_turns/duration, 'Turns': num_turns})
	
	df = pd.DataFrame(data)

	fig, ax = plt.subplots(figsize = (15,5))

	if mode == 'average':
		category = "Average_Turns"
		ytitle = 'Average number of turns per min'
	elif mode == 'turn':
		category = "Turns"
		ytitle = 'Number of turns per fly'

	sns.boxplot(data=df, x="Group", y=category, ax=ax)
	sns.swarmplot(data=df, x='Group', y=category, color='k', ax=ax)

	ax.set_title(ytitle)

	fig.tight_layout()
	fig.savefig(outputDir + f'boxplot_{mode}.pdf', transparent=True)
	fig.savefig(outputDir_png + f'boxplot_{mode}.png')

	fig.clf()
	plt.close('all')
 

def all_lr_diffs(groupNames, mode = 'angle', speed_threshold = None, angvel_threshold = None):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)
 
	# Create an empty DataFrame with predefined columns
	df = pd.DataFrame(columns=['groupName', 'diff'])

	baseline_lrs = lr_diff('WT', mode=mode,speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	baseline_lrs = np.array(baseline_lrs)
	
	baseline_diffs = 100*baseline_lrs[:, 0] if (mode == 'angle') else baseline_lrs[:, 0] - baseline_lrs[:, 1]
 
 
	for k, groupName in enumerate(groupNames):
		lrs = lr_diff(groupName, mode=mode,speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
		lrs = np.array(lrs)
  
		diffs = 100*lrs[:, 0] if (mode == 'angle') else lrs[:, 0] - lrs[:, 1]

		# Run 2 sample ks test on WT vs X efficiency
		if mode == 'angvel':
			ks_stat = stats.ks_2samp(baseline_diffs, diffs, alternative='greater')
		else:
			ks_stat = stats.ks_2samp(baseline_diffs, diffs, alternative='lesser')
		color = 'blue' if ks_stat.pvalue > 0.05 else 'red'
		
		# Add the values to the DataFrame
		new_data = pd.DataFrame({'groupName': [groupName] * len(diffs), 'diff': diffs, 'color': color})
		df = pd.concat([df, new_data], ignore_index=True)
  
	# Create the figure and axes
	fig, ax = plt.subplots(figsize=(8, 6))
 
	# Create the boxplot
	sns.boxplot(x='groupName', y='diff', hue='groupName', data=df, palette=df.set_index('groupName')['color'].to_dict(), ax=ax)

	# Customize the plot
	ax.set_xticks(range(len(ax.get_xticklabels())))  # Ensure fixed tick positions
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	ax.set_xlabel('Group Name')
 
	if mode == 'angle':
		ax.set_ylim(0, 100)
		ax.set_title('Percentage of time going down-gradient')
		ax.set_ylabel('Percentage')
	elif mode == 'speed':
		ax.set_title('Down speed - Up speed')
		ax.set_ylabel('Speed difference')
	elif mode == 'angvel':
		ax.set_title('Down angvel - Up angvel')
		ax.set_ylabel('Speed difference')

	fig.tight_layout()
	fig.savefig(outputDir + f'all_lr_diff_{mode}.pdf', transparent=True)
	fig.savefig(outputDir_png + f'all_lr_diff_{mode}.png')
	fig.clf()
	plt.close('all')
	
 
def all_efficiencies(groupNames, ):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)

 
	# Create an empty DataFrame with predefined columns
	df = pd.DataFrame(columns=['groupName', 'effx_val'])

	baseline, base_ct = efficiency_plot('WT', return_data=True)
	percent_reached = []
	
	for k, groupName in enumerate(groupNames):
		effx, ct = efficiency_plot(groupName, return_data=True)

		percent_reached.append(ct/len(effx))

		# Run 2 sample ks test on WT vs X efficiency
		ks_stat = stats.ks_2samp(baseline, effx, alternative='lesser')
		eff_color = 'blue' if ks_stat.pvalue > 0.05 else 'red'

		# Run 2 sample proportion test on WT vs X percent reached goal
		success_cnts = np.array([ct, base_ct])
		total_cnts = np.array([len(effx),len(baseline)])
		test_stat, pval = proportions_ztest(count=success_cnts, nobs=total_cnts, alternative='two-sided')
		perc_color = 'blue' if pval > 0.05 else 'red'
		
		# Add the values to the DataFrame
		new_data = pd.DataFrame({'groupName': [groupName] * len(effx), 'effx_val': effx, 'eff_color': eff_color, 'perc_color': perc_color})
		df = pd.concat([df, new_data], ignore_index=True)

	# Create the figure and axes
	fig, ax = plt.subplots(figsize=(8, 6))

	# Create the boxplot
	sns.boxplot(x='groupName', y='effx_val', data=df, palette=df.set_index('groupName')['eff_color'].to_dict(), ax=ax)


	bar_colors = [df[df['groupName'] == group]['perc_color'].iloc[0] for group in groupNames]
	ax.bar(groupNames, percent_reached, color=bar_colors, alpha=0.25, width=0.9, label='Random Value')

	# Customize the plot
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	ax.set_ylim(0, 1)
	ax.set_title('Efficiency (Boxplot) & Percentage Reached (Barplot)')
	ax.set_xlabel('Group Name')
	ax.set_ylabel('Efficiency/Percentage Reached')

	fig.tight_layout()
	fig.savefig(outputDir + f'combined_measure.pdf', transparent=True)
	fig.savefig(outputDir_png + f'combined_measure.png')
	fig.clf()
	plt.close('all')

	# Create the figure and axes
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.boxplot(x='groupName', y='effx_val', data=df, palette=df.set_index('groupName')['eff_color'].to_dict(), ax=ax)

	ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	ax.set_ylim(0, 1)
	ax.set_title('Efficiency')
	ax.set_xlabel('Group Name')
	ax.set_ylabel('Efficiency')

	fig.tight_layout()
	fig.savefig(outputDir + f'efficiency.pdf', transparent=True)
	fig.savefig(outputDir_png + f'efficiency.png')
	fig.clf()
	plt.close('all')

	# Create the figure and axes
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.bar(groupNames, percent_reached, color=bar_colors, alpha=0.25, width=0.9, label='Random Value')

	ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	ax.set_ylim(0, 1)
	ax.set_title('Percent Reached')
	ax.set_xlabel('Group Name')
	ax.set_ylabel('Percent Reached')

	fig.tight_layout()
	fig.savefig(outputDir + f'per_reached.pdf', transparent=True)
	fig.savefig(outputDir_png + f'per_reached.png')
	fig.clf()
	plt.close('all')


def all_heading_indices(groupNames):
	outputDir = 'plots/groupPlots/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/groupPlots/'
	create_directory(outputDir_png)
	
	groupName = 'WT'
	df = pd.read_csv(f'data/{groupName}/hi_{groupName}.csv', index_col=None)
	baseline = np.array(df['hi'])
	
	p_values = {}
	dfs = []
	for k, groupName in enumerate(groupNames):
		df = pd.read_csv(f'data/{groupName}/hi_{groupName}.csv', index_col=None)
		df['groupName'] = groupName
		dfs.append(df)

		# Do quick KS test
		compare = np.array(df['hi'])
		alt = 'two-sided' # 'less'
		p_value = stats.ks_2samp(baseline, compare, alternative=alt).pvalue
		p_values[groupName] = p_value
		# print(f"{groupName}: p-value = {p_value}")

	# Concatenate all DataFrames
	combined_df = pd.concat(dfs, ignore_index=True)
	# print(combined_df)
	
	# Create the figure and axes
	fig, ax = plt.subplots(figsize=(8, 6))
	
	sns.boxplot(data=combined_df, x="groupName", y="hi", ax=ax, showfliers=False)
 
	sns.swarmplot(data=combined_df, x="groupName", y="hi", ax=ax, color=".25", alpha=0.7)  # Adjust color and transparency

	# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
	# ax.set_ylim(-1, 1)
 
	# Significance thresholding function
	def get_asterisks(p):
		if p < 0.001:
			return '***'
		elif p < 0.01:
			return '**'
		elif p < 0.05:
			return '*'
		else:
			return None
 
	ax.set_ylim(-1, 1.1)  # Extend the y-axis slightly above 1
	y_max = 1.00  # Position for asterisks slightly below the top of the range
	# Add significance asterisks
	xticks = ax.get_xticks()
	for i, group in enumerate(groupNames):
		if group in p_values:
			asterisks = get_asterisks(p_values[group])
			if asterisks:
				ax.text(xticks[i], y_max, asterisks, ha='center', fontsize=12, color='red', fontweight='bold')

	ax.set_title(r'$\text{Heading Index: HI} = \frac{1}{T} \int_{0}^{T} \cos(\theta) d\theta$',fontsize=20)
	ax.set_xlabel('Genotype')
	ax.set_ylabel('Heading Index')

	fig.tight_layout()
	fig.savefig(outputDir + f'all_heading_indices.pdf', transparent=True)
	fig.savefig(outputDir_png + f'all_heading_indices.png')
	fig.clf()
	plt.close('all')
 
def all_turn_distribution_ash(
    groupNames,
    ht=np.pi/3,
    mode='turn_ash',
    side=False,
    speed_threshold=None,
    angle_threshold=None,
    plot_dir='explore'
):
    sns.reset_orig()
    sns.set_theme(style="ticks", font="Arial", font_scale=1.4)
    outputDir = 'plots/groupPlots/'
    create_directory(outputDir)
    outputDir_png = 'plots_png/groupPlots/'
    create_directory(outputDir_png)

    # --- Clean, readable color palette (balanced + high contrast) ---
    # You can choose from: "tab10", "Set2", "Dark2", "husl", "viridis"
    colors = sns.color_palette("Dark2", n_colors=len(groupNames))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))

    max_mag = 0

    # --- Plot each group ---
    for i, groupName in enumerate(groupNames):
        mid_angles, mag = turn_distribution(
            groupName,
            ht=ht,
            mode=mode,
            side=side,
            speed_threshold=speed_threshold,
            angle_threshold=angle_threshold,
            plot_dir=plot_dir,
            return_data=True
        )

        ax.plot(
            mid_angles,
            mag,
            linewidth=2.2,
            color=colors[i],
            label=groupName,
            alpha=0.95
        )

        max_mag = max(max_mag, np.max(mag))

    # --- Style --------------------------------------------------

    ax.set_theta_zero_location('E')   # 0° at East
    ax.set_theta_direction(1)         # cw orientation

    ax.set_thetagrids(
        range(0, 360, 45),
        labels=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    )

    # Dynamic radial ticks based on data
    ax.set_rticks(np.linspace(0, 135, 4))

    ax.tick_params(labelsize=10)

    ax.grid(
        color='gray',
        linestyle='--',
        linewidth=0.8,
        alpha=0.7
    )

    # --- Legend placement --------------------------------------
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.35, 1.10),
        frameon=False,
        fontsize=10
    )

    # --- Title ----------------------------
    fig.suptitle('Average turn size by orientation (deg)', fontsize=14)
    fig.tight_layout()

    # --- Save -----------------------------
    fig.savefig(outputDir + f'turn_distribution_{mode}.pdf', transparent=True)
    fig.savefig(outputDir_png + f'turn_distribution_{mode}.png', dpi=300)

    plt.close(fig)
    sns.reset_orig()


### DEPRECATED

# def all_turn_distribution(groupNames, ht = np.pi/2, speed_threshold = None):
# 	outputDir = 'plots/groupPlots/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/groupPlots/'
# 	create_directory(outputDir_png)

# 	fig, ax = plt.subplots()

# 	for k, groupName in enumerate(groupNames):
# 		controlGroups = ['WT', 'Kir-+', 'SS98+', 'SS90+', 'SS00096-+', 'SS408+']
# 		kde_x, kde_y = turn_distribution(groupName, ht=ht, speed_threshold=speed_threshold, mode='angle', plot_dir='important', return_data=True)

# 		if groupName in controlGroups:
# 			ax.plot(kde_x, kde_y, label = groupName, color='black')
# 		else:
# 			ax.plot(kde_x, kde_y, label = groupName)

# 	ax.legend()
# 	ax.set_xlim([0, 360])
# 	ax.set_xlabel('Angle (deg)')

# 	fig.tight_layout()
# 	fig.savefig(outputDir + f'all_turn_distributions.pdf', transparent=True)
# 	fig.savefig(outputDir_png + f'all_turn_distributions.png')

# 	fig.clf()
# 	plt.close('all')


# def all_up_down(groupNames, ht=np.pi/2, speed_threshold = None, angle_threshold = None, mode = 'speed', plot_dir = 'explore'):
# 	outputDir = 'plots/groupPlots/'
# 	create_directory(outputDir)
# 	outputDir_png = 'plots_png/groupPlots/'
# 	create_directory(outputDir_png)

# 	ups = []
# 	downs = []
# 	for k, groupName in enumerate(groupNames):
		
# 		if mode == 'speed':
# 			mag = velocity_plot(groupName, region='all', mode='vel', return_data='All', speed_threshold=speed_threshold)
# 		elif mode == 'angvel':
# 			mag = angvels_plot(groupName, region='all', return_data='All', speed_threshold=speed_threshold)
		
# 		up = list(mag[0]) + list(mag[5])
# 		down = list(mag[2])+ list(mag[3])

# 		ks_stat = stats.ks_2samp(up, down)
# 		print(groupName, ks_stat)
  
# 		ups.append(np.average(up))
# 		downs.append(np.average(down))

# 	ups = np.array(ups)
# 	downs = np.array(downs)

# 	fig, ax = plt.subplots(figsize = (15,5))

# 	if mode == 'turns':
# 		ytitle = 'Average number of turns per min'
# 	elif mode == 'speed':
# 		ytitle = 'Average speed (cm/s)'
# 	elif mode == 'angvel':
# 		ytitle = 'Average angular velocity (rad/s)'

# 	# Bar width
# 	bar_width = 0.35

# 	# Positions of the bars on the x-axis
# 	r1 = np.arange(len(groupNames))
# 	r2 = [x + bar_width for x in r1]

# 	bars1 = ax.bar(r2, ups, color='green', width=bar_width, edgecolor='black', label = 'Up gradient')
# 	bars2 = ax.bar(r1, downs, color='red', width=bar_width, edgecolor='black', label = 'Down gradient')

# 	ax.set_xticks([r + bar_width / 2 for r in range(len(groupNames))])
# 	ax.set_xticklabels(groupNames)
# 	ax.set_title(ytitle)
# 	ax.legend()

# 	fig.tight_layout()
# 	fig.savefig(outputDir + f'up_down_{mode}.pdf', transparent=True)
# 	fig.savefig(outputDir_png + f'up_down_{mode}.png')
# 	fig.clf()
# 	plt.close('all')
 