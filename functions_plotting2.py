from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
from matplotlib.colors import ListedColormap

bl_default = 2

def extract_quantity_from_region(groupName, region, mode, speed_threshold=None, angvel_threshold=None, invert=False, unmixed = False):
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	
	all_scalars= []

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angvels = np.abs(vels[:, 2])

		stageW, stageH = settings['stageW'], settings['stageH']
		if settings['gap']>0:
			gapL, gapR = settings['gapL'], settings['gapR']
		else:
			gapL, gapR = 0.5, 0.5

		# check only middle ones
		bl = bl_default	# border length
		inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (speed<5)
		
		if speed_threshold is not None:
			if invert == False:	
				inds = inds & (speed>=speed_threshold) 
			else:
				inds = inds & (speed<speed_threshold) 

		# if angvel_threshold is not None:
		# 	if invert == False:	
		# 		inds = inds & (angvels<angvel_threshold[1]) & (angvels>angvel_threshold[0])
		# 	else:
		# 		inds = inds & (angvels<angvel_threshold[1]) & (angvels>angvel_threshold[0])

		# splitting middle region into windows
		# num_windows = 3
		win_bins = np.linspace(0.2*stageW, 0.8*stageW, 3+1)

		if region == 'all':
			inds = (inds & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl)))

		if region == 'between':
			inds = (inds & (pos[:,0]>gapL*stageW) & (pos[:,0]<gapR*stageW))
		elif region == 'left':
			inds = (inds & (pos[:,0]>bl) & (pos[:,0]<gapL*stageW))
		elif region == 'right':
			inds = (inds & (pos[:,0]>gapR*stageW) & (pos[:,0]<(stageW-bl)))

		elif isinstance(region, int):
			inds = (inds & (pos[:,0]>win_bins[region]) & (pos[:,0]<win_bins[region+1]))
		
		if mode=='speed':
			scalar = speed[inds]
		elif mode=='angvel':
			scalar = angvels[inds]
		elif mode=='travel':
			scalar = np.arctan2(vels[inds,1], vels[inds,0])%(2*np.pi)
		elif mode == 'head':
			scalar = angles[inds]%(2*np.pi)
		elif mode == 'time':
			scalar = [np.sum(inds)/settings['fps']]
		elif mode == 'location':
			scalar = pos[inds, 0]
		else:
			# Mistake case
			print('Something went wrong! A typo?')
		
		all_scalars.append(scalar)

	if unmixed == True:
		return all_scalars
     
	return np.concatenate(all_scalars)
	

def value_plot(groupName, region='all', mode='speed', speed_threshold=None, weighted=False, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_values = extract_quantity_from_region(groupName=groupName, region=region, speed_threshold=speed_threshold, mode=mode)
	if mode == 'angvel':
		all_values = (180/np.pi)*all_values

	fig, ax = plt.subplots()

	if weighted ==  False:
		numbins = 1000
		ax.hist(all_values, bins=numbins, density=True)
		ax.set_ylabel('Probability density')
	else:
		counts, bins = np.histogram(all_values, bins=1000)
		barbins = (bins[:-1] + bins[1:])/2
		ax.bar(bins[:-1], counts*barbins, width=np.diff(bins), align='edge', edgecolor='black')
		ax.set_ylabel('xP(x)')

	ax.set_yscale('log')
	ax.axvline(np.median(all_values), color='red', linewidth=0.2)
	if mode == 'speed':
		ax.set_xlabel('Speed (cm/s)')
		ax.set_xlim([0, 5])
		# ax.set_ylim([0, ylim_max])
	elif mode == 'angvel':
		ax.set_xlabel('Angvel (deg/s)')
		ax.set_xlim([0, 540])

	
	message = '_distribution_'
	if weighted ==  True:
		message = '_weighted' + message

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + region + message + mode + '_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + region + message + mode + '_' +groupName+'.png')
	fig.clf()
	plt.close(fig)
 

from scipy.stats import laplace

def angvel_dist_plot(groupName, region='all', speed_threshold=None, plot_dir='explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)
	
	mode = 'angvel'

	all_values = extract_quantity_from_region(groupName=groupName, region=region, speed_threshold=speed_threshold, mode=mode)
 
	# all_values = all_values[all_values < 2*np.pi]
	
	# if mode == 'angvel':
	#     all_values = (180/np.pi)*all_values  # Commented out to keep rad/s

	fig, ax = plt.subplots()
	numbins = 100
	ax.hist(all_values, bins=numbins, density=True, alpha=0.6, color='blue', label='Data')

	# Fit Laplace with loc = 0
	loc = 0
	scale = np.mean(np.abs(all_values - loc))  # MLE for b with fixed loc
	x_fit = np.linspace(np.min(all_values), np.max(all_values), 1000)
	y_fit = laplace.pdf(x_fit, loc=loc, scale=scale)
	ax.plot(x_fit, y_fit, 'r', lw=2, label=f'Laplace fit, b={scale:.2f} rad/s')


	ax.set_ylabel('Probability density')
	ax.set_yscale('log')
	# ax.axvline(np.median(all_values), color='red', linewidth=0.2)
	ax.set_xlabel('Angvel (rad/s)')
	# ax.set_xlim([0, 2*np.pi])
	ax.legend()

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + region + '_distribution_' + mode + '_' + groupName + '.pdf', transparent=True)
	fig.savefig(outputDir_png + region + '_distribution_' + mode + '_' + groupName + '.png')
	fig.clf()
	plt.close(fig)


 
def orientation_plot(groupName, region='all', mode='vel', return_data = None, speed_threshold=None, ylim_max=1.5, angvel_threshold=None, invert=False, plot_dir = 'explore', forward_color='pink', backward_color='cyan'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold, invert=invert)
	all_speeds = extract_quantity_from_region(groupName=groupName, region=region, mode='speed', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold, invert=invert)
	all_angvels = extract_quantity_from_region(groupName=groupName, region=region, mode='angvel', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	num_bins = 24
	
	# Create color scheme - accepts any matplotlib color name
	def create_bin_colors(num_bins, forward_color, backward_color):
		colors = []
		gray = '#95A5A6'
		
		for i in range(num_bins):
			angle = (i + 0.5) * 2 * np.pi / num_bins
			angle_deg = np.degrees(angle)
			
			if angle_deg <= 60 or angle_deg >= 300:
				colors.append(forward_color)
			elif 120 <= angle_deg <= 240:
				colors.append(backward_color)
			else:
				colors.append(gray)
		return colors
	
	colors = create_bin_colors(num_bins, forward_color, backward_color)
	
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)
	mid_angles = bins[:-1] + np.pi / num_bins

	# calculate the average speed for each angle bin
	mag = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])
 
	width_factor = 1

	if mode == 'vmag':
		normalize_weighted_angs = mag*hist/np.sum(mag*hist)
		ax.bar(mid_angles, normalize_weighted_angs, width=width_factor*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k', linewidth=0.75)
		ax.set_yticklabels([])
		ax.set_yticks([0,0.1,0.2,0.3])
		ax.set_ylim([0,0.4])
		ax.set_title('v magnitude')
	
	elif mode == 'vdir':
		ax.bar(mid_angles, hist/np.sum(hist), width=width_factor*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k', linewidth=0.75)

		ax.set_yticklabels([])
		ax.set_yticks([0,0.1,0.2,0.3])
		ax.set_ylim([0,0.4])
		ax.set_title('v direction')
  

		dr, dtheta = mean_direction(all_angles)

		# Plot the arrow
		ax.quiver(0, 0, dtheta, dr, angles='xy', scale_units='xy', scale=2, color='black', label="Arrow", width=0.02, zorder = 20)
		
	elif mode == 'vel':
		ax.bar(mid_angles, mag, width=width_factor*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k', linewidth=0.75)
  
		if invert == True:
			ax.set_ylim([0,0.05])
		else:
			ax.set_ylim([0,ylim_max])
			ax.set_yticks(np.linspace(0,ylim_max,6))
		ax.set_title('Mean velocity/speed in each bin (cm/s)')

	elif mode == 'angvel':
		ylim_max = 120
		mag = (180/np.pi)*np.array([np.average(all_angvels[inds==(i+1)]) for i in range(num_bins)])

		ax.bar(mid_angles, mag, width=width_factor*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k', linewidth=0.75)
  
		if invert == True:
			ax.set_ylim([0,0.05])
		else:
			ax.set_ylim([0,ylim_max])
			ax.set_yticks(np.linspace(0,ylim_max,int(ylim_max/30)+1))
		ax.set_title('Mean absolute angvel in each bin (cm/s)')
		
  
	if return_data == True:
		plt.close('all')
		return mid_angles, mag

	elif return_data == 'All':
		plt.close('all')
		return [all_speeds[inds==(i+1)] for i in range(num_bins)]

	message = '_angles_'
	if invert == True:
		message = message + 'under_'

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + region + message + mode + '_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + region + message + mode + '_' +groupName+'.png')
	fig.clf()
	plt.close(fig)
	

def velocity_plot(groupName, region='all', mode='vel', return_data = None, speed_threshold=None, ylim_max=1.5, angvel_threshold=None, invert=False, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold, invert=invert)
	all_speeds = extract_quantity_from_region(groupName=groupName, region=region, mode='speed', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold, invert=invert)


	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	num_bins = 6
	colors = ['cyan','grey','pink','pink','grey','cyan']
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)
	mid_angles = bins[:-1] + np.pi / num_bins

	# calculate the average speed for each angle bin
	mag = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])


 
	# calculate median and quartiles
	# meds = np.array([np.median(all_speeds[inds==(i+1)]) for i in range(num_bins)])
	# qr1 = np.array([np.percentile(all_speeds[inds==(i+1)], 25) for i in range(num_bins)])
	# qr3 = np.array([np.percentile(all_speeds[inds==(i+1)], 75) for i in range(num_bins)])

	if mode == 'vmag':
		normalize_weighted_angs = mag*hist/np.sum(mag*hist)
		ax.bar(mid_angles, normalize_weighted_angs, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
		ax.set_yticklabels([])
		ax.set_yticks([0,0.1,0.2,0.3])
		ax.set_ylim([0,0.4])
		ax.set_title('v magnitude')
	
	elif mode == 'vdir':
		ax.bar(mid_angles, hist/np.sum(hist), width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')

		ax.set_yticklabels([])
		ax.set_yticks([0,0.1,0.2,0.3])
		ax.set_ylim([0,0.4])
		ax.set_title('v direction')
  

		dr, dtheta = mean_direction(all_angles)

		# Plot the arrow
		ax.quiver(0, 0, dtheta, dr, angles='xy', scale_units='xy', scale=2, color='black', label="Arrow", width=0.02, zorder = 20)
		
	elif mode == 'vel':
		ax.bar(mid_angles, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')

		# Plotting median and IQR
		# ax.bar(mid_angles, meds, color='none', width=0.75*(2 * np.pi / num_bins), align="center", edgecolor='r')
		# ax.bar(mid_angles, qr1, color='none', width=0.75*(2 * np.pi / num_bins), align="center", edgecolor='green')
		# ax.bar(mid_angles, qr3, color='none', width=0.75*(2 * np.pi / num_bins), align="center", edgecolor='green')
		# ax.bar(mid_angles, mag, width=0.75*(2 * np.pi / num_bins), align="center", color='none', edgecolor='k', linewidth = 2)
		if invert == True:
			ax.set_ylim([0,0.05])
		else:
			ax.set_ylim([0,ylim_max])
			ax.set_yticks(np.linspace(0,ylim_max,6))
		ax.set_title('Mean velocity/speed in each bin (cm/s)')

	if return_data == True:
		plt.close('all')
		return mid_angles, mag

	elif return_data == 'All':
		plt.close('all')
		return [all_speeds[inds==(i+1)] for i in range(num_bins)]

	message = '_angles_'
	if invert == True:
		message = message + 'under_'

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + region + message + mode + '_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + region + message + mode + '_' +groupName+'.png')
	fig.clf()
	plt.close(fig)


def lr_diff(groupName, region='all', mode='angle', speed_threshold=None, angvel_threshold=None, invert=False, bins_redux = False):
	
	unmixed_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold, invert=invert, unmixed = True)
 
	if mode != 'angle':
		unmixed_quant = extract_quantity_from_region(groupName=groupName, region=region, mode=mode, speed_threshold=speed_threshold, angvel_threshold=angvel_threshold, invert=invert, unmixed = True)

	lr_diffs = []
	for k in range(len(unmixed_angles)):
		angles = unmixed_angles[k]
  
		# Skip if somehow no angles
		if len(angles) == 0:
			print(f'There was a track with nothing in group: {groupName} ! Skipping...')
			continue

		if bins_redux == True:
			angles = angles % (2*np.pi)
			hist, bins = np.histogram(angles, bins=6, range=(0, 2 * np.pi))
		else:
			angles = (angles + np.pi/2) % (2*np.pi) - np.pi/2 # start at 90 deg
			hist, bins = np.histogram(angles, bins=2, range=(-np.pi/2, 1.5 * np.pi))

		if mode != 'angle':
			inds = np.digitize(angles, bins)
			quant = unmixed_quant[k]
			
			if bins_redux == True:
				mag1 = np.average(list(quant[inds==1]) + list(quant[inds==6]))
				mag2 = np.average(list(quant[inds==3]) + list(quant[inds==4]))
				mag = np.array([mag1, mag2])
			else:
				mag = np.array([np.average(quant[inds==(i+1)]) for i in range(2)])

		if mode == 'angle':
			hist = hist/np.sum(hist)
			if bins_redux == True:
				hist = [hist[0] + hist[5], hist[2] + hist[3]]
			else:
				lr_diffs.append(hist)
		else:
			lr_diffs.append(mag)
  
	return lr_diffs

def multiarm_plot(groupName, region='all', mode='vel', mode2='angvel', speed_threshold=None, ylim_max = 1.5, angvel_threshold=None, ylim_max_ang = 180, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all_speeds = extract_quantity_from_region(groupName=groupName, region=region, mode='speed', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)


	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	num_bins = 30
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)

	# calculate the average speed for each angle bin
	mag = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])
	mid_angles = bins[:-1] + np.pi / num_bins

	if mode == 'vel':
		ax.bar(mid_angles, mag, width=(2 * np.pi / num_bins), align="center", edgecolor='k')
		
		ax.set_ylim([0,ylim_max])
		ax.set_yticks(np.linspace(0,ylim_max,6))
		ax.set_title('Mean velocity/speed in each bin (cm/s)')

	elif mode == 'vmag':
		normalize_weighted_angs = mag*hist/np.sum(mag*hist)

		ax.bar(mid_angles, normalize_weighted_angs, width=(2 * np.pi / num_bins), align="center", edgecolor='k')
		
		ax.set_yticklabels([])
		ax.set_yticks([0,0.02,0.04,0.06])
		ax.set_ylim([0,0.08])
		ax.set_title('v magnitude')
	
	if mode == 'vdir':
		ax.bar(mid_angles, hist/np.sum(hist), width=(2 * np.pi / num_bins), align="center", edgecolor='k')
		ax.set_yticklabels([])
		ax.set_yticks([0,0.02,0.04,0.06])
		ax.set_ylim([0,0.08])
		ax.set_title('v direction')
	
	elif mode == 'combined':
		normalize_weighted_angs = mag*hist/np.sum(mag*hist)
		
		if mode2 == 'angvel':
			all_angvels = extract_quantity_from_region(groupName=groupName, region=region, mode='angvel', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
			mag_angvels = (180/np.pi)*np.array([np.average(all_angvels[inds==(i+1)]) for i in range(num_bins)])

			data_color = [x / ylim_max_ang for x in mag_angvels]
			my_cmap = plt.cm.get_cmap('viridis')
			colors = my_cmap(data_color)

			ax.bar(mid_angles, normalize_weighted_angs, width=(2 * np.pi / num_bins), align="center", edgecolor='k', color = colors)

			sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, ylim_max_ang))
			sm.set_array([])  # Dummy array required for ScalarMappable

			# Add color bar with limits
			cbar = plt.colorbar(sm, ax=ax, label='Angular Velocity (deg/s)')

		elif mode2 == 'speed':

			data_color = [x / ylim_max for x in mag]
			my_cmap = plt.cm.get_cmap('viridis')
			colors = my_cmap(data_color)

			ax.bar(mid_angles, normalize_weighted_angs, width=(2 * np.pi / num_bins), align="center", edgecolor='k', color = colors)

			sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, ylim_max))
			sm.set_array([])  # Dummy array required for ScalarMappable

			# Add color bar with limits
			cbar = plt.colorbar(sm, ax=ax, label='Speed (cm/s)')

		ax.set_yticklabels([])
		ax.set_yticks([0,0.02,0.04,0.06])
		ax.set_ylim([0,0.08])
		ax.set_title('v direction')

	fig.suptitle(groupName)
	fig.tight_layout()
	if mode != 'combined':
		fig.savefig(outputDir + region + '_arms_'+ mode + '_'+groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + region + '_arms_' + mode + '_' +groupName+'.png')
	else:
		fig.savefig(outputDir + region + '_arms_'+ mode + '_'+ mode2 + '_' + groupName+'.pdf', transparent=True)
		fig.savefig(outputDir_png + region + '_arms_' + mode + '_' + mode2 + '_' +groupName+'.png')
	fig.clf()
	plt.close('all')

def velocity_distribution_plot(groupName, region='all', speed_threshold=0, ylim_max = 3, angvel_threshold=None, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all_speeds = extract_quantity_from_region(groupName=groupName, region=region, mode='speed', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)

	num_bins = 6
	colors = ['cyan','grey','pink','pink','grey','cyan']
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)

	# calculate the average speed for each angle bin
	mag = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])
	range_bins = np.linspace(0, 5, 25+1)

	fig, ax = plt.subplots(6,1, figsize=(6, 18))
	for i  in range(num_bins):
		ax[i].hist(all_speeds[inds==(i+1)], bins=range_bins, color=colors[i], density=True)
		ax[i].set_xlim([0, 3])
		ax[i].set_ylim([0, ylim_max])
		ax[i].set_xlabel('Speed (cm/s)')
		ax[i].set_ylabel('Probability density')
		ax[i].axvline(mag[i], color='red')
		ax[i].set_title(f'[{bins[i]:.2f},{bins[i+1]:.2f}]')
		

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(f'{outputDir}velocity_distribution_thresh={np.round(speed_threshold,2)}_{groupName}.pdf', transparent=True)
	fig.savefig(f'{outputDir_png}velocity_distribution_thresh={np.round(speed_threshold,2)}_{groupName}.png')
	fig.clf()
	plt.close(fig)

def angvels_plot(groupName, region='all', return_data = None, speed_threshold=None, angvel_threshold=None, ylim_max_ang=180, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_angvels = extract_quantity_from_region(groupName=groupName, region=region, mode='angvel', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	# num_bins = 2
	# colors = ['cyan','pink']
	# all_travel_angles = all_travel_angles + (all_travel_angles>np.pi/2)*(-2*np.pi)
	# hist, bins = np.histogram(all_travel_angles, bins=num_bins, range=(-1.5*np.pi, np.pi/2))
	# inds = np.digitize(all_travel_angles, bins)

	num_bins = 6
	colors = ['green','grey','brown','brown','grey','green']
	hist, bins = np.histogram(all_angles, bins=num_bins, range=(0, 2 * np.pi))
	inds = np.digitize(all_angles, bins)


	mag = (180/np.pi)*np.array([np.average(all_angvels[inds==(i+1)]) for i in range(num_bins)])

	# ax.bar(bins[:-1], normalize_weighted_angs, width=(2 * np.pi / num_bins), align="edge", color=colors, edgecolor='k')
	mid_angles = bins[:-1] + np.pi / num_bins
	ax.bar(mid_angles, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
	# ax.set_yticklabels([])
	# ax.set_yticks(np.arange(10)/5)
	ax.set_ylim([0,ylim_max_ang])
	ax.set_yticks(np.linspace(0,ylim_max_ang,7))
	ax.set_title('Mean absolute angular speed in each bin (deg/s)')

	if return_data == True:
		plt.close('all')
		return mid_angles, mag
	elif return_data == 'All':
		plt.close('all')
		return [(180/np.pi)*all_angvels[inds==(i+1)] for i in range(num_bins)]


	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + 'angVelsPlot_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'angVelsPlot_'+groupName+'.png')
	fig.clf()
	plt.close(fig)

def combined_plot(groupName, region='all', speed_threshold=None, ylim_max = 1.5, angvel_threshold=None, ylim_max_ang = 180, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all_angles = extract_quantity_from_region(groupName=groupName, region=region, mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all_speeds = extract_quantity_from_region(groupName=groupName, region=region, mode='speed', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all_angvels = extract_quantity_from_region(groupName=groupName, region=region, mode='angvel', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)

	fig, ax = plt.subplots()

	num_bins = 16
	half = int(num_bins/2)
	all_angles = (all_angles + np.pi/2) % (2*np.pi) - np.pi/2 # start at 90 deg

	hist, bins = np.histogram(all_angles, bins=num_bins, range=(-np.pi/2, 3*np.pi/2))
	inds = np.digitize(all_angles, bins)

	mid_angles = (180/np.pi)*(bins[1:] + bins[:-1])/2
	
	# calculate the average speed for each angle bin
	mag_speeds = np.array([np.average(all_speeds[inds==(i+1)]) for i in range(num_bins)])
	mag_angvels = (180/np.pi)*np.array([np.average(all_angvels[inds==(i+1)]) for i in range(num_bins)])

	data_color = [x / ylim_max_ang for x in mag_angvels]
	my_cmap = plt.cm.get_cmap('viridis')
	# colors = my_cmap(data_color)

	mult = 0.9

	ax.barh(mid_angles[:half], mag_speeds[:half], height = mult*(180/np.pi)*(2*np.pi/num_bins), color = my_cmap(data_color[:half]))
	ax.barh(mid_angles[:half], -mag_speeds[half:num_bins], height = mult*(180/np.pi)*(2*np.pi/num_bins), color = my_cmap(data_color[half:]))

	ax.axvline(0)
	ax.axhline(-45)
	ax.axhline(45)

	sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, ylim_max_ang))
	sm.set_array([])  # Dummy array required for ScalarMappable

	# Add color bar with limits
	cbar = plt.colorbar(sm, ax=ax, label='Angular Velocity (deg/s)')

	ax.set_xlim([-ylim_max, ylim_max])
	ax.set_yticks((180/np.pi)*bins[:half+1])
	ax.set_xlabel('Speed (cm/s)')
	ax.set_ylabel('Heading angle (deg)')


	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + 'combinedPlot_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'combinedPlot_'+groupName+'.png')

def triple_region_plot(groupName, mode='vel', speed_threshold=None, ylim_max = 1.5, angvel_threshold=None, ylim_max_ang = 180, plot_dir = 'explore'):
	
	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	fig, ax = plt.subplots(1,3, subplot_kw={'projection': 'polar'}, figsize=(12, 4))

	if mode == 'vel':
		num_bins = 6
		colors = ['cyan','grey','pink','pink','grey','cyan']
		for i in range(3):
			mid_angles, mag = velocity_plot(groupName, region=i, mode='vel', return_data=True, speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
			ax[i].bar(mid_angles, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
			ax[i].set_ylim([0,ylim_max])
			ax[i].set_yticks(np.linspace(0,ylim_max,6))
		fig.suptitle('Mean velocity/speed in each bin (cm/s)')

	elif mode == 'angvel':
		# num_bins = 2
		# colors = ['cyan','pink']
		num_bins = 6
		colors = ['green','grey','brown','brown','grey','green']
		for i in range(3):
			mid_angles, mag = angvels_plot(groupName, region=i, return_data=True, speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
			ax[i].bar(mid_angles, mag, width=0.75*(2 * np.pi / num_bins), align="center", color=colors, edgecolor='k')
			ax[i].set_ylim([0,ylim_max_ang])
			ax[i].set_yticks(np.linspace(0,ylim_max_ang,7))
		fig.suptitle('Mean absolute angular speed in each bin (deg/s)')

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + f'triple_region_{mode}_{groupName}.pdf',transparent=True)
	fig.savefig(outputDir_png + f'triple_region_{mode}_{groupName}.png')
	fig.clf()
	plt.close('all')

def joint_distribution_plot(groupName, mode1 = 'angvel', mode2='speed', speed_threshold=None, angvel_threshold=None, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all1 = extract_quantity_from_region(groupName=groupName, region='all', mode=mode1, speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all2 = extract_quantity_from_region(groupName=groupName, region='all', mode=mode2, speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	
	message_dict = {'angvel': 'angular velocity rad/s', 'speed': 'speed cm/s', 'head': 'angle (rad)', 'location': 'xlocation (cm)'}

	if mode1 == 'location':
		n_bins = 14
		bin_edges = np.linspace(start=min(all1), stop=max(all1), num=n_bins + 1)
		bins = np.digitize(all1, bins=bin_edges)
		# Exclude the last index
		valid_indices = bins <= len(bin_edges) - 1
		all1 = all1[valid_indices]
		all2 = all2[valid_indices]
		bins = bins[valid_indices]

		data = pd.DataFrame({'x': all1, 'y': all2, 'bin': bins})
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
		bin_stats = data.groupby('bin')['y'].agg(['mean', 'std'])
		bin_stats = data.groupby('bin').agg(mean=('y', 'mean'), std=('y', 'std'), count=('y', 'count'))
		bin_stats['sem'] = bin_stats['std'] / np.sqrt(bin_stats['count'])
		bin_stats['bin_center'] = bin_centers
		
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.bar(bin_stats['bin_center'], bin_stats['mean'], yerr=bin_stats['sem'], align='center', width = bin_edges[1] - bin_edges[0], alpha=0.7, ecolor='black', capsize=10)
		ax.set_xlabel('Location')
		ax.set_ylabel(f'Average {mode2} value')
		# ax.grid(True)
		
	else:
		jp = sns.jointplot(x=all1, y=all2, kind="hist")
		jp.set_axis_labels(message_dict[mode1], message_dict[mode2])
		jp.figure.tight_layout() 
		fig = jp.figure

	# fig, ax = plt.subplots()
	# sns.histplot(x=all1, y=all2, bins=20, cbar=True, ax=ax)
	# ax.set_xlabel(message_dict[mode1])
	# ax.set_ylabel(message_dict[mode2])

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + f'jointPlot_{mode1}_{mode2}_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + f'jointPlot_{mode1}_{mode2}_'+groupName+'.png')
	fig.clf()
	plt.close(fig)

def head_travel_correlation(groupName, speed_threshold = 0.25, angvel_threshold=None, plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	all1 = extract_quantity_from_region(groupName=groupName, region='all', mode='head', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)
	all2 = extract_quantity_from_region(groupName=groupName, region='all', mode='travel', speed_threshold=speed_threshold, angvel_threshold=angvel_threshold)

	diff = np.abs(all1 - all2)%(2*np.pi)
	diff = diff + (diff>=np.pi)*(2*np.pi - 2*diff)

	fig, ax = plt.subplots()
	# sns.regplot(x=all1, y=all2, scatter_kws={'s':2}, ax=ax)
	# sns.histplot(x=all1, y=all2, bins=20, cbar=True, ax=ax)
	sns.histplot(x=diff, bins=20, cbar=True, ax=ax)
	ax.set_xlabel('absolute difference btwn head & travel')
	
	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir + 'head_travel_correlation_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'head_travel_correlation_'+groupName+'.png')
	fig.clf()
	plt.close(fig)

def peaks_distribution(groupName, mode = 'speed', plot_dir = 'explore'):

	outputDir = 'plots/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir)
	outputDir_png = 'plots_png/' + groupName + f'/{plot_dir}/'
	create_directory(outputDir_png)

	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	
	all_scalars= []

	for file in dirs:
		if 'output' not in file.split('/')[-1]:
			continue

		f1 = open(inputDir + file, 'rb')
		pos, angles, vels, settings, originalTrackingInfo, _ = pickle.load(f1)
		f1.close()

		speed = np.sqrt((vels[:,0])**2 + (vels[:,1])**2)
		angvels = np.abs(vels[:, 2])

		if mode == 'speed':
			x = np.copy(speed)
		elif mode == 'angvel':
			x = np.copy(angvels)

		# only keep turns not touching wall
		bl = bl_default	# border length
		stageW, stageH = settings['stageW'], settings['stageH']
		inds = (pos[:,1]>bl) & (pos[:,1]<(stageH-bl)) & (pos[:,0]>bl) & (pos[:,0]<(stageW-bl)) & (speed>=0) & (speed<5)
		actual_inds = np.where(inds)[0]

		# Limit to peaks that are not too close to border

		peaks, _ = find_peaks(x, height=0)
		peaks = np.array([p for p in peaks if p in actual_inds])
		if len(peaks)==0:
			continue
		all_scalars.append(x[peaks])

	all_scalars = np.concatenate(all_scalars)
	if mode == 'angvel':
		all_scalars = (180/np.pi)*all_scalars

	fig, ax = plt.subplots()
	numbins = 1000
	ax.hist(all_scalars, bins=numbins, density=True)
	ax.axvline(np.median(all_scalars), color='red', linewidth=0.2)
	ax.set_yscale('log')

	ax.set_ylabel('Probability density')
	if mode == 'speed':
		ax.set_xlabel('Speed (cm/s)')
		ax.set_xlim([0, 5])
		# ax.set_ylim([0, ylim_max])
	elif mode == 'angvel':
		ax.set_xlabel('Angvel (deg/s)')
		ax.set_xlim([0, 540])

	fig.suptitle(groupName)
	fig.tight_layout()
	fig.savefig(outputDir  + 'peaks_distribution_'+ mode + '_'+groupName+'.pdf', transparent=True)
	fig.savefig(outputDir_png + 'peaks_distribution_' + mode + '_' +groupName+'.png')
	fig.clf()
	plt.close(fig)

	