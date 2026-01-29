from functions_tracking import *
from functions_plotting import *
from functions_plotting2 import *
from functions_plotting_all import *
from functions_display import *

import matplotlib
import matplotlib.colors as colors
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = "Arial"

def run_all_plots(groupName):
	inputDir = 'outputs/outputs_' + groupName + '/'
	dirs = os.listdir(inputDir)
	dirs = [file for file in dirs if (file.split('.')[-1] == 'output')]

	for file in tqdm(dirs):
		if 'output' not in file.split('/')[-1]:
			continue
		
		# plot_final(inputDir+file, groupName, ht = 2*np.pi/3, speed_threshold=0.25)
		# plot_track(inputDir+file, groupName, ht = 2*np.pi/3, speed_threshold=0.25)
		# plot_track_segmented(inputDir+file, groupName, ht = 2*np.pi/3, speed_threshold=0.25)
		# plot_scalar(inputDir+file, groupName, limit=60, ht = 2*np.pi/3)

	fly_progression_plot(groupName)
	distance_reached_plot(groupName, plot_dir='important')
	distance_reached_plot(groupName, mode = 'ten', plot_dir='important')
	efficiency_plot(groupName,plot_dir='explore' )
	heading_index(groupName, speed_threshold = 0.25, plot_dir = 'important', mode='orientation')
	# value_plot(groupName)
	# value_plot(groupName, mode='angvel', speed_threshold=0.25)
	# value_plot(groupName, weighted=True)
	# value_plot(groupName, mode='angvel', weighted=True, speed_threshold=0.25)
	# peaks_distribution(groupName)
	# peaks_distribution(groupName, mode='angvel')

	# velocity_plot(groupName, mode='vel', speed_threshold=0.25, invert=True)
	# velocity_plot(groupName, mode='vdir', speed_threshold=0.25, invert=True)
	
	# angvels_plot(groupName, speed_threshold=0.25, plot_dir='important')
	orientation_plot(groupName, mode='vel', speed_threshold=0.25, plot_dir='important')
	orientation_plot(groupName, mode='angvel', speed_threshold=0.25, plot_dir='important', forward_color='limegreen', backward_color='peru')


	# velocity_plot(groupName, mode='vel', speed_threshold=0.25, plot_dir='important')
	# velocity_plot(groupName, mode='vdir', speed_threshold=0.25, plot_dir='important')
	multiarm_plot(groupName, mode='vel', speed_threshold=0.25, plot_dir='important')
	multiarm_plot(groupName, mode='vdir', speed_threshold=0.25, plot_dir='important')

	angvel_dist_plot(groupName, speed_threshold=0.25)

	# multiarm_plot(groupName, mode='combined', mode2='angvel', speed_threshold=0.25)
	# multiarm_plot(groupName, mode='combined', mode2='speed',speed_threshold=0.25)
	# velocity_distribution_plot(groupName, speed_threshold=0, ylim_max=6)
	# velocity_distribution_plot(groupName, speed_threshold=0.25, ylim_max=3)
	# combined_plot(groupName, region='all', angvel_threshold=None, speed_threshold=0.25)
	# head_travel_correlation(groupName)
		
	# triple_region_plot(groupName, mode='vel', speed_threshold=0.25)
	# triple_region_plot(groupName, mode='angvel', speed_threshold=0.25)
	# avp_statistics(groupName, speed_threshold=0.25)
 
	# stop_statistics(groupName, speed_threshold = 0.25, mode = 'time')
	# stop_statistics(groupName, speed_threshold = 0.25, mode = 'joint')
	# stop_statistics(groupName, speed_threshold = 0.25, mode = 'location')

	# sample_group(groupName)

	# border_touch(groupName)
	# joint_distribution_plot(groupName, speed_threshold = 0.25, mode1 = 'location', mode2 = 'speed')
	# joint_distribution_plot(groupName, speed_threshold = 0.25, mode1 = 'location', mode2 = 'angvel')
	# joint_distribution_plot(groupName, mode1 = 'head2')
	# scalar_plot(groupName, region='all', mode='speed')
	# scalar_plot(groupName, region='all', mode='angvel')
	# scalar_plot(groupName, region='all', mode='time')
	# regression_tracks(groupName)

	# if 'mm' in groupName:
	# 	all_angles_plot(groupName, region = 'between')
 
	# Turning stuff

	num_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important', mode='ash')
 
	# num_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important')
	# distribution_in_out(groupName, ht=2*np.pi/3, speed_threshold=0.25)
	# joyplot_polar(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important')

	polar_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important', nbins = 4)
	# # polar_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='stratify')
	# polar_ash(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important')

	turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='angle', plot_dir='important')
	# turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='turn_size_binned', plot_dir='important')
	turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='turn_ash', plot_dir='important')
 
	# # turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='time')
	# # turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='temp')
	# # turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='temp_restrict')
	# # turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='curve')
	# # turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='joint')

	# before_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25)
	# peaks_verification(groupName, speed_threshold=0.25)
	acf(groupName)



# groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'L-ablated', 'R-ablated', 'AC-Kir', 'AC-+'] + ['HC-Kir', 'HC-+', 'GR28-TRPA1', 'Gr28-exc66'] + ['R4-+', 'R31A12-+'] + ['R4-Kir', 'Gr28-exc8'] + ['R31A12-Kir']

# groupNames = ['SS98+', 'SS98Kir', 'SS90+', 'SS90Kir', 'SS00096-+', 'SS00096-Kir', 'SS408+', 'SS408-Kir']

# groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+'] +  ['SS408-Kir', 'SS408+', 'SS98+', 'SS98Kir'] + ['SS90+', 'SS90Kir'] + ['SS00096-+', 'SS00096-Kir'] + ['L-ablated', 'R-ablated', 'AC-Kir', 'AC-+'] + ['HC-Kir', 'HC-+', 'GR28-TRPA1', 'Gr28-exc66'] + ['R4-+', 'R31A12-+'] + ['R4-Kir', 'Gr28-exc8'] + ['R31A12-Kir']


# groupNames = ['SS408+', 'SS408-Kir', 'SS98+', 'SS98Kir']+['SS90+', 'SS90Kir']+['SS00096-+', 'SS00096-Kir', 'HdC+', 'HdC_Kir'] 

# groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+']
# groupNames = ['61933+', '61933_kir', 'DNB05+', 'DNB05_Kir','HdB+', 'HdB_Kir','HdC+', 'HdC_Kir']


# groupNames = ['WT', 'Flat25', 'Ablated']

# groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated']
# groupNames = ['WT']
# groupNames = ['WT', 'Ablated']

# groupNames = ['WT', 'Kir-+','HdC+', 'HdC_Kir', 'Ablated', 'Flat25', 'R-ablated', 'L-ablated']

groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'AC-+', 'AC-Kir', 'Gr28-exc8', 'Gr28-exc66', 'GR28-TRPA1', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated']

groupNames = groupNames+['WT', 'Ablated', 'Flat25', 'Kir-+', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated']

groupNames =  groupNames+['WT', 'Ablated', 'Kir-+', 'SS90+', 'SS90Kir', 'SS00096-+', 'SS00096-Kir', 'SS98+', 'SS98Kir', 'SS131+', 'SS131Kir','SS408+', 'SS408-Kir', 'HdB_Kir','HdC+', 'HdC_Kir']

groupNames = list(set(groupNames))

groupNames = ['72923x601898_FL50', '72923x601898_Kir', '72923X69470_FL50', '72923X69470_Kir'] + ['w1118-sweden'] + ['Kir-+']

# groupNames = ['w1118-sweden']


# all_heading_indices(groupNames)

# groupNames = ['PFL3Kir', 'FR', 'ZI', 'AblatedSteph', 'FC2Kir', 'FC2+']
# groupNames = ['WT']

groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated', 'AC-+', 'AC-Kir', 'Gr28-exc8', 'Gr28-exc66']

groupNames = ['FB5AB_87833_Kir', 'FB5AB_87833_FL50', 'FB5AB_86833_Kir', 'FB5AB_86833_FL50']

groupNames = ['72923x601898_FL50', '72923x601898_Kir', '72923X69470_FL50', '72923X69470_Kir'] + ['w1118-sweden'] + ['Kir-+']

groupNames = ['WT']

for groupName in groupNames:
	run_all_plots(groupName)


# groupNames = ['WT', 'Ablated', 'SS90+', 'SS00096-+', 'SS98+','SS131+', 'SS408+', 'SS90Kir', 'SS00096-Kir', 'SS98Kir', 'SS131Kir', 'SS408-Kir', 'Kir-+']

groupNames = ['WT', 'Ablated', 'Flat25', 'SS90+', 'SS90Kir', 'SS00096-+', 'SS00096-Kir', 'SS98+', 'SS98Kir', 'SS131+', 'SS131Kir','SS408+', 'SS408-Kir','Kir-+']

groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'AC-+', 'AC-Kir', 'Gr28-exc8', 'Gr28-exc66', 'GR28-TRPA1', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated']

groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated']

groupNames =  ['WT', 'Ablated', 'Kir-+', 'SS90+', 'SS90Kir', 'SS00096-+', 'SS00096-Kir', 'SS98+', 'SS98Kir', 'SS131+', 'SS131Kir','SS408+', 'SS408-Kir', 'HdB_Kir','HdC+', 'HdC_Kir']

groupNames =  ['WT', 'Ablated', 'Flat25', 'Kir-+', 'SS131+', 'SS131Kir','HdC+', 'HdC_Kir']

# 'SS408+', 'SS408-Kir', 'HdB_Kir', 'SS00096-+', 'SS00096-Kir', 'SS90+', 'SS90Kir', 'SS98+', 'SS98Kir'

# groupNames = ['WT', 'Ablated']

# all_turn_distribution_ash(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25)

# # all_up_down(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25, mode='speed')
# # # all_up_down(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25, mode='turns')
# # all_up_down(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25, mode='angvel')
# all_num_turns(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25)

# all_efficiencies(groupNames)
# all_polar_ratios(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25)
# all_lr_diffs(groupNames, speed_threshold = 0.25)
# all_lr_diffs(groupNames, speed_threshold = 0.25, mode = 'speed')
# all_lr_diffs(groupNames, speed_threshold = 0.25, mode = 'angvel')

# all_reach_plot(groupNames, mode='distance', tenMin = True)
# all_reach_plot(groupNames, mode='time', tenMin = True)
# all_reach_plot(groupNames, mode='distance')
# all_reach_plot(groupNames, mode='time')
# all_stop_distribution(groupNames, mode = 'time')
# all_stop_distribution(groupNames, mode = 'location')
# all_turn_distribution(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25)
# all_turn_boxplots(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25, mode='average')
# all_turn_boxplots(groupNames, ht = 2*np.pi/3, speed_threshold = 0.25, mode='turn')


# predict_taxis(groupNames)
# stop_threshold_stats(groupNames)