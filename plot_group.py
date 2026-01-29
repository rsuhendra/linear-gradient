from functions_tracking import *
from functions_plotting import *
from functions_plotting2 import *

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
		plot_track(inputDir+file, groupName, ht = 2*np.pi/3, speed_threshold=0.25)
		# plot_track_segmented(inputDir+file, groupName, ht = 2*np.pi/3, speed_threshold=0.25)
		# plot_scalar(inputDir+file, groupName, limit=60, ht = 2*np.pi/3)

	fly_progression_plot(groupName)
	distance_reached_plot(groupName, plot_dir='important')
	distance_reached_plot(groupName, mode = 'ten', plot_dir='important')
	heading_index(groupName, speed_threshold = 0.25, plot_dir = 'important', mode='orientation')

	orientation_plot(groupName, mode='vel', speed_threshold=0.25, plot_dir='important')
	orientation_plot(groupName, mode='angvel', speed_threshold=0.25, plot_dir='important', forward_color='limegreen', backward_color='peru')

	angvel_dist_plot(groupName, speed_threshold=0.25)

	# Turning stuff

	num_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important', mode='ash')
 
	polar_turns(groupName, ht=2*np.pi/3, speed_threshold=0.25, plot_dir='important', nbins = 4)

	turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='angle', plot_dir='important')

	turn_distribution(groupName, ht=2*np.pi/3, speed_threshold=0.25, mode='turn_ash', plot_dir='important')

	acf(groupName)





groupNames = ['WT', 'Ablated', 'Flat25', 'Kir-+', 'HC-+', 'HC-Kir', 'R-ablated', 'L-ablated', 'AC-+', 'AC-Kir', 'Gr28-exc8', 'Gr28-exc66']

groupNames = ['FB5AB_87833_Kir', 'FB5AB_87833_FL50', 'FB5AB_86833_Kir', 'FB5AB_86833_FL50']

groupNames = ['72923x601898_FL50', '72923x601898_Kir', '72923X69470_FL50', '72923X69470_Kir'] + ['w1118-sweden'] + ['Kir-+']

groupNames = ['WT']

for groupName in groupNames:
	run_all_plots(groupName)
