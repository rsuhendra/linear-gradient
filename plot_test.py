from functions_tracking import *
from functions_plotting import *
from functions_plotting2 import *


if __name__ == '__main__':
	groupName = 'WT' #HdC_Kir #FL50_7mm
	outputDir = 'outputs/outputs_'+groupName+'/'
	d0 = 'videos/'+groupName+'/'
	dirs = os.listdir(d0)
	create_directory(outputDir)

	file = dirs[0] # 'gradient_10-22-2020_17-09.mp4'
	# file = 'shallow_10-17-2024_13-19.mp4'

	pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2) = track_video(fin = d0+file, mode = 'testing')
	
	f1 = open(outputDir+file.split('.')[0]+'.output','wb')
	pickle.dump((pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)), f1)
	f1.close()

	inputDir = 'outputs/outputs_' + groupName + '/'
	plot_track_segmented(inputDir+file.split('.')[0]+'.output', groupName, ht = 2*np.pi/3, speed_threshold=0.25)
	# plot_scalar(inputDir+file.split('.')[0]+'.output', groupName, limit=60)


