from functions_turning import *


# groupNames=['WT', 'EPG+', 'EPG_Kir', 'flat-25', 'DNB05_Kir', 'DNB05+','Ablated']

# groupNames = ['WT', 'Ablated', 'flat-25',  'EPG+', 'Kir-+', 'AC-Kir']

# groupNames =['WT', 'EPG+', 'EPG_Kir', 'flat-25', 'Ablated', 'L-ablated', 'R-ablated', 'AC-Kir', 'HC-Kir', 'AC-+', 'HC-+', 'Kir-+', 'SS00096-+', 'SS00096-Kir']

groupNames = ['WT', 'WT2'] #+ ['SS00096-Kir', 'SS00096-+']



for groupName in groupNames:
	# distribution_in_out(groupName, ht=np.pi/3)
	# num_turns_direction(groupName, ht=np.pi/3)
	# joyplot_in_out(groupName, ht=np.pi/3)
	# polar_turns(groupName, ht=np.pi/3)

	# turn_distribution(groupName, ht=np.pi/3, mode='angle')
	# turn_distribution(groupName, ht=np.pi/3, mode='time')
	# turn_distribution(groupName, ht=np.pi/3, mode='temp')
	
	# distribution_in_out(groupName, ht=np.pi/2, pm=3, speed_threshold=(0.25, 3), angle_threshold=30)
	# num_turns_direction(groupName, ht=np.pi/2, pm=3, speed_threshold=(0.25, 3), angle_threshold=30)
	# joyplot_in_out(groupName, ht=np.pi/2, pm=3, speed_threshold=(0.25, 3), angle_threshold=30)
	# polar_turns(groupName, ht=np.pi/2, pm=3, speed_threshold=(0.25, 3), angle_threshold=30)

	# distribution_in_out(groupName, ht=np.pi/3, speed_threshold=0.25)
	# num_turns_direction(groupName, ht=np.pi/3, speed_threshold=0.25)
	# joyplot_in_out(groupName, ht=np.pi/3, speed_threshold=0.25)

	# polar_turns(groupName, ht=np.pi/3, speed_threshold=0.25)
	# polar_turns(groupName, ht=np.pi/3, speed_threshold=0.25, mode='stratify')

	# turn_distribution(groupName, ht=np.pi/3, speed_threshold=0.25, mode='angle')
	# turn_distribution(groupName, ht=np.pi/3, speed_threshold=0.25, mode='time')
	# turn_distribution(groupName, ht=np.pi/3, speed_threshold=0.25, mode='temp')
	# turn_distribution(groupName, ht=np.pi/3, speed_threshold=0.25, mode='temp_restrict')
	# turn_distribution(groupName, ht=np.pi/3, speed_threshold=0.25, mode='curve')

	# before_turns(groupName, ht=np.pi/3, speed_threshold=0.25)
	peaks_verification(groupName, speed_threshold=0.25)

	