# ===== IMPORTS =====

import numpy as np
import datetime


# ===== THRESHOLDING AND SENSITIVITY PARAMETERS =====

def best_settings(groupName, settings):
	# Determine optimal binarization thresholds for different fly genotypes
	# Different genotypes have different visual contrast with background, so thresholds must be adjusted
	# Parameters:
	#   groupName: fly genotype/experimental group name
	#   settings: tracking settings dictionary (used to access date for historical genotypes)
	# Returns: (threshold, percentile_threshold)
	#   threshold: primary binarization threshold (intensity value)
	#   percentile_threshold: percentile for background estimation (0-100)
	
	# ===== GENOTYPE GROUPINGS =====
	# Group flies by similar thresholding requirements
	
	# Group A: Standard/control genotypes - moderate contrast
	group1 = ['WT', 'Flat25', 'SS408+', 'SS408-Kir', 'SS90+', 'SS90Kir', 
	          'SS98+', 'SS98Kir', 'SS131+', 'SS131Kir']
	group2 = ['Kir-+', 'L-ablated', 'R-ablated', 'AC-Kir', 'AC-+', 'Gr28-exc8',
	          'R4-+', 'R4-Kir', 'R31A12-Kir', 'R31A12-+']
	group3 = ['HC-Kir', 'HC-+', 'GR28-TRPA1', 'Gr28-exc66', 'HdBKir2', 
	          'AblatedSteph', 'ZI', 'FR']
	
	# Group B: Other neural circuit genotypes - higher percentile needed
	group4 = ['SS00096-+', 'SS00096-Kir']
	group5 = ['PFNd-+', 'PFNd-Kir', 'PFNv-+', 'PFNv-Kir']
	
	# Group C: Different laboratory stock - requires higher sensitivity
	groupC = ['61933+', '61933_kir', 'DNB05+', 'DNB05_Kir', 'HdB+', 'HdB_Kir', 'HdC+', 'HdC_Kir']
	
	# Group D: Other stocks and recent additions
	group6 = ['FC2+', 'FC2Kir', 'PFL3Kir']
	group7 = ['72923x601898_FL50', '72923x601898_Kir', '72923X69470_FL50', '72923X69470_Kir', 
	          'w1118-sweden', 'FB5AB_87833_Kir', 'FB5AB_87833_FL50', 'FB5AB_86833_Kir', 'FB5AB_86833_FL50']
	
	# Combine groups for organizational purposes
	groupA = group1 + group2 + group3
	groupB = group4 + group5
	groupD = group6 + group7  # All remaining genotypes use the same thresholds
	
	# ===== ASSIGN THRESHOLDS BY GENOTYPE GROUP =====
	
	if groupName in groupA:
		# Standard genotypes: moderate threshold, 30th percentile background
		thresh = 40
		perc = 30
	elif groupName in groupB:
		# Group B genotypes: same threshold, slightly higher percentile
		thresh = 40
		perc = 35
	elif groupName in groupC:
		# Group C: lower threshold (more sensitive), higher percentile for background
		thresh = 35
		perc = 40
	elif groupName in groupD:
		# Group D: high threshold (less sensitive), low percentile
		thresh = 50
		perc = 30
	elif groupName == 'Ablated':
		# Ablated flies: threshold changed over time (lighting conditions changed)
		date1 = datetime.datetime.strptime("10-28-2020", "%m-%d-%Y")
		if settings['date'] < date1:
			# Earlier recordings: lower contrast
			thresh = 20
			perc = 35
		else:
			# Later recordings: improved lighting
			thresh = 40
			perc = 35

	return thresh, perc


# ===== LIGHTING MASK FUNCTIONS =====

def mask_lighting(dframed, adjustStartFlag, settings):
	# Remove artifacts from flickering lights and LED indicators in specific arena regions
	# Different regions and genotypes have different lighting artifacts that must be masked out
	# Parameters:
	#   dframed: difference frame (absolute difference from background)
	#   adjustStartFlag: whether arena is still settling at start of video
	#   settings: tracking settings including file name, date, genotype
	# Returns: masked difference frame with artifact regions set to zero
	
	dframe = np.copy(dframed)

	# ===== EXTRACT METADATA =====
	fname = settings['file']
	date = datetime.datetime.strptime(fname.split('_')[-2], "%m-%d-%Y")

	# Define key dates when experimental setup changed
	date1 = datetime.datetime.strptime("10-20-2020", "%m-%d-%Y")
	date2 = datetime.datetime.strptime("12-04-2020", "%m-%d-%Y")
	date3 = datetime.datetime.strptime("02-25-2021", "%m-%d-%Y")
	date4 = datetime.datetime.strptime("04-03-2021", "%m-%d-%Y")
	date5 = datetime.datetime.strptime("05-25-2021", "%m-%d-%Y")
	date6 = datetime.datetime.strptime("08-25-2023", "%m-%d-%Y")

	# ===== TEMPERATURE GRADIENT SETUP MASKS (date-dependent) =====
	# These masks remove LED indicators and flickering lights from the thermal gradient apparatus
	
	if date < date3:
		# Early gradient setup (before Feb 25, 2021): extensive LED indicators on sides
		dframe[0:20, 0:150] = 0    # Upper left corner
		dframe[20:40, 0:40] = 0    # Upper left corner
		dframe[40:100, 0:20] = 0   # Left edge
		dframe[-20:, 0:400] = 0    # Bottom left edge
		dframe[0:20, -200:] = 0    # Upper right corner
		dframe[20:60, -40:] = 0    # Right edge
		dframe[60:200, -20:] = 0   # Right edge

		# Bottom-left LED position changed over time
		if date1 <= date < date2:
			dframe[500:590, 0:25] = 0    # Bottom left LED position (Oct-Nov 2020)
		elif date == date2:
			dframe[500:625, 0:25] = 0    # Bottom left LED position (Dec 4, 2020)
		elif date2 < date < date3:
			dframe[525:675, 0:40] = 0    # Bottom left LED position (Dec 2020 - Feb 2021)
	
	elif date3 <= date < date4:
		# Mid gradient setup (Feb 25 - Apr 3, 2021): modified LED positions
		dframe[0:20, 0:150] = 0     # Upper left corner
		dframe[20:100, 0:120] = 0   # Left side
		dframe[100:130, 0:20] = 0   # Left edge
		dframe[-20:, 0:400] = 0     # Bottom left edge
		dframe[0:20, -200:] = 0     # Upper right corner
		dframe[20:60, -60:] = 0     # Right side
		dframe[60:200, -20:] = 0    # Right edge
		dframe[525:700, 0:40] = 0   # Bottom left LED position

	elif date4 <= date < date5:
		# Updated gradient setup (Apr 3 - May 25, 2021): more extensive left side masking
		dframe[0:20, 0:200] = 0     # Upper left corner
		dframe[20:170, 0:140] = 0   # Large left side region
		dframe[100:130, 0:20] = 0   # Left edge
		dframe[-20:, 0:400] = 0     # Bottom left edge
		dframe[0:20, -200:] = 0     # Upper right corner
		dframe[20:100, -120:] = 0   # Right side
		dframe[100:200, -20:] = 0   # Right edge
		dframe[525:650, 0:40] = 0   # Bottom left LED position

	elif date5 < date < date6:
		# Later gradient setup (May 25 - Aug 25, 2023): minimalist masking
		dframe[0:50, 0:50] = 0      # Upper left corner
		dframe[250:, 0:30] = 0      # Bottom left edge
		dframe[0:50, -50:] = 0      # Upper right corner
	
	elif date >= date6:
		# Recent gradient setup (Aug 25, 2023+): minimal or no LED masking needed
		pass

	# ===== SHALLOW ARENA SETUP MASKS (genotype-dependent) =====
	# Different fly lines have different visual characteristics requiring specific masking
	
	if settings['groupName'] in ['61933+', '61933_kir', 'DNB05+', 'DNB05_Kir', 
	                              'HdB+', 'HdB_Kir', 'HdC+', 'HdC_Kir']:
		# These genotypes have higher optical density, need more aggressive LED masking
		dframe[0:100, 0:30] = 0     # Upper left corner (large region)
		dframe[0:15, 0:200] = 0     # Upper left edge (wide)
		dframe[310:, 0:30] = 0      # Bottom left corner
		dframe[280:310, 0:100] = 0  # Bottom left side
		dframe[0:100, -50:] = 0     # Upper right corner (large region)
		dframe[0:20, 500:] = 0      # Upper right side
		dframe[310:, -30:] = 0      # Bottom right corner

	# ===== INITIALIZATION MASK =====
	# During the first ~seconds of recording, the arena is often unstable
	# Mask entire frame to prevent tracking during this adjustment period
	if adjustStartFlag:
		dframe[:, :] = 0

	return dframe


# ===== SLEEP/INACTIVITY DETECTION =====

def skip_settings(settings):
	# Skip initial portion of video where fly is asleep or inactive
	# Returns the number of seconds to skip from the beginning of each video
	# Skipping ensures analysis only includes active behavior
	# Parameters:
	#   settings: tracking settings dictionary (contains genotype and filename)
	# Returns: skip_time in seconds (will be converted to frames by multiplying by fps)
	
	fname = settings['file'].split('.')[0]
	
	# Default: don't skip any frames
	skip = 0

	# ===== WILD-TYPE (WT) =====
	if settings['groupName'] == 'WT':
		# Individual calibration for each WT recording
		if fname == 'gradient_10-22-2020_17-21':
			skip = 10
		elif fname == 'gradient_10-28-2020_17-39':
			skip = 10
		elif fname == 'gradient_11-04-2020_15-36':
			skip = 10
		elif fname == 'shallow_02-26-2024_10-30':
			skip = 10
		elif fname == 'shallow_02-26-2024_12-40':
			skip = 20   # Longer sleep period
		elif fname == 'shallow_02-26-2024_14-30':
			skip = 10
		elif fname == 'shallow_02-27-2024_12-49':
			skip = 750  # Very long sleep/inactive period
		elif fname == 'shallow_02-27-2024_15-07':
			skip = 10
		elif fname == 'shallow_03-04-2024_11-28':
			skip = 10

	# ===== ABLATED FLIES =====
	elif settings['groupName'] == 'Ablated':
		if fname == 'shallow_02-22-2024_12-32':
			skip = 10

	# ===== SS90+ GENOTYPE =====
	elif settings['groupName'] == 'SS90+':
		skip = 10  # Default skip for this line
		# Additional adjustments for specific recordings
		if fname == 'shallow_03-19-2024_10-29':
			skip = 20
		elif fname == 'shallow_03-21-2024_11-36':
			skip = 20
		elif fname == 'shallow_03-21-2024_13-32':
			skip = 25
		elif fname == 'shallow_03-21-2024_14-37':
			skip = 20
		elif fname == 'shallow_03-21-2024_16-04':
			skip = 20
		elif fname == 'shallow_03-22-2024_12-42':
			skip = 20
		elif fname == 'shallow_03-22-2024_13-14':
			skip = 15
		elif fname == 'shallow_03-22-2024_13-38':
			skip = 15
		elif fname == 'shallow_03-22-2024_15-26':
			skip = 20
		elif fname == 'shallow_03-25-2024_10-19':
			skip = 40   # Longer sleep
		elif fname == 'shallow_03-25-2024_13-24':
			skip = 15
		elif fname == 'shallow_03-25-2024_13-45':
			skip = 20
		elif fname == 'shallow_03-25-2024_14-07':
			skip = 20
		elif fname == 'shallow_03-25-2024_15-11':
			skip = 15

	# ===== SS90Kir GENOTYPE =====
	elif settings['groupName'] == 'SS90Kir':
		skip = 10  # Default skip for this line
		if fname == 'shallow_03-19-2024_11-37':
			skip = 20
		elif fname == 'shallow_03-19-2024_14-55':
			skip = 20
		elif fname == 'shallow_03-19-2024_15-41':
			skip = 40   # Longer sleep
		elif fname == 'shallow_03-21-2024_12-25':
			skip = 20
		elif fname == 'shallow_03-21-2024_13-09':
			skip = 50   # Longer sleep
		elif fname == 'shallow_03-21-2024_14-16':
			skip = 12
		elif fname == 'shallow_03-21-2024_15-42':
			skip = 20
		elif fname == 'shallow_03-25-2024_14-28':
			skip = 20

	# ===== SS408+ GENOTYPE =====
	elif settings['groupName'] == 'SS408+':
		if fname == 'shallow_03-14-2024_15-43':
			skip = 20
		elif fname == 'shallow_03-15-2024_14-57':
			skip = 15
		elif fname == 'shallow_03-15-2024_14-33':
			skip = 15
		elif fname == 'shallow_03-18-2024_14-38':
			skip = 40   # Longer sleep
		elif fname == 'shallow_03-18-2024_14-59':
			skip = 15
		elif fname == 'shallow_03-18-2024_15-20':
			skip = 25
		elif fname == 'shallow_03-18-2024_16-24':
			skip = 20

	# ===== SS408-Kir GENOTYPE =====
	elif settings['groupName'] == 'SS408-Kir':
		if fname == 'shallow_03-15-2024_10-05':
			skip = 15
		elif fname == 'shallow_03-15-2024_11-30':
			skip = 60   # Very long sleep
		elif fname == 'shallow_03-15-2024_12-41':
			skip = 10
		elif fname == 'shallow_03-18-2024_13-26':
			skip = 20
		elif fname == 'shallow_03-18-2024_16-02':
			skip = 20

	# ===== SS98+ GENOTYPE =====
	elif settings['groupName'] == 'SS98+':
		if fname == 'shallow_03-27-2024_10-24':
			skip = 15
		elif fname == 'shallow_03-29-2024_10-02':
			skip = 50
		elif fname == 'shallow_03-29-2024_11-44':
			skip = 330  # Very long sleep
		elif fname == 'shallow_03-29-2024_12-29':
			skip = 15
		elif fname == 'shallow_03-29-2024_12-52':
			skip = 15
		elif fname == 'shallow_03-29-2024_13-36':
			skip = 15
		elif fname == 'shallow_04-01-2024_09-36':
			skip = 10
		elif fname == 'shallow_04-01-2024_10-44':
			skip = 30
		elif fname == 'shallow_04-04-2024_12-43':
			skip = 125  # Very long sleep
		elif fname == 'shallow_04-04-2024_14-08':
			skip = 10

	# ===== SS98Kir GENOTYPE =====
	elif settings['groupName'] == 'SS98Kir':
		if fname == 'shallow_03-27-2024_10-46':
			skip = 30
		elif fname == 'shallow_03-28-2024_15-11':
			skip = 10
		elif fname == 'shallow_04-01-2024_10-22':
			skip = 15
		elif fname == 'shallow_04-01-2024_12-18':
			skip = 30
		elif fname == 'shallow_04-01-2024_12-47':
			skip = 15
		elif fname == 'shallow_04-01-2024_13-43':
			skip = 20
		elif fname == 'shallow_04-02-2024_14-00':
			skip = 15
		elif fname == 'shallow_04-02-2024_14-44':
			skip = 20
		elif fname == 'shallow_04-02-2024_15-06':
			skip = 10
		elif fname == 'shallow_04-02-2024_15-29':
			skip = 15
		elif fname == 'shallow_04-02-2024_16-34':
			skip = 20

	# ===== SS00096-+ GENOTYPE =====
	elif settings['groupName'] == 'SS00096-+':
		if fname == 'gradient_03-09-2021_13-14':
			skip = 20
		elif fname == 'gradient_03-09-2021_15-05':
			skip = 100  # Very long sleep
		elif fname == 'gradient_03-10-2021_14-24':
			skip = 20
		elif fname == 'gradient_03-10-2021_14-52':
			skip = 15
		elif fname == 'gradient_03-10-2021_15-53':
			skip = 10
		elif fname == 'gradient_12-21-2020_17-30':
			skip = 15
		elif fname == 'gradient_12-21-2020_17-57':
			skip = 15
		elif fname == 'gradient_12-22-2020_16-57':
			skip = 10

	# ===== SS00096-Kir GENOTYPE =====
	elif settings['groupName'] == 'SS00096-Kir':
		skip = 10  # Default skip for this line
		if fname == 'gradient_03-09-2021_12-06':
			skip = 15
		elif fname == 'gradient_12-21-2020_16-46':
			skip = 290  # Very long sleep

	# ===== Kir-+ GENOTYPE =====
	elif settings['groupName'] == 'Kir-+':
		if fname == 'gradient_11-25-2020_12-49':
			skip = 10
		elif fname == 'shallow_03-08-2024_09-56':
			skip = 90   # Very long sleep
		elif fname == 'shallow_03-08-2024_14-52':
			skip = 15

	# ===== FC2Kir GENOTYPE =====
	elif settings['groupName'] == 'FC2Kir':
		if fname == 'shallow_10-22-2024_11-49':
			skip = 60   # Long sleep
		elif fname == 'shallow_10-22-2024_11-25':
			skip = 30

	return skip