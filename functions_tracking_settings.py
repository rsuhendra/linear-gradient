import numpy as np
import datetime

def mask_lighting(dframed, adjustStartFlag, settings):
	dframe = np.copy(dframed)
	# groupName = settings['groupName']

	fname = settings['file']
	date = datetime.datetime.strptime(fname.split('_')[-2], "%m-%d-%Y")

	date1 = datetime.datetime.strptime("10-20-2020", "%m-%d-%Y")
	date2 = datetime.datetime.strptime("12-04-2020", "%m-%d-%Y")
	date3 = datetime.datetime.strptime("02-25-2021", "%m-%d-%Y")
	date4 = datetime.datetime.strptime("04-03-2021", "%m-%d-%Y")
	date5 = datetime.datetime.strptime("05-25-2021", "%m-%d-%Y")
	date6 = datetime.datetime.strptime("08-25-2023", "%m-%d-%Y")

	if date < date3:
		dframe[0:20,0:150] = 0	# upper left
		dframe[20:40,0:40] = 0	# upper left
		dframe[40:100,0:20] = 0	# upper left

		dframe[-20:,0:400] = 0	# bottom left

		dframe[0:20,-200:] = 0	# upper right
		dframe[20:60,-40:] = 0	# upper right
		dframe[60:200,-20:] = 0	# upper left

		if date1 <= date < date2:
			dframe[500:590,0:25] = 0	# bottom left
		elif date == date2:
			dframe[500:625,0:25] = 0	# bottom left
		elif date2 < date < date3:
			dframe[525:675,0:40] = 0	# bottom left
	
	elif date3 <= date < date4:
		dframe[0:20,0:150] = 0	# upper left
		dframe[20:100,0:120] = 0	# upper left
		dframe[100:130,0:20] = 0	# upper left

		dframe[-20:,0:400] = 0	# bottom left

		dframe[0:20,-200:] = 0	# upper right
		dframe[20:60,-60:] = 0	# upper right
		dframe[60:200,-20:] = 0	# upper left
		
		dframe[525:700,0:40] = 0	# bottom left

	elif date4 <= date < date5:
		dframe[0:20,0:200] = 0	# upper left
		dframe[20:170,0:140] = 0	# upper left
		dframe[100:130,0:20] = 0	# upper left

		dframe[-20:,0:400] = 0	# bottom left

		dframe[0:20,-200:] = 0	# upper right
		dframe[20:100,-120:] = 0	# upper right
		dframe[100:200,-20:] = 0	# upper left
		
		dframe[525:650,0:40] = 0	# bottom left

	elif date5 < date < date6:
		dframe[0:50,0:50] = 0	# upper left
		dframe[250:,0:30] = 0	# bottom left
		dframe[0:50,-50:] = 0	# upper right
	
	# elif date > date6:
	# 	dframe[0:100,-300:] = 0	# upper right
  
	elif date >= date6:
		# dframe[0:75,-300:] = 0	# upper right
		pass

	if settings['groupName'] in ['61933+', '61933_kir', 'DNB05+', 'DNB05_Kir','HdB+', 'HdB_Kir','HdC+', 'HdC_Kir']:
		dframe[0:100,0:30] = 0	# upper left
		dframe[0:15,0:200] = 0	# upper left
		dframe[310:,0:30] = 0	# bottom left
		dframe[280:310,0:100] = 0	# bottom left
		dframe[0:100,-50:] = 0	# upper right
		dframe[0:20,500:] = 0	# upper right
		dframe[310:,-30:] = 0	# bottom right


	# if groupName == 'FL50_7mm':
	# 	dframe[0:50,0:30] = 0	# upper left
	# 	dframe[0:20,0:200] = 0	# upper left
	# 	dframe[270:,0:30] = 0	# bottom left
	# 	dframe[0:20,-100:] = 0	# upper right
	# 	dframe[0:100,-30:] = 0	# upper right
	# 	dframe[300:,-50:] = 0	# bottom right

	# elif groupName == 'FL50_20mm':
	# 	dframe[0:100,0:30] = 0	# upper left
	# 	dframe[0:15,0:200] = 0	# upper left
	# 	dframe[310:,0:30] = 0	# bottom left
	# 	dframe[0:100,-30:] = 0	# upper right
	# 	dframe[0:20,500:] = 0	# upper right
	# 	dframe[310:,-30:] = 0	# bottom right
	
	# elif 'HdC' in groupName:
	# 	dframe[0:100,0:30] = 0	# upper left
	# 	dframe[0:15,0:200] = 0	# upper left
	# 	dframe[310:,0:30] = 0	# bottom left
	# 	dframe[280:310,0:100] = 0	# bottom left
	# 	dframe[0:100,-50:] = 0	# upper right
	# 	dframe[0:20,500:] = 0	# upper right
	# 	dframe[310:,-30:] = 0	# bottom right
	
	# elif groupName in ['HdB_Kir', 'HdB+', '61933_kir', 
	# 				'61933+', 'DNB05_Kir', 'DNB05+','kir+']:
	# 	dframe[0:50,0:50] = 0	# upper left
	# 	dframe[250:,0:30] = 0	# bottom left
	# 	dframe[0:50,-50:] = 0	# upper right

	if adjustStartFlag:
		dframe[:,:] = 0

	return dframe

def best_settings(groupName, settings):

	split = ['Ablated']

	group1 = ['WT', 'Flat25', 'SS408+', 'SS408-Kir', 'SS90+', 'SS90Kir', 'SS98+', 'SS98Kir', 'SS131+', 'SS131Kir']
	group2 = ['Kir-+', 'L-ablated', 'R-ablated', 'AC-Kir', 'AC-+', 'Gr28-exc8'] + ['R4-+','R4-Kir', 'R31A12-Kir', 'R31A12-+']
	group3 = ['HC-Kir', 'HC-+', 'GR28-TRPA1', 'Gr28-exc66']

	group4 = ['SS00096-+', 'SS00096-Kir']
	group5 = ['PFNd-+', 'PFNd-Kir', 'PFNv-+', 'PFNv-Kir']
 
	group6 = ['FC2+', 'FC2Kir']

	groupC = ['61933+', '61933_kir', 'DNB05+', 'DNB05_Kir','HdB+', 'HdB_Kir','HdC+', 'HdC_Kir']

	groupA = group1 + group2 + group3
	groupB = group4 + group5
	groupD = group6

	if groupName in groupA:
		thresh = 40
		perc = 30
	elif groupName in groupB:
		thresh = 40
		perc = 35
	elif groupName in groupC:
		thresh = 35
		perc = 40
	elif groupName in groupD:
		thresh = 50
		perc = 30
	elif groupName in split:
		if groupName == 'Ablated':
			date1 = datetime.datetime.strptime("10-28-2020", "%m-%d-%Y")
			if settings['date'] < date1:
				thresh = 20
				perc = 35
			else:
				thresh = 40
				perc = 35

	return thresh, perc

def skip_settings(settings):
	fname = settings['file'].split('.')[0]

	# Skip certain parts of video where fly is asleep

	skip = 0

	if settings['groupName'] == 'WT':
		if fname == 'gradient_10-22-2020_17-21':
			skip = 10
		elif fname == 'gradient_10-28-2020_17-39':
			skip = 10
		elif fname == 'gradient_11-04-2020_15-36':
			skip = 10
		elif fname == 'shallow_02-26-2024_10-30':
			skip = 10
		elif fname == 'shallow_02-26-2024_12-40':
			skip = 20
		elif fname == 'shallow_02-26-2024_14-30':
			skip = 10
		elif fname == 'shallow_02-27-2024_12-49':
			skip = 750
		elif fname == 'shallow_02-27-2024_15-07':
			skip = 10
		elif fname == 'shallow_03-04-2024_11-28':
			skip = 10
	
	elif settings['groupName'] == 'Ablated':
		if fname == 'shallow_02-22-2024_12-32':
			skip = 10

	elif settings['groupName'] == 'SS90+':
		skip = 10
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
			skip = 40
		elif fname == 'shallow_03-25-2024_13-24':
			skip = 15
		elif fname == 'shallow_03-25-2024_13-45':
			skip = 20
		elif fname == 'shallow_03-25-2024_14-07':
			skip = 20
		elif fname == 'shallow_03-25-2024_15-11':
			skip = 15
		

 
	elif settings['groupName'] == 'SS90Kir':
		skip = 10
		if fname == 'shallow_03-19-2024_11-37':
			skip = 20
		elif fname == 'shallow_03-19-2024_14-55':
			skip = 20
		elif fname == 'shallow_03-19-2024_15-41':
			skip = 40
		elif fname == 'shallow_03-21-2024_12-25':
			skip = 20
		elif fname == 'shallow_03-21-2024_13-09':
			skip = 50
		elif fname == 'shallow_03-21-2024_14-16':
			skip = 12
		elif fname == 'shallow_03-21-2024_15-42':
			skip = 20
		elif fname == 'shallow_03-25-2024_14-28':
			skip = 20

	elif settings['groupName'] == 'SS408+':
		if fname == 'shallow_03-14-2024_15-43':
			skip = 20
		elif fname == 'shallow_03-15-2024_14-57':
			skip = 15
		elif fname == 'shallow_03-15-2024_14-33':
			skip = 15
		elif fname == 'shallow_03-18-2024_14-38':
			skip = 40
		elif fname == 'shallow_03-18-2024_14-59':
			skip = 15
		elif fname == 'shallow_03-18-2024_15-20':
			skip = 25
		elif fname == 'shallow_03-18-2024_16-24':
			skip = 20
	
	elif settings['groupName'] == 'SS408-Kir':
		if fname == 'shallow_03-15-2024_10-05':
			skip = 15
		elif fname == 'shallow_03-15-2024_11-30':
			skip = 60
		elif fname == 'shallow_03-15-2024_12-41':
			skip = 10
		elif fname == 'shallow_03-18-2024_13-26':
			skip = 20
		elif fname == 'shallow_03-18-2024_16-02':
			skip = 20
	
	elif settings['groupName'] == 'SS98+':
		if fname == 'shallow_03-27-2024_10-24':
			skip = 15
		elif fname == 'shallow_03-29-2024_10-02':
			skip = 50
		elif fname == 'shallow_03-29-2024_11-44':
			skip = 330
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
			skip = 125
		elif fname == 'shallow_04-04-2024_14-08':
			skip = 10
	
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
	
	elif settings['groupName'] == 'SS00096-+':
		if fname == 'gradient_03-09-2021_13-14':
			skip = 20
		elif fname == 'gradient_03-09-2021_15-05':
			skip = 100
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

	elif settings['groupName'] == 'SS00096-Kir':
		skip = 10
		if fname == 'gradient_03-09-2021_12-06':
			skip = 15
		elif fname == 'gradient_12-21-2020_16-46':
			skip = 290

	elif settings['groupName'] == 'Kir-+':
		if fname == 'gradient_11-25-2020_12-49':
			skip = 10
		elif fname == 'shallow_03-08-2024_09-56':
			skip = 90
		elif fname == 'shallow_03-08-2024_14-52':
			skip = 15
   
	elif settings['groupName'] == 'FC2Kir':
		if fname == 'shallow_10-22-2024_11-49':
			skip = 60
		elif fname == 'shallow_10-22-2024_11-25':
			skip = 30
   
	return skip