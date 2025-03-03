from functions_tracking import *
import concurrent.futures 
import functools
import time

# create partial function here to pass extra fixed variables to the function
gap = 0
frobenius_threshold = 20000
track_video_params = functools.partial(track_video, gap=gap, frob_thresh = frobenius_threshold)


groupNames = ['FC2Kir']

if __name__ == '__main__':
	for groupName in groupNames:

		inputDir = 'videos/'+groupName+'/'
		dirs = os.listdir(inputDir)
		dirs = [file for file in dirs if (file.split('.')[-1] == 'avi') or (file.split('.')[-1] == 'mp4')] # keep only mp4 and avi files

		# create output directory
		outputDir = 'outputs/outputs_'+groupName+'/'
		CHECK_FOLDER = os.path.isdir(outputDir)
		if not CHECK_FOLDER:
			os.makedirs(outputDir)
			print('Created folder: ' + outputDir)


		start = time.perf_counter()

		dirs0 = [inputDir+file for file in dirs]
		with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
			futures = {executor.submit(track_video_params, file): file for file in dirs0}

			for future in concurrent.futures.as_completed(futures):
				file_path = futures[future]
				try:
					result = future.result()
					pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2) = result

					if isinstance(pos, float):  # Tracking did not work
						continue

					file_name = os.path.basename(file_path)
					output_file = outputDir + file_name.split('.')[0] + '.output'
					with open(output_file, 'wb') as f1:
						pickle.dump((pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)), f1)
					print(f'Saved results for {file_name}')

				except Exception as e:
					print(f'Error processing {file_path}: {e}')

		finish = time.perf_counter()
		print(f'Finished processing {groupName} in {round(finish - start, 2)} seconds')
	
      
    
      
      
      
      
      
      
		# 	results = executor.map(track_video_params, dirs0)

		# for k,result in enumerate(results):
		# 	pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2) = result

		# 	if isinstance(pos, float):	# Tracking did not work
		# 		continue

		# 	file = dirs[k]

		# 	f1 = open(outputDir+file.split('.')[0]+'.output','wb')
		# 	pickle.dump((pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)),f1)
		# 	f1.close()
		
		# finish = time.perf_counter()
		# print(f'Finished in {round(finish-start,2)} seconds')








# Parameters
		
## ['WT', 'Ablated', 'flat-25', 'Kir-+', 'L-ablated', 'R-ablated', 'AC-Kir', 'AC-+']
# thresh = 40
# perc = 25
# gap = 0

## ['HC-Kir', 'HC-+', 'GR28-TRPA1', 'Gr28-exc66']
# thresh = 40
# perc = 20
		
# ['R4-+', 'R31A12-+']
# thresh = 35
# perc = 35
		
# ['R4-Kir', 'Gr28-exc8'] ['newAblated', 'newWT']
# thresh = 40
# perc = 30
		
# ['R31A12-Kir', 'SS00096-+']
# thresh = 40
# perc = 35
		