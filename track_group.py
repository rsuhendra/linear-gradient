from functions_tracking import *
import concurrent.futures 
import time
import os
import pickle

# ===== CONFIG =====
gap = 0
frobenius_threshold = 20000

# max_workers recommendations:
#   Laptop (4-8 cores):     max_workers = 2-4
#   Desktop (8-16 cores):   max_workers = 6-8
#   Workstation (16+ cores): max_workers = 12-16
#   Cluster job (per node): max_workers = 2-4 (let cluster handle parallelization)

max_workers = 4
groupNames = ['WT']

# ===== RUN =====

def process_video(file_path):
	"""Process single video and return results"""
	try:
		result = track_video(file_path, gap=gap, frob_thresh=frobenius_threshold)
		pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2) = result
		
		if isinstance(pos, float):  # Tracking failed
			return None, f"Failed: {file_path}"
		
		return (pos, angles, vels, settings, originalTrackingInfo, (pos2, angles2, vels2)), None
	except Exception as e:
		return None, f"Error: {file_path}: {e}"


if __name__ == '__main__':
	for groupName in groupNames:
		inputDir = f'videos/{groupName}/'
		outputDir = f'outputs/outputs_{groupName}/'
		create_directory(outputDir)
		
		dirs = [os.path.join(inputDir, f) for f in os.listdir(inputDir) 
		        if f.endswith(('.mp4', '.avi'))]
		
		if not dirs:
			print(f"No videos found in {inputDir}\n")
			continue
		
		print(f"Processing {len(dirs)} videos with {max_workers} workers\n")
		start = time.perf_counter()
		
		with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
			futures = {executor.submit(process_video, f): f for f in dirs}
			
			for future in concurrent.futures.as_completed(futures):
				result, error = future.result()
				file_path = futures[future]
				
				if error:
					print(f"⚠️  {error}")
					continue
				
				# Save results
				file_name = os.path.basename(file_path).split('.')[0]
				output_file = os.path.join(outputDir, file_name + '.output')
				with open(output_file, 'wb') as f:
					pickle.dump(result, f)
				print(f"✓ {file_name}")
		
		elapsed = round(time.perf_counter() - start, 2)
		print(f"\nDone in {elapsed}s\n")