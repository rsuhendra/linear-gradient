import cv2
import os
import ffmpeg


def crop_videos1(input_folder, output_folder, crop_region):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over video files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            if 'shallow_02-22-2024_09-45' in filename:
                continue
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            # Open the input video file
            cap = cv2.VideoCapture(input_filepath)
            if not cap.isOpened():
                print(f"Error: Unable to open {input_filepath}")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            
            # Define the codec and create VideoWriter object
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (crop_region[2], crop_region[3]))
            
            # Read and crop each frame, then write to output video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cropped_frame = frame[crop_region[1]:crop_region[1]+crop_region[3], crop_region[0]:crop_region[0]+crop_region[2]]
                out.write(cropped_frame)
                
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()


def cropvid2(input_file, output_file, x, y, width, height):
    (
        ffmpeg
        .input(input_file)
        .crop(x, y, width, height)
        .output(output_file)
        .run()
    )
    
def crop_videos2(input_directory, output_directory, crop_region):
	
	(x, y, width, height) = crop_region 
	
	# List input files in the directory
	input_files = [f for f in os.listdir(input_directory) if f.endswith('.mp4')]

	# Loop through input files and crop them
	for input_file in input_files:
		input_path = os.path.join(input_directory, input_file)
		output_path = os.path.join(output_directory, input_file)
		cropvid2(input_path, output_path, x, y, width, height)



# Usage
input_folder = "videos/Flat25"
output_folder = "videos/Flat25y"
crop_region = (20, 80, 1240, 690)  # (x, y, width, height) of the region to crop

crop_videos1(input_folder, output_folder, crop_region)
# crop_videos2(input_folder, output_folder, crop_region)


