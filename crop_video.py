import cv2
import os
import ffmpeg

# ===== VIDEO CROPPING FUNCTIONS =====

# Method 1: Crop videos using OpenCV
# Reads video frame-by-frame, crops each frame, and writes to output file
def crop_videos1(input_folder, output_folder, crop_region):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each video file in the input folder
    for filename in os.listdir(input_folder):
        # Only process video files
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            # Skip this specific file (likely problematic or already processed)
            if 'shallow_02-22-2024_09-45' in filename:
                continue
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            # Open the input video file
            cap = cv2.VideoCapture(input_filepath)
            if not cap.isOpened():
                print(f"Error: Unable to open {input_filepath}")
                continue
            
            # Extract video properties (frames per second, dimensions)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Set codec to mp4v for H.264 video encoding
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            
            # Initialize VideoWriter with output file, codec, fps, and cropped frame dimensions
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (crop_region[2], crop_region[3]))
            
            # Process each frame: read, crop, and write to output video
            while True:
                ret, frame = cap.read()
                # Break if no more frames to read
                if not ret:
                    break
                
                # Crop the frame using the specified region (x, y, width, height)
                cropped_frame = frame[crop_region[1]:crop_region[1]+crop_region[3], crop_region[0]:crop_region[0]+crop_region[2]]
                out.write(cropped_frame)
                
            # Clean up resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()


# Helper function for Method 2: Crops a single video file using ffmpeg
# Parameters: input file path, output file path, and crop coordinates (x, y, width, height)
def cropvid2(input_file, output_file, x, y, width, height):
    # Use ffmpeg-python to crop video and save to output file
    (
        ffmpeg
        .input(input_file)
        .crop(x, y, width, height)
        .output(output_file)
        .run()
    )

    
# Method 2: Crop videos using ffmpeg
# More efficient for large video files, uses hardware acceleration where available
def crop_videos2(input_directory, output_directory, crop_region):
	
	# Unpack crop region coordinates
	(x, y, width, height) = crop_region 
	
	# Get list of all mp4 files in the input directory
	input_files = [f for f in os.listdir(input_directory) if f.endswith('.mp4')]

	# Process each video file
	for input_file in input_files:
		input_path = os.path.join(input_directory, input_file)
		output_path = os.path.join(output_directory, input_file)
		# Crop the video using ffmpeg
		cropvid2(input_path, output_path, x, y, width, height)


# ===== USAGE =====
# Set input and output directories
input_folder = "videos/Flat25"
output_folder = "videos/Flat25y"
# Define crop region as (x_offset, y_offset, width, height)
crop_region = (20, 80, 1240, 690)  # (x, y, width, height) of the region to crop

# Run the cropping function (Method 1 using OpenCV)
crop_videos1(input_folder, output_folder, crop_region)
# Alternative: Use Method 2 with ffmpeg (commented out)
# crop_videos2(input_folder, output_folder, crop_region)
