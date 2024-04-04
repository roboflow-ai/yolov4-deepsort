import os
import subprocess

def find_and_process_mp4_files(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mp4'):
                mp4_path = os.path.join(root, file)
                cmd = [
                    'python3', 'clip_object_tracker_bbox_only_separate_columns.py',
                    '--weights', 'models/yolov7x.pt',
                    '--conf', '0.5',
                    '--save-txt',
                    '--save-conf',
                    '--name', os.path.splitext(file)[0],  # Name of the mp4 file without extension
                    '--source', mp4_path,  # Path to the mp4 file
                    '--detection-engine', 'yolov7',
                    '--info'
                ]
                print(f"Running command for file: {mp4_path}")
                subprocess.run(cmd)

# Replace 'path_to_folder' with the path to your target folder
find_and_process_mp4_files('/home/marya/Desktop/zero-shot-object-tracking/data/video/US_12_ToD_Analysis')
