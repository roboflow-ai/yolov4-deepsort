import pandas as pd
import os
from glob import glob

def merge_csv_with_timestamps(output_csv_dir, timestamps_csv_dir, merged_csv_dir):
    output_csv_files = glob(os.path.join(output_csv_dir, '*.csv'))
    
    for output_csv_path in output_csv_files:
        base_file_name = os.path.basename(output_csv_path)
        timestamps_csv_path = os.path.join(timestamps_csv_dir, base_file_name)
        
        if not os.path.exists(timestamps_csv_path):
            print(f"No matching timestamp file found for {base_file_name}. Skipping...")
            continue
        
        try:
            output_df = pd.read_csv(output_csv_path, on_bad_lines='warn')
            timestamps_df = pd.read_csv(timestamps_csv_path, on_bad_lines='warn')
        except Exception as e:
            print(f"Error reading {base_file_name}: {e}")
            continue

        merged_df = pd.merge(output_df, timestamps_df, left_on='Frame', right_on='Frame_Number', how='left')
        final_df = merged_df[['Datetime', 'Frame', 'Track', 'Class', 'Class_ID', 'xmin', 'ymin', 'xmax', 'ymax', 'Time', 'Date', 'Red_Light']].copy()
        final_df.rename(columns={'Frame': 'FrameNumber', 'Red_Light': 'Color'}, inplace=True)
        final_df['Color'].fillna('Not Red', inplace=True)
        final_df['Color'] = final_df['Color'].apply(lambda x: 'Red' if x == True else 'Not Red')
        
        merged_csv_path = os.path.join(merged_csv_dir, base_file_name)
        final_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved to {merged_csv_path}")


output_csv_dir = '/home/marya1/Documents/MnDoTNRToR/inference/detect_objects/processing'
timestamps_csv_dir = '/home/marya1/Documents/MnDoTNRToR/inference/detect_red'
merged_csv_dir = '/home/marya1/Documents/MnDoTNRToR/inference/final_csv'

# Use the directories as defined in the script
merge_csv_with_timestamps(output_csv_dir, timestamps_csv_dir, merged_csv_dir)
