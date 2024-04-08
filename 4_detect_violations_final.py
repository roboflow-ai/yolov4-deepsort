import cv2
import pandas as pd
from os.path import join, basename, splitext
from glob import glob
from sklearn.cluster import DBSCAN
import numpy as np

csv_path = '/home/marya1/Documents/MnDoTNRToR/inference/final_csv'
output_path='/home/marya1/Documents/MnDoTNRToR/inference/detect_violations'
videos_path = '/home/marya1/Documents/MnDoTNRToR/videos/processing'
output_file_path='/home/marya1/Documents/MnDoTNRToR/inference/detect_violations_final'

# Define global variables for ROI drawing
drawing = False
roi_selected = False
top_left_pt, bottom_right_pt = None, None

def draw_roi(event, x, y, flags, param):
    global drawing, roi_selected, top_left_pt, bottom_right_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        roi_selected = True

def is_within_roi(xmin, ymin, xmax, ymax, roi):
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
    return xmin >= roi_xmin and xmax <= roi_xmax and ymin >= roi_ymin and ymax <= roi_ymax

def calculate_speed(tracks_df):
    tracks_df['x_center'] = (tracks_df['xmin'] + tracks_df['xmax']) / 2
    tracks_df['y_center'] = (tracks_df['ymin'] + tracks_df['ymax']) / 2
    tracks_df.sort_values(by=['Track', 'DateTime'], inplace=True)
    tracks_df['x_diff'] = tracks_df.groupby('Track')['x_center'].diff()
    tracks_df['y_diff'] = tracks_df.groupby('Track')['y_center'].diff()
    tracks_df['time_diff'] = tracks_df.groupby('Track')['DateTime'].diff().dt.total_seconds()
    tracks_df['speed'] = ((tracks_df['x_diff']**2 + tracks_df['y_diff']**2)**0.5) / tracks_df['time_diff']
    tracks_df['speed'].fillna(0, inplace=True)
    return tracks_df

def vehicle_exited_roi(tracks_df, roi, buffer_distance=10):
    def did_exit(row):
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
        near_exit = row['xmax'] > roi_xmax - buffer_distance or row['ymax'] > roi_ymax - buffer_distance or row['xmin'] < roi_xmin + buffer_distance or row['ymin'] < roi_ymin + buffer_distance
        likely_to_exit = row['speed'] > 0 and near_exit
        return likely_to_exit
    
    # Create a copy to avoid SettingWithCopyWarning when modifying
    modified_df = tracks_df.copy()
    
    # Ensure 'Exited_ROI' column exists to avoid issues with assigning to a non-existent column
    if 'Exited_ROI' not in modified_df.columns:
        modified_df['Exited_ROI'] = False

    # Get the last row index of each group
    last_row_indices = modified_df.groupby('Track').tail(1).index
    
    # Use .loc to update the 'Exited_ROI' column for these indices
    modified_df.loc[last_row_indices, 'Exited_ROI'] = modified_df.loc[last_row_indices].apply(did_exit, axis=1)
    
    return modified_df


def count_vehicles_and_violations(csv_file, roi, valid_classes=[1, 2, 3, 5, 7], min_frames=60):
    df = pd.read_csv(csv_file)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    
    df = calculate_speed(df)  # Calculate speed for the df
    
    red_light_phases = df[df['Color'] == 'Red'].groupby((df['Color'] != df['Color'].shift()).cumsum())
    results = []

    for _, phase in red_light_phases:
        start_of_red_light = phase['DateTime'].iloc[0]
        next_phase_start = phase['DateTime'].iloc[-1] + pd.Timedelta(seconds=1)
        
        if (next_phase_start - start_of_red_light).total_seconds() <= 1:
            continue

        tracks_in_roi = df[(df['DateTime'] >= start_of_red_light) & (df['DateTime'] < next_phase_start)]
        tracks_in_roi = tracks_in_roi[tracks_in_roi.apply(lambda row: is_within_roi(row['xmin'], row['ymin'], row['xmax'], row['ymax'], roi), axis=1)]
        valid_tracks_in_roi = tracks_in_roi[tracks_in_roi['Class_ID'].isin(valid_classes)]
        
        valid_tracks_in_roi = vehicle_exited_roi(valid_tracks_in_roi, roi)  # Check for exits

        frame_counts = valid_tracks_in_roi.groupby('Track').size()
        tracks_more_than_min_frames = frame_counts[frame_counts >= min_frames].index
        valid_tracks_in_roi = valid_tracks_in_roi[valid_tracks_in_roi['Track'].isin(tracks_more_than_min_frames)]

        exiting_tracks = valid_tracks_in_roi[valid_tracks_in_roi['Exited_ROI'] == True].groupby('Track').last()
        exiting_tracks = exiting_tracks[exiting_tracks.index.isin(tracks_more_than_min_frames)]
        violations = exiting_tracks[exiting_tracks['Color'] == 'Red'].index.nunique()
        violating_track_ids = exiting_tracks[exiting_tracks['Color'] == 'Red'].index.unique().tolist()
        total_vehicles = valid_tracks_in_roi['Track'].nunique()

        total_vehicles_complying = valid_tracks_in_roi['Track'].nunique() - violations
        complying_track_ids = valid_tracks_in_roi[~valid_tracks_in_roi['Track'].isin(violating_track_ids)]['Track'].unique().tolist()

        results.append({
            'Start_of_Red_Light': start_of_red_light.strftime('%Y-%m-%d %H:%M:%S'),
            'Beginning_of_Not_Red': next_phase_start.strftime('%Y-%m-%d %H:%M:%S'),
            'Total_Vehicles': total_vehicles,
            'Vehicles_Complying': total_vehicles_complying,
            'Vehicles_Violating': violations,
            'Complying Tracks': complying_track_ids,
            'Violating Tracks': violating_track_ids,
            'Compliance Rate': (total_vehicles - violations) / total_vehicles if total_vehicles > 0 else 1
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = pd.concat([results_df, pd.DataFrame([{'Start_of_Red_Light': 'Overall', 'Total_Vehicles': results_df['Total_Vehicles'].sum(), 'Vehicles_Violating': results_df['Vehicles_Violating'].sum(), 'Compliance Rate': results_df['Compliance Rate'].mean()}])], ignore_index=True)
    
    return results_df

def process_csv_with_time_condition(input_file_path, output_file_path):
    """
    Processes a CSV file by sorting based on 'Start_of_Red_Light', then iterates
    through each row to keep rows where 'Start_of_Red_Light' or 'Beginning_of_Not_Red'
    in the next row is not the same as in the current row, and if the time difference
    between them is exactly 1 second. The first row is always kept.

    Parameters:
    - input_file_path: Path to the input CSV file.
    - output_file_path: Path where the processed CSV file will be saved.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)
    
    # Convert columns to datetime, coercing errors to NaT
    df['Start_of_Red_Light'] = pd.to_datetime(df['Start_of_Red_Light'], errors='coerce')
    df['Beginning_of_Not_Red'] = pd.to_datetime(df['Beginning_of_Not_Red'], errors='coerce')

    # Filter out rows where datetime conversion was unsuccessful
    df = df.dropna(subset=['Start_of_Red_Light', 'Beginning_of_Not_Red'])

    # Sort the DataFrame based on the 'Start_of_Red_Light' column
    df_sorted = df.sort_values(by='Start_of_Red_Light')
    
    # Initialize a list to keep rows that meet the criteria, starting with the first row
    rows_to_keep = [0]
    
    # Iterate through the sorted DataFrame starting from the second row
    for i in range(1, len(df_sorted)):
        previous_row = df_sorted.iloc[i - 1]
        current_row = df_sorted.iloc[i]
        
        # Calculate time differences
        diff_start = abs((current_row['Start_of_Red_Light'] - previous_row['Start_of_Red_Light']).total_seconds())
        diff_end = abs((current_row['Beginning_of_Not_Red'] - previous_row['Beginning_of_Not_Red']).total_seconds())
        duration=abs((current_row['Beginning_of_Not_Red']-current_row['Start_of_Red_Light']).total_seconds())
        
        # Determine if the current row should be kept based on the conditions
        if (duration>=10):
            if not(diff_start <= 5 or diff_end <= 5):
                rows_to_keep.append(i)
    
    # Filter the DataFrame based on the calculated rows to keep
    df_filtered = df_sorted.iloc[rows_to_keep]

    # Save the filtered DataFrame to the output CSV file
    #df_filtered.to_csv(output_file_path, index=False)
    results_df_final = pd.DataFrame(df_filtered)
    if not results_df_final.empty:
        results_df_final = pd.concat([results_df_final, pd.DataFrame([{'Start_of_Red_Light': 'Overall', 'Total_Vehicles': results_df_final['Total_Vehicles'].sum(), 'Vehicles_Violating': results_df_final['Vehicles_Violating'].sum(), 'Compliance Rate': results_df_final['Compliance Rate'].mean()}])], ignore_index=True)
    results_df_final.to_csv(output_file_path, index=False)
        
    print(f"Processed file has been saved to: {output_file_path}")

video_files = glob(join(videos_path, '*.mp4'))

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        continue

    # Reset RoI selection variables
    drawing = False
    roi_selected = False
    top_left_pt, bottom_right_pt = None, None

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_roi)
    print("Draw a bounding box around the area of interest and press 'Enter'.")

    while True:
        frame_copy = first_frame.copy()
        if roi_selected:
            cv2.rectangle(frame_copy, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.imshow('Frame', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key is pressed
            break

    if not roi_selected:
        print(f"RoI not selected for video: {video_path}. Skipping...")
        cv2.destroyAllWindows()
        continue

    roi = (top_left_pt[0], top_left_pt[1], bottom_right_pt[0], bottom_right_pt[1])
    cv2.destroyAllWindows()
    cap.release()

    # Proceed with analysis using selected RoI
    csv_file = join(csv_path, splitext(basename(video_path))[0] + '.csv')
    analysis_results = count_vehicles_and_violations(csv_file, roi)
    output_file = join(output_path, splitext(basename(video_path))[0] + '_violations_final_test.csv')
    output_file_final = join(output_file_path, splitext(basename(video_path))[0] + '_violations.csv')
    analysis_results.to_csv(output_file, index=False)
    print(f"Analysis saved to {output_file}")
    process_csv_with_time_condition(output_file, output_file_final)





