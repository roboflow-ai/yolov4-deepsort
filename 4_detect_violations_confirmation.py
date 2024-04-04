import cv2
import pandas as pd
from os.path import join, basename, splitext
from glob import glob

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


def count_vehicles_and_violations(csv_file, roi, valid_classes=[1, 2, 3, 5, 7], min_frames=60):
    df = pd.read_csv(csv_file)
    # Assuming 'Date' column is in 'YYYY-MM-DD' format and 'Time' in 'HH:MM:SS' format
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    red_light_phases = df[df['Color'] == 'Red'].groupby((df['Color'] != df['Color'].shift()).cumsum())
    results = []

    for _, phase in red_light_phases:
        start_of_red_light = phase['DateTime'].iloc[0]
        next_phase_start = phase['DateTime'].iloc[-1] + pd.Timedelta(seconds=1)
        
        # Skip intervals less than 1 second
        if (next_phase_start - start_of_red_light).total_seconds() <= 1:
            continue

        tracks_in_roi = df[(df['DateTime'] >= start_of_red_light) & (df['DateTime'] < next_phase_start)]
        tracks_in_roi = tracks_in_roi[tracks_in_roi.apply(lambda row: is_within_roi(row['xmin'], row['ymin'], row['xmax'], row['ymax'], roi), axis=1)]
        valid_tracks_in_roi = tracks_in_roi[tracks_in_roi['Class_ID'].isin(valid_classes)]

        # Count frames for each track and filter out those with less than min_frames
        frame_counts = valid_tracks_in_roi.groupby('Track').size()
        tracks_more_than_min_frames = frame_counts[frame_counts >= min_frames].index
        valid_tracks_in_roi = valid_tracks_in_roi[valid_tracks_in_roi['Track'].isin(tracks_more_than_min_frames)]

        exiting_tracks = valid_tracks_in_roi.groupby('Track').last()
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
        overall_compliance_rate = (results_df['Total_Vehicles'].sum() - results_df['Vehicles_Violating'].sum()) / results_df['Total_Vehicles'].sum()
        overall_summary = pd.DataFrame([{'Start_of_Red_Light': 'Overall', 'Beginning_of_Not_Red': '', 'Total_Vehicles': results_df['Total_Vehicles'].sum(), 'Vehicles_Violating': results_df['Vehicles_Violating'].sum(), 'Compliance Rate': overall_compliance_rate}])
        results_df = pd.concat([results_df, overall_summary], ignore_index=True)

    return results_df

csv_path = '/home/marya1/Documents/MnDoTNRToR/inference/final_csv'
output_path='/home/marya1/Documents/MnDoTNRToR/inference/detect_violations_trial'
videos_path = '/home/marya1/Documents/MnDoTNRToR/videos/processing'

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
    output_file = join(output_path, splitext(basename(video_path))[0] + '_violations.csv')
    analysis_results.to_csv(output_file, index=False)
    print(f"Analysis saved to {output_file}")