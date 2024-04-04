import cv2
import os
import pandas as pd
import numpy as np
from os.path import splitext, basename, join
from datetime import datetime, timedelta
from glob import glob

# Function to detect red light in the ROI
def detect_red_light(image):
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.countNonZero(mask) > 0

# Function to manually draw the ROI
def draw_roi(event, x, y, flags, param):
    global traffic_light_box, drawing, top_left_pt, bottom_right_pt, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        roi_selected = True

# Paths
videos_path = 'C:/Users/marya1/Box/MnDOT DNRTOR Project/Meenakshi/videos'
output_path = 'C:/Users/marya1/Box/MnDOT DNRTOR Project/Meenakshi/detect_red'
#videos_path = 'C:/Users/ASUS/Box/MnDOT DNRTOR Project/Pratik'
#output_path = 'C:/Users/ASUS/Box/MnDOT DNRTOR Project/Pratik/output'
#videos_path = 'E:/MnDOT/Videos'
#output_path = 'E:/MnDOT/Videos/output'


# List all video files
video_files = glob(join(videos_path, '*.mp4'))

for video_path in video_files:
    # Initialize variables for each video
    traffic_light_box = None
    roi_selected = False
    
    # Display video and select ROI
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video:", video_path)
        cap.release()
        continue  # Skip to the next video

    # Input for date and time
    video_date = input(f"Enter the video start date (yyyy-mm-dd) for {basename(video_path)}: ")
    video_start_time = input(f"Enter the video start time (hh-mm-ss) for {basename(video_path)}: ")
    start_datetime = datetime.strptime(f"{video_date} {video_start_time}", "%Y-%m-%d %H-%M-%S")
    
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_roi)

    print("Draw a bounding box around the traffic light and press 'Enter'.")
    while True:
        frame_copy = first_frame.copy()
        if roi_selected:
            cv2.rectangle(frame_copy, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.imshow('Frame', frame_copy)
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            traffic_light_box = (top_left_pt[0], top_left_pt[1], bottom_right_pt[0] - top_left_pt[0], bottom_right_pt[1] - top_left_pt[1])
            break

    cv2.destroyAllWindows()

    df = pd.DataFrame(columns=['Frame_Number', 'Date', 'Time', 'Red_Light'])
    frame_counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = start_datetime + timedelta(seconds=frame_counter / fps)
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S")

        traffic_light_roi = frame[traffic_light_box[1]:traffic_light_box[1]+traffic_light_box[3], 
                                  traffic_light_box[0]:traffic_light_box[0]+traffic_light_box[2]]
        red_light_detected = detect_red_light(traffic_light_roi)

        new_row = {'Frame_Number': frame_counter, 'Date': date_str, 'Time': time_str, 'Red_Light': red_light_detected}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        frame_counter += 1

    cap.release()

    # Save the DataFrame to a CSV file
    csv_file_name = splitext(basename(video_path))[0] + '.csv'
    csv_path = join(output_path, csv_file_name)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved at {csv_path}")
