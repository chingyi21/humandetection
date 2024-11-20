from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from persondetection import DetectorAPI
from flask_cors import CORS
from PIL import Image
import random

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Folder setup
UPLOAD_FOLDER = './static/uploads'
OUTPUT_FOLDER = './static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize the DetectorAPI (ensure processFrame is implemented)
odapi = DetectorAPI(path_to_ckpt='./models/frozen_inference_graph.pb')
THRESHOLD = 0.7  # Detection threshold

# Constants for filtering and human size
MIN_HEIGHT = 150
MAX_HEIGHT = 300
MAX_DISTANCE = 100
MAX_HUMAN_LIMIT = 25

# Global variables for camera functionality
camera = cv2.VideoCapture(0)
detection_data = []

# Global Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# -------------------- ROUTES -------------------- #
# Route: Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Selection Screen
@app.route('/selection')
def selection():
    return render_template('selection.html')

# Route: Detect from Camera
@app.route('/detect_camera')
def detect_camera():
    return render_template('detect_camera.html')

# Route: Detect Video Page
@app.route('/detect_video')
def detect_video():
    filename = request.args.get('filename', None)  # Get filename from query string
    return render_template('detect_video.html', filename=filename)


# -------------------- DETECT CAMERA FUNCTIONALITY -------------------- #

# Function: IoU computation
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function: Generate frames for live feed with live count updates
def generate_camera_frames():
    global detection_data
    total_count = 0
    leaving_count = 0
    tracked_people = {}  # {ID: (current_box, last_direction)}
    next_id = 1
    iou_threshold = 0.5  # IoU threshold for matching
    direction_threshold = 20  # Minimum pixel movement to detect a direction change

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (800, 600))
        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                if MIN_HEIGHT <= h <= MAX_HEIGHT:  # Filter by height
                    detected_boxes.append([x, y, x + w, y + h])

        # Tracking logic
        new_tracked_people = {}
        for box in detected_boxes:
            matched = False
            for pid, (prev_box, last_direction) in tracked_people.items():
                if iou(box, prev_box) > iou_threshold:  # Match box using IoU
                    # Calculate direction
                    current_direction = box[0] - prev_box[0]  # Vertical movement
                    if abs(current_direction) > direction_threshold:
                        if current_direction * last_direction < 0:  # Direction changed
                            leaving_count += 1
                    new_tracked_people[pid] = (box, current_direction)
                    matched = True
                    break

            if not matched:
                # Assign a new ID if no match found
                new_tracked_people[next_id] = (box, 0)
                next_id += 1

        # Update tracked people
        tracked_people = new_tracked_people

        # Update counts
        current_count = len(tracked_people)
        total_count = current_count + leaving_count
        
        # Record the start time
        start_time = time.time() 

        # Record time and data
        elapsed_time = round(time.time() - start_time, 2)  # Time in seconds
        detection_data.append({
            'time': elapsed_time,
            'count': current_count,
            'accuracy': 1.0,  # Placeholder for accuracy
        })


        # Draw bounding boxes and text
        for box, _ in tracked_people.values():
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display stats
        cv2.putText(frame, f"Current Count: {current_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Total Count: {total_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Leaving: {leaving_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




# Route: Video feed for camera detection
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------- DETECT VIDEO FUNCTIONALITY -------------------- #
# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Helper function: Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# #Function to generate video frames 
# def generate_video_frames(video_path):
#     print(f"Opening video: {video_path}")
#     video = cv2.VideoCapture(video_path)
    
#     if not video.isOpened():
#         print(f"Error: Cannot open video file {video_path}")
#         return

#     total_people_count = 0  # Initialize total people count

#     while video.isOpened():
#         success, frame = video.read()
#         if not success:
#             print("End of video or error reading frame.")
#             break

#         # Resize the frame for consistent processing
#         frame = cv2.resize(frame, (800, 600))
#         print("Processing a new frame...")

#         # Perform person detection
#         boxes, scores, classes, num = odapi.processFrame(frame)
#         print(f"Boxes: {boxes}, Scores: {scores}, Classes: {classes}, Num: {num}")

#         # Filter detected objects based on confidence score and class
#         current_people_count = 0
#         for i in range(len(boxes)):
#             if classes[i] == 1 and scores[i] > THRESHOLD:  # Class 1 for 'person'
#                 box = boxes[i]
#                 x_min, y_min, x_max, y_max = box
#                 box_width = x_max - x_min
#                 box_height = y_max - y_min

#                 # Filter out overly large boxes (likely false positives)
#                 if box_width < 500 and box_height < 500:
#                     current_people_count += 1
#                     total_people_count += 1

#                     # Draw bounding box
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

#         # Add text overlays to display counts
#         cv2.putText(frame, f"Current Count: {current_people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(frame, f"Total People: {total_people_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Save a debug frame (optional)
#         cv2.imwrite("debug_frame.jpg", frame)

#         # Encode the frame as JPEG for streaming
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         # Yield the frame for live streaming
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     video.release()
#     print("Video processing completed.")




# Route: Upload and Process Video
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file uploaded'})

    video_file = request.files['video']

    if not allowed_file(video_file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type. Allowed: mp4, avi, mov'})

    # Save the uploaded video file
    filename = f"{int(time.time())}_{video_file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)

    # Redirect to video feed page with the filename
    return redirect(url_for('detect_video', filename=filename))

# Route: Serve Video Feed
@app.route('/uploaded_video_feed/<filename>')
def uploaded_video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Streaming video from: {video_path}")

    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return "Error: File not found.", 404

    return Response(
        generate_video_frames(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# Variable to track total people
total_people_count = 0


# Function that makes the video play
def generate_video_frames(video_path):
    print(f"Opening video: {video_path}")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    while video.isOpened():
        success, frame = video.read()
        if not success:
            print("End of video or error reading frame.")
            break

        print("Processing frame...")
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    print("Video processing completed.")


# -------------------- Plotting -------------------- #

detection_data = [] 

# Route: Enumeration Plot
@app.route('/enumeration_plot')
def enumeration_plot():
    global detection_data
    if not detection_data:
        return "No data available to generate plot."

    times = [data['time'] for data in detection_data]
    counts = [data['count'] for data in detection_data]

    plt.figure(figsize=(8, 6))
    plt.plot(times, counts, marker='o', color='blue', label='Human Count')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Human Count")
    plt.title("Enumeration Plot")
    plt.legend()
    path = os.path.join(OUTPUT_FOLDER, 'enumeration_plot.png')
    plt.savefig(path)
    plt.close()
    return send_from_directory(OUTPUT_FOLDER, 'enumeration_plot.png')

# Route: Accuracy Plot
@app.route('/accuracy_plot')
def accuracy_plot():
    global detection_data
    if not detection_data:
        return "No data available to generate plot."

    times = [data['time'] for data in detection_data]
    accuracies = [data['accuracy'] for data in detection_data]

    plt.figure(figsize=(8, 6))
    plt.plot(times, accuracies, marker='o', color='green', label='Accuracy')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy Plot")
    plt.legend()
    path = os.path.join(OUTPUT_FOLDER, 'accuracy_plot.png')
    plt.savefig(path)
    plt.close()
    return send_from_directory(OUTPUT_FOLDER, 'accuracy_plot.png')

# Route: Generate Crowd Report
@app.route('/generate_report')
def generate_report():
    global detection_data
    if not detection_data:
        return "No data available to generate report."

    max_count = max(data['count'] for data in detection_data)
    avg_accuracy = sum(data['accuracy'] for data in detection_data) / len(detection_data)
    status = "Crowded" if max_count > MAX_HUMAN_LIMIT else "Not Crowded"

    report = f"""
    CROWD REPORT:
    Max Human Limit: {MAX_HUMAN_LIMIT}
    Max Human Count: {max_count}
    Avg. Accuracy: {avg_accuracy:.2f}
    Status: {status}
    """
    path = os.path.join(OUTPUT_FOLDER, 'crowd_report.txt')
    with open(path, 'w') as file:
        file.write(report)
    return send_from_directory(OUTPUT_FOLDER, 'crowd_report.txt')


# -------------------- MAIN FUNCTION -------------------- #
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
