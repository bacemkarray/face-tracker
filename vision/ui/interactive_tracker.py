
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified by Bacem Karray for personal use

import cv2
import time
from serial import Serial
import struct

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from vision.ui import tracking_utils 
from vision.oldAgent.task_executor import TaskExecutor

from vision.oldAgent.face_memory import FaceMemory


from vision.agent import graph

task_executor = TaskExecutor()
face_memory = FaceMemory()

user_input = input("Give a command that you would like to run: ")
command = graph.invoke({"instructions": user_input}) # currently outputs a task to do
task_executor.add_task(command['task'])
current_task_id = 1

# for face ids
previous_ids = {}

#
s = Serial(port="COM6", baudrate=115200)

# config
enable_gpu = True  # Set True if running with CUDA
model_file = "vision/yolov11l-face.pt"  # Path to model file
show_fps = True  # If True, shows current FPS in top-left corner
show_conf = False  # Display or hide the confidence score
save_video = False  # Set True to save output video
video_output_path = "interactive_tracker_output.avi"  # Output video file name


conf = 0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
iou = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
max_det = 5  # Maximum objects per image (increase for crowded scenes)

tracker = "bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
track_args = {
    "persist": True,  # Keep frames history as a stream for continuous tracking
    "verbose": False,  # Print debug info from tracker
}

window_name = "Ultralytics YOLO Interactive Tracking"  # Output window name

LOGGER.info("Initializing model...")
if enable_gpu:
    LOGGER.info("Using GPU...")
    model = YOLO(model_file)
    model.to("cuda")
else:
    LOGGER.info("Using CPU...")
    model = YOLO(model_file, task="detect")

classes = model.names  # Store model class names

cap = cv2.VideoCapture(0)  # Replace with video path if needed

# Initialize video writer
vw = None
if save_video:
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    vw = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

selected_object_id = None
selected_bbox = None
selected_center = None
results = None
latest_frame = None


# move this soon
def click_event(event: int, x: int, y: int, flags: int, param) -> None:
    """
    Handle mouse click events to select an object for focused tracking.

    Args:
        event (int): OpenCV mouse event type.
        x (int): X-coordinate of the mouse event.
        y (int): Y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV.
        param (Any): Additional parameters (not used).
    """
    global current_task_id, selected_object_id, results
    if event == cv2.EVENT_LBUTTONDOWN and results is not None:
        detections = results[0].boxes.data if results[0].boxes is not None else []
        if detections is not None:
            min_area = float("inf")
            best_bbox = None
            for track in detections:
                track = track.tolist()
                if len(track) >= 6:
                    x1, y1, x2, y2 = map(int, track[:4])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < min_area:
                            min_area = area
                            best_bbox = (x1, y1, x2, y2)
            if best_bbox:
                x1, y1, x2, y2 = best_bbox
                # crop = im[y1:y2, x1:x2]
                matched_id = face_memory.match_or_add(im, best_bbox)
                if matched_id:
                    selected_object_id = matched_id
                    print(f"🔵 TRACKING STARTED: memory (ID {selected_object_id})")
                current_task_id = task_executor.add_task({"task": "track"})


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

fps_counter, fps_timer, fps_display = 0, time.time(), 0

while cap.isOpened():
    start = time.time()
    success, im = cap.read()
    if not success:
        break
    
    if show_fps:
        fps_counter, fps_display, fps_timer = tracking_utils.show_fps(
            im, 
            fps_counter, 
            fps_display,
            fps_timer)
    
    results = model.track(im, conf=conf, iou=iou, max_det=max_det, tracker=tracker, **track_args)
    detections = results[0].boxes.data if results[0].boxes is not None else []
    center = None
    annotated_im, center = tracking_utils.process_detections(
        frame=im,
        detections=detections,
        selected_id=selected_object_id,
        face_memory=face_memory, 
        previous_ids=previous_ids)


    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)
    # Terminal logging
    # LOGGER.info(f"🟡 DETECTED {len(detections)} OBJECT(S): {' | '.join(detected_objects)}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    if current_task_id:
        goal = task_executor.step(center)
        packet = struct.pack('<BHH', current_task_id, goal[0], goal[1])
        # send data to MCU (little endian)
        s.write(packet)
        # LOGGER.info(f"Sent {goal}")

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()


