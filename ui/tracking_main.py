# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified by Bacem Karray for personal use

import cv2
import time
from serial import Serial
import struct

from ui import tracking_utils
from oldAgent.face_memory import FaceMemory
from insightface.app import FaceAnalysis



# config
enable_gpu = True  # Set True if running with CUDA
show_fps = True  # If True, shows current FPS in top-left corner
save_video = False  # Set True to save output video
video_output_path = "tracker_output.avi"  # Output video file name
model_name = "buffalo_l"


face_memory = FaceMemory()

#
# s = Serial(port="COM6", baudrate=115200)


if enable_gpu:
    source=['CUDAExecutionProvider']
    ctx=0
else:
    source=['CPUExecutionProvider']
    ctx=-1

window_name = "Tracking Window"  # Output window name

# loads SCRFD-500MF detector + MobileFaceNet recognizer
app = FaceAnalysis(name=model_name, provider=source)
app.prepare(ctx_id=ctx, det_size=(640, 640))

cap = cv2.VideoCapture(1)

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
    pass
    # global current_task_id, selected_object_id, results
    # if event == cv2.EVENT_LBUTTONDOWN and results is not None:
    #     detections = results[0].boxes.data if results[0].boxes is not None else []
    #     if detections is not None:
    #         min_area = float("inf")
    #         best_bbox = None
    #         for track in detections:
    #             track = track.tolist()
    #             if len(track) >= 6:
    #                 x1, y1, x2, y2 = map(int, track[:4])
    #                 if x1 <= x <= x2 and y1 <= y <= y2:
    #                     area = (x2 - x1) * (y2 - y1)
    #                     if area < min_area:
    #                         min_area = area
    #                         best_bbox = (x1, y1, x2, y2)
    #         if best_bbox:
    #             x1, y1, x2, y2 = best_bbox
    #             # crop = im[y1:y2, x1:x2]
    #             matched_id = face_memory.match_or_add(im, best_bbox)
    #             if matched_id:
    #                 selected_object_id = matched_id
    #                 print(f"🔵 TRACKING STARTED: memory (ID {selected_object_id})")


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
    

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)


    center = None
    annotated_im, center = tracking_utils.process_detections(
        frame=im,
        faces=faces,
        selected_id=selected_object_id,
        memory=face_memory)
    

    # packet = struct.pack('<HH', center[0], center[1])
    # send data to MCU (little endian)
    # s.write(packet)
    # LOGGER.info(f"Sent {goal}")


    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)
    # Terminal logging
    # LOGGER.info(f"🟡 DETECTED {len(detections)} OBJECT(S): {' | '.join(detected_objects)}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()


