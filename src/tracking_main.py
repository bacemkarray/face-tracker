# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified by Bacem Karray for personal use

import cv2
import time
from insightface.app import FaceAnalysis
from serial import Serial
import struct

from src import tracking_utils 
from src.redis_helper import RedisHelper
from src.face_memory import FaceMemory



# instantiate redis helper
redis_helper = RedisHelper(host="localhost", port=6379)
selected_object_id = None

def handle_command(message):
    global selected_object_id
    cmd_type = message.get("type")
    payload = message.get("data", {})

    if cmd_type == "track":
        selected_object_id = payload.get("target")
        print(f"🔁 Now tracking ID {selected_object_id}")

    elif cmd_type == "stop":
        target = payload.get("target")
        print(f"🛑 Stopped tracking {target}")
        if selected_object_id == target:
            selected_object_id = None

    elif cmd_type == "rename":
        old = payload.get("target")
        new = payload.get("new_name")
        print(f"✏️ Rename request: ID {old} -> {new}")
        face_memory.rename_face(old, new)

    else:
        print(f"[WARN] Unknown command: {message}")

redis_helper.subscribe("realtime_commands", handle_command)
redis_helper.start()


# config
enable_gpu = True  # Set True if running with CUDA
show_fps = True  # If True, shows current FPS in top-left corner
save_video = False  # Set True to save output video
video_output_path = "tracker_output.avi"  # Output video file name
window_name = "Tracking Window"  # Output window name
model_name = "buffalo_l" # InsightFace provided model

# setup
face_memory = FaceMemory()
s = Serial(port="COM6", baudrate=115200)


if enable_gpu:
    source=['CUDAExecutionProvider']
    ctx=0
else:
    source=['CPUExecutionProvider']
    ctx=-1


# loads face detector and recognizer 
app = FaceAnalysis(name=model_name, provider=source)
app.prepare(ctx_id=ctx, det_size=(640, 640))

cap = cv2.VideoCapture(0)

# Initialize video writer
vw = None
if save_video:
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    vw = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


selected_object_id = None
selected_bbox = None
selected_center = None
latest_frame = None
faces = []


cv2.namedWindow(window_name)

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
    
    # Send to MCU
    if center:
        packet = struct.pack('<HH', center[0], center[1])
        # send data to MCU (little endian)
        s.write(packet)


    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break        

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()


