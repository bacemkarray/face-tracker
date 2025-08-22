# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modified by Bacem Karray for personal use

import cv2
import time
from insightface.app import FaceAnalysis
from serial import Serial
import struct

from ui import tracking_utils 
from ui.redis_helper import RedisHelper


from oldAgent.face_memory import FaceMemory


face_memory = FaceMemory()

# user_input = input("Give a command that you would like to run: ")
# command = graph.invoke({"instructions": user_input}) # currently outputs a task to do
# task_executor.add_task(command['task'])



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
        print(f"✏️ Rename request: ID {old} → {new}")
        # later you can wire this into face_memory.rename_face

    else:
        print(f"[WARN] Unknown command: {message}")


redis_helper.subscribe("realtime_commands", handle_command)
redis_helper.start()


#
# s = Serial(port="COM6", baudrate=115200)

# config
enable_gpu = True  # Set True if running with CUDA
show_fps = True  # If True, shows current FPS in top-left corner
save_video = False  # Set True to save output video
video_output_path = "tracker_output.avi"  # Output video file name
model_name = "buffalo_l"


face_memory = FaceMemory()

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
    # global faces, selected_object_id
    # if event == cv2.EVENT_LBUTTONDOWN and faces:
    #         if not faces:
    #             return

    #         min_area = float("inf")
    #         best_face = None
    #         for face in faces:
    #             x1, y1, x2, y2 = map(int, face.bbox)
    #             if x1 <= x <= x2 and y1 <= y <= y2:
    #                 area = (x2 - x1) * (y2 - y1)
    #                 if area < min_area:
    #                     min_area = area
    #                     best_face = face
            
    #         if best_face:
    #             matched_id = face_memory.match_or_add(best_face.embedding)
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
    
    # Send to MCU
    # packet = struct.pack('<HH', center[0], center[1])
    # send data to MCU (little endian)
    # s.write(packet)


    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("b"):
        # face_memory.rename_face("???:1", "Bacem")
        redis_helper.publish("realtime_commands", {"data": "???:1"})
        

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()


