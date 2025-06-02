import mediapipe as mp
import numpy as np
import cv2 as cv
import time
import serial
import struct
from visualization_utils import visualize
from noise_correction import SlidingAverage
from controls import PDController

s = serial.Serial(port="COM3",
                  baudrate=115200)

# Create a FaceDetector object.
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='vision/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)
detector = FaceDetector.create_from_options(options)

cap = cv.VideoCapture(1)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) # width of frame
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) # height of frame

frame_cx = width//2 # center of frame on x-axis

frame_cy = height//2 # center of frame on y-axis

controller = PDController(frame_cx, frame_cy)

y_smoother = SlidingAverage(window_size=5)
x_smoother = SlidingAverage(window_size=5)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    frame_timestamp_ms = int(time.time()*1000)
    
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Call the visualize function to annotate the image. Also obtain the (x,y) of the face's center point.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image, face_pos = visualize(image_copy, detection_result)
    cv.imshow('Camera', annotated_image)

    if face_pos:
        face_x, face_y = face_pos
        current_x, current_y = controller.update(face_x, face_y, x_smoother, y_smoother)
                
        # send data to MCU
        packet = struct.pack('<HH', int(current_x), int(current_y))
        s.write(packet)
    
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the captuqre
cap.release()
cv.destroyAllWindows()