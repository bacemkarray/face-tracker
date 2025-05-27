import mediapipe as mp
import numpy as np
import cv2 as cv
import time
from visualization_utils import visualize
import serial

# Configure serial communication
s = serial.Serial(port='COM3',
                  baudrate=9600)

# Create a FaceDetector object.
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='vision/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)
detector = FaceDetector.create_from_options(options)

# Default servo angles are set to 90 degrees.
prevX = 90
prevY = 90

cap = cv.VideoCapture(0)
ws = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
hs = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

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

        # Takes the face_pos coordinates, and assigns a servo angle based on their position relative to the image frame
        servoX = 180-np.interp(face_pos[0], [0, ws], [0, 180])
        servoY = 180-np.interp(face_pos[1], [0, hs], [0, 180])

        # Simple exponential smoothing
        alpha = 0.2
        smoothedX = int(alpha * servoX + (1 - alpha) * prevX)
        smoothedY = int(alpha * servoY + (1 - alpha) * prevY)
        prevX = smoothedX
        prevY = smoothedY
        
        # send data to arduino
        s.write([smoothedX,smoothedY])

    # if face_pos[0] is greater than center of screen, move servo-1 at an angle that reduces this value. while face_pos greater than center, 
    # find out window height and width
    

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()