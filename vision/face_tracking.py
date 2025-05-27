import mediapipe as mp
import numpy as np
import cv2 as cv
from visualization_utils import visualize
import time


# Create a FaceDetector object.
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='vision/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO)

detector = FaceDetector.create_from_options(options)

cap = cv.VideoCapture(0)

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

    frame_timestamp_ms = int(time.time()*1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    cv.imshow('Camera', annotated_image)

    if cv.waitKey(1) == ord('q'):
        break
 

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()