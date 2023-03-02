import cv2
import mediapipe as mp
import time

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        #Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #process the image and find faces
        results = face_detection.process(image)
        #Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections: 
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection)

            end = time.time()
            totalTime = end - start
            fps = 1/ totalTime
            print("FPS: ", fps)
            cv2.putText(image, f'FPS: {int(fps)}')
        