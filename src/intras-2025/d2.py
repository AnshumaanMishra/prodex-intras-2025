from time import sleep
from ultralytics import YOLO
import cv2
import math 
import torch
import numpy as np

torch.cuda.set_device(0)
# start webcam
# cap = cv2.VideoCapture("http://192.168.4.193:4747/video")
cap = cv2.VideoCapture("/dev/video0")
# cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person"]


personLocation = np.array([0, 0])
lightLocation = np.array([30, 30])
personWidth = np.array([0, 0, 0])
movement = np.array([0, 0])
rect = np.array([0, 0, 0, 0])
maxConf = 0
lastcls = 0

def detectRedDot():
    ret, captured_frame = cap.read()
    output_frame = captured_frame.copy()

    # Convert original image to BGR, since Lab is only available from BGR
    # cv2.circle(captured_frame, (300, 30), radius=10, color=(0, 0, 255), thickness=10)
    captured_frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2BGR)
    # First blur to reduce noise prior to color space conversion
    captured_frame_bgr = cv2.medianBlur(captured_frame_bgr, 3)
    # Convert to Lab color space, we only need to check one channel (a-channel) for red here
    captured_frame_lab = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2Lab)
    # Threshold the Lab image, keep only the red pixels
    # Possible yellow threshold: [20, 110, 170][255, 140, 215]
    # Possible blue threshold: [20, 115, 70][255, 145, 120]
    captured_frame_lab_red = cv2.inRange(captured_frame_lab, np.array([20, 150, 150]), np.array([190, 255, 255]))
    # Second blur to reduce more noise, easier circle detection
    captured_frame_lab_red = cv2.GaussianBlur(captured_frame_lab_red, (5, 5), 2, 2)
    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(captured_frame_lab_red, cv2.HOUGH_GRADIENT, 1, captured_frame_lab_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)

	# If we have extracted a circle, draw an outline
	# We only need to detect one circle here, since there will only be one reference object
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        org = [rect[0], rect[1]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(output_frame, str(f"{np.array([circles[0, 0], circles[0, 1]])}"), org, font, fontScale, color, thickness)
        cv2.circle(output_frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
        return {"center": np.array([circles[0, 0], circles[0, 1]]), "radius": circles[0, 2]}
    return None



while True:
    # sleep(0.2)
    success, img = cap.read()
    results = model(img, stream=True)
    # coordinates
    # cv2.circle(img, lightLocation, radius=10, color=(0, 0, 255), thickness=-1)
    for r in results:
        boxes = r.boxes
        maxArea = 0
        # box = boxes[0]
        # print(personLocation, lightLocation, movement, maxArea)
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1 * 1.5), int(y1 * 1.5), int(x2 * 0.9), int(y2 * 0.9)
            if(abs(x2 - x1) * abs(y2 - y1) < maxArea):
                continue
            else:
                maxArea = abs(x2 - x1) * abs(y2 - y1)
            # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                lastcls = cls
                rect = np.array([x1, y1, x2, y2])
                if(cls == 0):
                    personLocation = np.array(((x1 + x2) // 2, (y1 + y2) // 2))
                    personWidth = np.array((abs(x2 - x1), abs(y2 - y1), confidence))
            
            print((x1, x2, y1, y2), personLocation, lightLocation, movement, maxArea)
        
        cls = lastcls
        if(cls == 0):
            movement = (personLocation - lightLocation)
            # x = detectRedDot()
            # print(x)
            # lightLocation = x['center'] if x is not None else (30, 30)
            # lightLocation = (personLocation + lightLocation) // 2
            print(lightLocation, personLocation)
            lightLocation += movement // 25
            print("Class name -->", classNames[cls])
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 3)
            cv2.circle(img, lightLocation, radius=10, color=(0, 0, 255), thickness=-1)
            # object details
            org = [rect[0], rect[1]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls] + str(f"{personLocation, movement}"), org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()