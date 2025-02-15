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
lightLocation = np.array([0, 0])
personWidth = np.array([0, 0, 0])
movement = np.array([0, 0])
rect = np.array([0, 0, 0, 0])
maxConf = 0
lastcls = None

while True:
    sleep(0.2)
    success, img = cap.read()
    results = model(img, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes
        maxArea = 0
        # box = boxes[0]
        # print(personLocation, lightLocation, movement, maxArea)
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1 * 1.4), int(y1 * 1.4), int(x2 * 0.8), int(y2 * 0.8)
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
                    # if(personWidth[0] == persoqnWidth[1] == personWidth[2] == 0 and personLocation[0] == personLocation[1] == 0):
                    personLocation = np.array(((x1 + x2) // 2, (y1 + y2) // 2))
                    personWidth = np.array((abs(x2 - x1), abs(y2 - y1), confidence))
                
                    
            
            
            print((x1, x2, y1, y2), personLocation, lightLocation, movement, maxArea)
        
        cls = lastcls
        movement = (personLocation - lightLocation)
        lightLocation = (personLocation + lightLocation) // 2
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