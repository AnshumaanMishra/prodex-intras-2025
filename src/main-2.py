from ultralytics import YOLO
import cv2
import math 
import pygame
import numpy as np

# Uncomment this line to use the GPU for YOLO
'''
import torch
torch.cuda.set_device(0)
cap = cv2.VideoCapture(0)
'''

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/dev/video1")
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person"]

# Container Variables
coordinateSet = np.array([0, 0, 0, 0])
personLocation = np.array([0, 0])
lightLocation = np.array([30, 30])
personWidth = np.array([0, 0])
preverror = 0
interror = 0
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
        print("Circles, ", circles)
        # org = [rect[0], rect[1]]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale = 1
        # color = (255, 0, 0)
        # thickness = 2
        # cv2.circle(output_frame, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
        # cv2.putText(output_frame, str(f"{np.array([circles[0, 0], circles[0, 1]])}"), org, font, fontScale, color, thickness)
        return circles
    return np.array([[0, 0, 0]])


# Helper Functions
def getCoordinatesAndArea(box, startFactor, endFactor) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.xyxy[0]
    cset = np.array([[int(x1 * startFactor), int(y1 * startFactor)], [int(x2 * endFactor), int(y2 * endFactor)]])
    return cset, abs(x2 - x1) * abs(y2 - y1)

def personLocationAndWidth(coordinateSet):
    personLocation = (coordinateSet[0] + coordinateSet[1]) // 2
    personWidth = abs(coordinateSet[1] - coordinateSet[0])
    return personLocation, personWidth

def drawShapes(img, coordinateSet, circles):
    cv2.rectangle(img, coordinateSet[0], coordinateSet[1], (255, 0, 255), 3)
    # cv2.circle(img, lightLocation, radius=10, color=(0, 0, 255), thickness=-1)
    org = [coordinateSet[0][0], coordinateSet[0][1]]
    org2 = [circles[0, 0], circles[0, 1]]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(img, str(f"{personLocation}"), org, font, fontScale, color, thickness)
    cv2.circle(img, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
    cv2.putText(img, str(f"{np.array([circles[0, 0], circles[0, 1]])}"), org2, font, fontScale, color, thickness)

def calculateError(setpoint: np.ndarray[int, int], current=np.array([0, 0])) -> float:
    # error = (setpoint - current) ** 2
    # return sum(error) ** 0.5
    return setpoint - current

def calculateProportionality(error, proportionality) -> float:
    return proportionality * error

def calculateIntegral(error, currentIntegralError, frameRate, proportionality) -> tuple[float, float]:

    integral = currentIntegralError + error * (1 / frameRate)
    return (integral, proportionality * integral)

def calculateDifferential(error, prevError, frameRate, proportionality) -> float:
    time = 1 / frameRate
    differential = (error - prevError) / time
    return differential * proportionality



fps = 30
clock1 = pygame.time.Clock()
# Main Loop
while True:
    success, img = cap.read()
    clock1.tick(fps)
    results = model(img, stream=True)
    lightLocation1 = detectRedDot()
    print("ll1, ", lightLocation1)
    for r in results:
        boxes = r.boxes
        maxArea = 0
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]
            coordinateSet, currentArea = getCoordinatesAndArea(box, 1.5, 0.9)
            if(currentArea < maxArea):
                continue
            else:
                maxArea = currentArea
                confidence = math.ceil((box.conf[0]*100))/100

                print("Confidence --->",confidence)

                cls = int(box.cls[0])
                lastcls = cls
                if(cls == 0):
                    personLocation, personWidth = personLocationAndWidth(coordinateSet)
        
        cls = lastcls
        if(cls == 0):
            if(sum(sum(lightLocation1)) != 0):
                lightLocation = np.array([lightLocation1[0, 0], lightLocation1[0, 1]])
                error = calculateError(current=lightLocation, setpoint=personLocation)
                print(lightLocation, personLocation)
                properror = calculateProportionality(error, 15e-2)
                interror, currentInt = calculateIntegral(error, interror, fps, 42e-3)
                differror = calculateDifferential(error, preverror, fps, 15e-1)
                preverror = error
                print(error, properror, interror, currentInt, differror)
                # print(lightLocation)
                lightLocation += np.int64(properror + currentInt + differror)
                print("movement= ", np.int64(properror + currentInt + differror))

            # movement = (personLocation - lightLocation)
            print(lightLocation, personLocation)
            # lightLocation += movement // 25
            print("Class name -->", classNames[cls])
            drawShapes(img, coordinateSet, lightLocation1)
            

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
