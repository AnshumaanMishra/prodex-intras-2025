from ultralytics import YOLO
import cv2
import math 
import torch

torch.cuda.set_device(0)
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    personLocation = (0, 0)
    personWidth = (0, 0, 0)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            if(cls == 0 and ((personWidth[0] < abs(x2 - x1) or personWidth[1] < abs(y2 - y1)) and confidence > personWidth[2])):
                print("Class name -->", classNames[cls])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                personLocation = ((x1 + x2) / 2, (y1 + y2) / 2)
                personWidth = (abs(x2 - x1), abs(y2 - y1), confidence)
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls] + str(f"{personLocation}"), org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()