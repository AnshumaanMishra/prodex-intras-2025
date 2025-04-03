# import time
import pygame
import cv2  # type: ignore
import numpy as np # type: ignore

clock1 = pygame.time.Clock()
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

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

success, img = cap.read()
cv2.imshow('Webcam', img)
current = np.array([0, 0], dtype=np.int64)
setpoint = np.array([100, 100], dtype=np.int64)
preverror = 0
interror = 0
cv2.circle(img, current, radius=10, color=(0, 0, 255), thickness=-1)

count = 0
sum = 0
max = 0
min = 1000000000

# t1 = time.time()
# time.sleep(2)
for i in range(100):
    clock1.tick(30)
    count += 1
    success, img = cap.read()
    cv2.imshow('Webcam', img)
    # print((int(current[0]), int(current[1])))
    # cv2.circle(img, center=(int(current[0]), int(current[1])), radius=10, color=(0, 0, 255), thickness=-1)
    cv2.circle(img, center=(200, 200), radius=10, color=(0, 0, 255), thickness=-1)
    # t2 = time.time()
    # print(t1, t2, (t2 - t1))
    error = calculateError(current=current, setpoint=setpoint)
    # fps = 1 / (t2 - t1)
    fps = 30
    # t1 = t2
    properror = calculateProportionality(error, 15e-3)
    # properror = calculateProportionality(error, 1 * 1e-10)
    interror, currentInt = calculateIntegral(error, interror, fps, 42)
    # interror, currentInt = calculateIntegral(error, interror, fps, 115)
    # differror = calculateDifferential(error, preverror, fps, 0.95 * 1e-3)
    differror = calculateDifferential(error, preverror, fps, 15e-3)
    # print(f"Error: {error}, prop: {properror}, int: {interror, currentInt}, diff: {differror}")
    preverror = error

    current += np.int64(properror + currentInt + differror)
    sum += current[0]
    if(min > current[0]):
        min = current[0]

    if(max < current[0]):
        max = current[0]
    # current[1] += int(properror + interror + differror)

    # if cv2.waitKey(1) == ord('q'):
    #     break

print(f"Average = {sum / count}, min = {min}, max = {max}")
cap.release()
cv2.destroyAllWindows()
