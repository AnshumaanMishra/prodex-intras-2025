import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

servo1 = GPIO.PWM(11, 50)  # 50Hz for standard servos
servo1.start(0)

def SetAngle(angle):
    duty = angle / 18 + 2  # maps 0-180° to ~2-12% duty
    minT = 0.37
    if(angle < 0):
        duty = 12
        time1 = (minT / 90) * -angle
    else:
        duty = 2
        time1 = (minT / 90) * angle
    servo1.ChangeDutyCycle(duty)
    time.sleep(time1)
    servo1.ChangeDutyCycle(0)  # optional: stop signal to reduce jitter

# Example: rotate to 10°
angle = int(input())
SetAngle(angle)

servo1.stop()
GPIO.cleanup()


