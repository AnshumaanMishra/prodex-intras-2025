import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

GPIO.setup(11, GPIO.OUT)
servo1 = GPIO.PWM(11, 50)

servo1.start(0)
print("Wait: 2s")
time.sleep(2)

print("Rotating in 10 steps:")

duty = 2
'''
while duty <= 12:
    servo1.ChangeDutyCycle(duty)
    time.sleep(5)
    servo1.ChangeDutyCycle(0)
    print(f"step {duty} done")
    time.sleep(5)
    duty = duty + 1

time.sleep(2)
'''
print("90 degrees")
def SetAngle(angle):
    servo1.ChangeDutyCycle(2)
    time.sleep(10)
    print(7)
    #servo1.ChangeDutyCycle(2)
    #time.sleep(1)
    #print(0)
    #servo1.ChangeDutyCycle(0)
    #time.sleep(1)

# servo1.ChangeDutyCycle(5)
# time.sleep(2)
#servo1.ChangeDutyCycle(2)
#time.sleep(0.35)
#servo1.ChangeDutyCycle(0)

SetAngle(10)

servo1.stop()
GPIO.cleanup()
