import R64.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)

GPIO.setup(3, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)

while True:
    GPIO.output(3, GPIO.LOW)    #茶
    GPIO.output(5, GPIO.LOW)    #赤
    GPIO.output(15, GPIO.HIGH)   #橙
    GPIO.output(16, GPIO.HIGH)   #黄
    sleep(0.02)

    GPIO.output(3, GPIO.HIGH)
    GPIO.output(5, GPIO.LOW)
    GPIO.output(15, GPIO.LOW)
    GPIO.output(16, GPIO.HIGH)
    sleep(0.02)

    GPIO.output(3, GPIO.HIGH)
    GPIO.output(5, GPIO.HIGH)
    GPIO.output(15, GPIO.LOW)
    GPIO.output(16, GPIO.LOW)
    sleep(0.02)

    GPIO.output(3, GPIO.LOW)
    GPIO.output(5, GPIO.HIGH)
    GPIO.output(15, GPIO.HIGH)
    GPIO.output(16, GPIO.LOW)
    sleep(0.02)
