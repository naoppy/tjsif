# coding=utf-8
# ステップ角1.8度
# 定格電圧2.2 定格電流1.6 内部抵抗1.35
import R64.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)  # ピン番号でピンを指定

GPIO.setup(3, GPIO.OUT)  # ピンを出力に設定
GPIO.setup(5, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)


def right_spin_7_2degree():
    GPIO.output(3, GPIO.LOW)  # 茶(モータの線の色)
    GPIO.output(5, GPIO.LOW)  # 赤
    GPIO.output(15, GPIO.HIGH)  # 橙
    GPIO.output(16, GPIO.HIGH)  # 黄
    sleep(0.02)  # 脱調を防ぐため

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


def left_spin_7_2degree():
    GPIO.output(3, GPIO.LOW)  # 茶(モータの線の色)
    GPIO.output(5, GPIO.LOW)  # 赤
    GPIO.output(15, GPIO.HIGH)  # 橙
    GPIO.output(16, GPIO.HIGH)  # 黄
    sleep(0.02)

    GPIO.output(3, GPIO.LOW)
    GPIO.output(5, GPIO.HIGH)
    GPIO.output(15, GPIO.HIGH)
    GPIO.output(16, GPIO.LOW)
    sleep(0.02)

    GPIO.output(3, GPIO.HIGH)
    GPIO.output(5, GPIO.HIGH)
    GPIO.output(15, GPIO.LOW)
    GPIO.output(16, GPIO.LOW)
    sleep(0.02)

    GPIO.output(3, GPIO.HIGH)
    GPIO.output(5, GPIO.LOW)
    GPIO.output(15, GPIO.LOW)
    GPIO.output(16, GPIO.HIGH)
    sleep(0.02)


# 回転角度=ステップ角*指令パルス数　
# 時計回りに360度回転
def right_spin_360degree():
    for _ in range(0, 50):
        right_spin_7_2degree()


# 反時計回りに360度回転
def left_spin_360degree():
    for _ in range(0, 50):
        left_spin_7_2degree()
