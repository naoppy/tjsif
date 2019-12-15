# coding=utf-8
# ステップ角1.8度
# 定格電圧2.2 定格電流1.6 内部抵抗1.35
import RPi.GPIO as GPIO
from time import sleep
import atexit

GPIO.setmode(GPIO.BOARD)  # ピン番号でピンを指定

out_channels = [11, 12, 13, 15]

GPIO.setup(out_channels, GPIO.OUT)  # ピンを出力に設定


def __outputs(channels, settings):
    for ch, value in zip(channels, settings):
        GPIO.output(ch, value)


def right_spin_7_2degree():
    # 茶, 赤, 橙, 黄
    __outputs(out_channels, [GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.HIGH])
    sleep(0.02)  # 脱調を防ぐため

    __outputs(out_channels, [GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH])
    sleep(0.02)

    __outputs(out_channels, [GPIO.HIGH, GPIO.HIGH, GPIO.LOW, GPIO.LOW])
    sleep(0.02)

    __outputs(out_channels, [GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW])
    sleep(0.02)


def left_spin_7_2degree():
    # 茶, 赤, 橙, 黄
    __outputs(out_channels, [GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.HIGH])
    sleep(0.02)

    __outputs(out_channels, [GPIO.LOW, GPIO.HIGH, GPIO.HIGH, GPIO.LOW])
    sleep(0.02)

    __outputs(out_channels, [GPIO.HIGH, GPIO.HIGH, GPIO.LOW, GPIO.LOW])
    sleep(0.02)

    __outputs(out_channels, [GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.HIGH])
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


def __cleanup():
    GPIO.cleanup()


atexit.register(__cleanup)
