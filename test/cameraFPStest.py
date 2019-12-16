import cv2
import time
from products import misc


def main():
    cap = cv2.VideoCapture(0)
    # Camera Settings
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Print Camera Information
    print("fps:%d" % (cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera Height:%d Width:%d" % (height, width))
    print("Camera Encoding:%s" % (misc.decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))))

    count = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        count += 1.0
        if count % 50 == 0:
            fps = count / (time.time() - start_time)
            print('FPS: {:.2f}'.format(fps))
            count = 0
            start_time = time.time()


if __name__ == '__main__':
    main()
