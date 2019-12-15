import cv2
from products import video_writer_helper

def decode_fourcc(v):
    """"
    THIS IS FOR DEBUG
    """
    # https://amdkkj.blogspot.com/2017/06/opencv-python-for-windows-playing-videos_17.html
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:%d" % fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera Height:%d Width:%d" % (height, width))
    print("Camera Encoding:%s" % (decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))))

    import datetime
    filename = "output_{0:%Y%m%d-%H-%M-%S}.avi".format(datetime.datetime.now())
    codec = "H264"
    out = video_writer_helper.VideoWriteHelper(fps, height, width, filename, codec)

    while True:
        ret, frame = cap.read()

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write_frame(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
