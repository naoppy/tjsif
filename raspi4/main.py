import cv2

import video_writer_helper as writer
from raspi4 import misc
from raspi4 import detect_image


def main():
    cap = cv2.VideoCapture(0)

    height = int(cap.get(3))
    width = int(cap.get(4))
    print("Camera Height:%d Width:%d" % (height, width))

    out = writer.VideoWriteHelper(15.0, height, width)

    model = "models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
    labels = "models/coco_labels.txt"
    threshold = 0.6

    labels = detect_image.load_labels(labels)
    interpreter = detect_image.make_interpreter(model)
    interpreter.allocate_tensors()

    while True:
        ret, frame = cap.read()
        frame = misc.cv2pil(frame)

        processed_frame = edge_detect(frame)

        cv2.imshow('processed_frame', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def edge_detect(frame):
    pass


if __name__ == '__main__':
    main()
