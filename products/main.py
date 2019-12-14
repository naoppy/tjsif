import cv2
import numpy as np
from PIL import Image, ImageDraw

import video_writer_helper as writer
from raspi4 import detect_image, detect


def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print("fps:%d" % (cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera Height:%d Width:%d" % (height, width))

    model_file = "../all_models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
    label_file = "../all_models/coco_labels.txt"
    threshold = 0.6

    labels = detect_image.load_labels(label_file)
    interpreter = detect_image.make_interpreter(model_file)
    interpreter.allocate_tensors()

    while True:
        ret, frame = cap.read()

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def edge_detect_person(interpreter, frame_cv, threshold, labels):
    """
    object-detection using edge tpu
    detect person only
    :param labels:
    :param threshold:
    :param interpreter:
    :param frame_cv: OpenCV image
    :return:
    """
    # OpenCVはBGR、PillowはRGB
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGB")

    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))

    interpreter.invoke()

    objs = detect.get_output(interpreter, threshold, scale)
    persons = [x for x in objs if x.id == 0]

    for obj in objs:
        print('person')
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    detect_image.draw_objects(ImageDraw.Draw(image), persons, labels)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv_image


if __name__ == '__main__':
    main()
