import argparse
import cv2
import imp
import numpy as np
import os
import re
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def get_person_list(objs, labels):
    return_list = []
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
        percent = int(100 * obj.score)
        if labels[obj.label_id] == "person":
            return_list += (x0, y0, x1, y1, percent)

    return return_list


def write_rect(img, tuple):
    for x0, y0, x1, y1, percent in tuple:
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 5)


def prepare_edgetpu():
    default_model_dir = './all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='class score threshold')
    args = parser.parse_args()

    print("Loading %s with %s labels." % (args.model, args.labels))
    engine = DetectionEngine(args.model)
    labels = load_labels(args.labels)
    threshold = args.threshold
    top_k = args.top_k
    return engine, labels, threshold, top_k,


def main_loop():
    cap = cv2.VideoCapture(0)

    height = int(cap.get(3))
    width = int(cap.get(4))
    print("Camera Height:%d Width:%d" % (height, width))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (height, width))

    # prepare EdgeTPU
    engine, labels, threshold, top_k = prepare_edgetpu()

    # main func
    def main_func(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)

        start_time = time.monotonic()
        objs = engine.DetectWithImage(im_pil, threshold=threshold,
                                      keep_aspect_ratio=True, relative_coord=True,
                                      top_k=top_k)
        end_time = time.monotonic()
        text_lines = [
            'Inference: %.2f ms' % ((end_time - start_time) * 1000),
            'FPS: %.2f fps' % (1.0 / (end_time - start_time)),
            '%d object found' % len(objs),
        ]
        print(' '.join(text_lines))
        person_list = get_person_list(objs, labels)
        for e in person_list:
            write_rect(image, e)
        return image

    while True:
        ret, frame = cap.read()

        processed_frame = main_func(frame)

        cv2.imshow('processed_frame', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_loop()
