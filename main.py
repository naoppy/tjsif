import argparse
import imp
import os
import re
from edgetpu.detection.engine import DetectionEngine
import numpy as np
import cv2


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def get_person_list(objs, labels):
    returnlist = []
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().toList()
        percent = int(100 * obj.score)
        if labels[obj.label_id] == "person":
            returnlist += (x0, y0, x1, y1, percent)

    return returnlist


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


def main_loop(function):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow('frame', frame)

        function(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    def empty_func(frame):
        pass
    
    main_loop(empty_func)
