import argparse
import gstreamer
import imp
import os
import re
import svgwrite
import time
from edgetpu.detection.engine import DetectionEngine


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
        label = '%d%% %s' % (percent, lables[obj.label_id])
        if label == "person":
            returnlist += (x0, y0, x1, y1, percent)

    return returnlist


def main():
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

    

if __name__ == '__main__':
    main()
