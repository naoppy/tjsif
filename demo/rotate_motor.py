import time

import cv2

from motor import control_stepping_motor as motorCtl
from products import detect_image, misc
from products.main import edge_detect_person

last_move_time = time.time()
last_move_direction = "L"


def rotate(persons, width):
    if not persons:
        return

    global last_move_time
    now = time.time()

    if now - last_move_time < 2:
        return

    min_x = min([person.bbox.xmin for person in persons])
    max_x = min([person.bbox.xmax for person in persons])
    left_diff = min_x
    right_diff = width - max_x

    if left_diff == right_diff:
        return

    move_direction = "R" if left_diff > right_diff else "L"

    # Reduce left and right round trip
    global last_move_direction
    if now - last_move_time < 3 and last_move_direction != move_direction:
        return

    if move_direction == "R":
        # rotate right
        print("rotate RIGHT")
        last_move_time = now
        last_move_direction = "R"
        motorCtl.right_spin_7_2degree()
    else:
        print("rotate LEFT")
        last_move_time = now
        last_move_direction = "L"
        motorCtl.left_spin_7_2degree()


def main():
    cap = cv2.VideoCapture(0)
    # Camera Settings
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Print Camera Information
    print("fps:%d" % (cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera Height:%d Width:%d" % (height, width))
    print("Camera Encoding:%s" % (misc.decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))))

    model_file = "../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
    label_file = "../all_models/coco_labels.txt"
    threshold = 0.6

    labels = detect_image.load_labels(label_file)
    interpreter = detect_image.make_interpreter(model_file)
    interpreter.allocate_tensors()

    start_time = time.time()
    count = 0.0

    while True:
        ret, frame = cap.read()

        persons = edge_detect_person(interpreter, frame, threshold)

        rotate(persons, width)

        count += 1.0
        if count % 120 == 0:
            fps = count / (time.time() - start_time)
            print('FPS: {:.2f}'.format(fps))
            count = 0.0
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
