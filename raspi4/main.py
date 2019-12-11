import cv2
from PIL import Image, ImageDraw

import video_writer_helper as writer
from raspi4 import detect_image, detect


def main():
    cap = cv2.VideoCapture(0)

    height = int(cap.get(3))
    width = int(cap.get(4))
    print("Camera Height:%d Width:%d" % (height, width))

    out = writer.VideoWriteHelper(15.0, height, width)

    model_file = "models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
    label_file = "models/coco_labels.txt"
    threshold = 0.6

    labels = detect_image.load_labels(label_file)
    interpreter = detect_image.make_interpreter(model_file)
    interpreter.allocate_tensors()

    while True:
        ret, frame = cap.read()

        processed_frame = edge_detect(interpreter, frame, threshold, labels)

        cv2.imshow('processed_frame', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def edge_detect(interpreter, frame_cv, threshold, labels):
    """
    object-detection using edge tpu
    :param labels:
    :param threshold:
    :param interpreter:
    :param frame_cv: OpenCV image
    :return:
    """
    # OpenCVはBGR、PillowはRGB
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))

    interpreter.invoke()

    objs = detect.get_output(interpreter, threshold, scale)

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    detect_image.draw_objects(ImageDraw.Draw(image), objs, labels)
    cv_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv_bgr


if __name__ == '__main__':
    main()
