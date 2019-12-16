import cv2
from PIL import Image, ImageDraw
from products import detect_image, misc, detect
import time


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

        processed_frame = edge_detect(interpreter, frame, threshold, labels)

        cv2.imshow('processed_frame', processed_frame)

        count += 1.0
        if count % 30 == 0:
            fps = count / (time.time() - start_time)
            print('FPS: {:.2f}'.format(fps))
            count = 0.0
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
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
    image = misc.cv2pil(frame_cv)

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
    cv_bgr = misc.pil2cv(image)
    return cv_bgr


if __name__ == '__main__':
    main()
