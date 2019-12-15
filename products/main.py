import cv2
import time
import numpy as np
from PIL import Image, ImageDraw

from products import detect_image, detect, motion_detect, video_writer_helper, calc_lotation


# Implementation of tjsif_flowchart.svg

def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print("fps:%d" % (cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera Height:%d Width:%d" % (height, width))
    print("Camera Encoding:%s" % (decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))))

    model_file = "../all_models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
    label_file = "../all_models/coco_labels.txt"
    threshold = 0.6

    labels = detect_image.load_labels(label_file)
    interpreter = detect_image.make_interpreter(model_file)
    interpreter.allocate_tensors()

    while True:
        ret, frame = cap.read()

        # DEBUG CODE===
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # FIN DEBUG CODE===

        if motion_detect.frame_diff_detection(frame):
            persons = edge_detect_person(interpreter, frame, threshold, labels)
            if persons:
                # There are moving persons
                # start Recording Loop

                # DEBUG CODE===
                # cv2.destroyAllWindows()
                # FIN DEBUG CODE===
                recording_loop(cap, interpreter, threshold, labels)
            else:
                # Do Nothing
                pass
        else:
            # Do Nothing
            pass

    cap.release()
    cv2.destroyAllWindows()


def recording_loop(cap, interpreter, threshold, labels):
    print("start recording")

    last_detect_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    import datetime
    filename = "output_{0:%Y%m%d-%H-%M-%S}.avi".format(datetime.datetime.now())
    codec = "H264"
    out = video_writer_helper.VideoWriteHelper(fps, height, width, filename, codec)

    while True:
        ret, frame = cap.read()

        persons = edge_detect_person(interpreter, frame, threshold, labels)

        if persons:
            last_detect_time = time.time()
            calc_lotation.lotate(persons)
            out.write_frame(frame)
        else:
            now = time.time()
            # no person and 30 sec passed...
            if now - last_detect_time >= 30:
                # Write Out Movie and exit this loop
                out.release()
                break
            else:
                out.write_frame(frame)

    print("finish recording")


def edge_detect_person(interpreter, frame_cv, threshold, labels):
    """
    object-detection using edge tpu
    detect person only
    :param frame_cv: OpenCV image
    :param threshold: score threshold
    :return: detected persons list
    """
    # OpenCVはBGR、PillowはRGB
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert("RGB")

    scale = detect.set_input(interpreter, image.size,
                             lambda size: image.resize(size, Image.ANTIALIAS))

    interpreter.invoke()

    objs = detect.get_output(interpreter, threshold, scale)
    persons = [x for x in objs if x.id == 0]

    # THIS IS FOR DEBUG
    # for obj in objs:
    #     print('person')
    #     print('  score: ', obj.score)
    #     print('  bbox:  ', obj.bbox)
    #
    # detect_image.draw_objects(ImageDraw.Draw(image), persons, labels)
    # cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # return cv_image
    return persons


def decode_fourcc(v):
    """"
    THIS IS FOR DEBUG
    """
    # https://amdkkj.blogspot.com/2017/06/opencv-python-for-windows-playing-videos_17.html
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


if __name__ == '__main__':
    main()
