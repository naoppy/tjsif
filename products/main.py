import cv2
import time
from PIL import Image, ImageDraw

from products import detect_image, detect, motion_detect, video_writer_helper, calc_lotation, misc


# Implementation of tjsif_flowchart.svg

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
    # TPU settings
    model_file = "../all_models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
    label_file = "../all_models/coco_labels.txt"
    threshold = 0.6

    labels = detect_image.load_labels(label_file)  # Unused
    interpreter = detect_image.make_interpreter(model_file)
    interpreter.allocate_tensors()

    while True:
        ret, frame = cap.read()

        # DEBUG CODE===
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        # FIN DEBUG CODE===

        if motion_detect.frame_diff_detection(frame):
            persons = edge_detect_person(interpreter, frame, threshold)
            if persons:
                # There are moving persons
                # start Recording Loop

                # DEBUG CODE===
                # cv2.destroyAllWindows()
                # FIN DEBUG CODE===
                recording_loop(cap, interpreter, threshold)
            else:
                # Do Nothing
                pass
        else:
            # Do Nothing
            pass

    cap.release()
    cv2.destroyAllWindows()


def recording_loop(cap, interpreter, threshold):
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

        persons = edge_detect_person(interpreter, frame, threshold)

        # DEBUG CODE===
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        # FIN DEBUG CODE===

        if persons:
            last_detect_time = time.time()
            calc_lotation.rotate(persons, width)
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


def edge_detect_person(interpreter, frame_cv, threshold):
    """
    object-detection using edge tpu
    detect person only
    :param frame_cv: OpenCV image
    :param threshold: score threshold
    :return: detected persons list
    """
    # OpenCVはBGR、PillowはRGB
    image = misc.cv2pil(frame_cv)

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
    # DEBUG CODE===
    # misc.draw_persons(ImageDraw.Draw(image), persons)
    # cv_image = misc.pil2cv(image)
    # cv2.imshow("recording frame", cv_image)
    # FIN DEBUG CODE===
    return persons


if __name__ == '__main__':
    main()
