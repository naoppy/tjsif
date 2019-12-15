from PIL import Image
import cv2
import numpy as np


def cv2pil(image):
    """ OpenCV型 -> PIL型 """
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(new_image).convert("RGB")


def pil2cv(image):
    """ PIL型 -> OpenCV型 """
    new_image = np.array(image, dtype=np.uint8)
    return cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)


def decode_fourcc(v):
    """"
    THIS IS FOR DEBUG
    """
    # https://amdkkj.blogspot.com/2017/06/opencv-python-for-windows-playing-videos_17.html
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def draw_persons(draw, persons):
    """
    Draws the bounding box for each person.
    :param draw: PIL Draw Object
    :param persons: detected persons
    """
    for person in persons:
        bbox = person.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  'person\n%.2f' % person.score,
                  fill='red')
