# coding=utf-8
import cv2
from products import misc

avg = None


def extract_contours(contours, size):
    """
    contours:領域の四点のx,y座標。
    size:どのくらいのサイズ以上だったら抽出するのか、という閾値。小さすぎると腕以外のものも検出してしまう。
    返り値:「size」で指定した面積以上の領域をリスト形式で返す。
    """
    list_extracted_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area >= size:
            list_extracted_contours.append(i)

    return list_extracted_contours


def write_rect_to_img(img, contours):
    """
    Just For Debug
    :param img: image which is overdraw rectangles
    :param contours:
    :return: given image
    """
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def frame_diff_detection(frame):
    """

    :param frame:opencv_image
    :return: if none, it is the first call to this function else given image (or overdraw given image)
    """
    # convert BGR(opencv_image) to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    global avg
    if avg is None:
        avg = gray.copy().astype("float")
        return None

    # accumulate
    # avg = (1-alpha)*avg + alpha*gray
    cv2.accumulateWeighted(gray, avg, 0.2)
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    # 閾値は第二引数
    thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    min_size = 1500
    list_extracted_contours = extract_contours(contours, min_size)
    cv2.imshow("thresh", thresh)

    # THIS IS FOR DEBUG
    retimg = write_rect_to_img(frame, list_extracted_contours)

    return retimg


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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera Failed")

        img = frame_diff_detection(frame)
        if img is None:
            continue
        cv2.imshow("motion", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
