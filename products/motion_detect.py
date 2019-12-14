# coding=utf-8
import cv2

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


def frame_diff_detection(frame):
    """
    画像に変化があったかを判定する
    面積閾値は1500, 二値化閾値は40
    :param frame:opencv image
    :return: if True, it means the image has changed, if False, means the image has not changed
    """
    # convert BGR(opencv_image) to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    global avg
    # The first call
    if avg is None:
        avg = gray.copy().astype("float")
        return False

    # accumulate
    # avg = (1-alpha)*avg + alpha*gray
    cv2.accumulateWeighted(gray, avg, 0.2)
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    # 閾値は第二引数
    thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    min_size = 1500
    list_extracted_contours = extract_contours(contours, min_size)

    # THIS IS FOR DEBUG
    # cv2.imshow("thresh", thresh)
    # cv2.drawContours(frame, list_extracted_contours, -1, (0, 255, 0), 2)
    # cv2.imshow("frame", frame)

    if list_extracted_contours:
        return True
    else:
        return False
