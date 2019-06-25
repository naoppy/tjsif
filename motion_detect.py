# coding=utf-8
import cv2


# 輪郭を絞り込む関数（サイズで絞り込み）
def extract_contours(contours, size):
    """
    contours:領域の四点のx,y座標。
    size:どのくらいのサイズ以上だったら抽出するのか、という閾値。小さすぎると腕以外のものも検出してしまう。
    返り値:「size」で指定した面積以上の領域をリスト形式で返す。
    """
    area = 0
    list_extracted_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area >= size:
            list_extracted_contours.append(i)

    return list_extracted_contours


# 輪郭（長方形）を抽出し、画像に出力する関数
def get_rect(img, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def main():
    cap = cv2.VideoCapture(0)

    height = int(cap.get(3))
    width = int(cap.get(4))
    print("Camera Height:%d Width:%d" % (height, width))

    avg = None
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray, avg, 0.2)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
        _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        min_size = 500
        list_extracted_contours = extract_contours(contours, min_size)
        img = get_rect(frame, list_extracted_contours)

        cv2.imshow("thresh", thresh)
        cv2.imshow("motion", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
