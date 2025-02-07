# coding=utf-8
import time
import cv2


class VideoWriteHelper:
    def __init__(self, fps, height, width, filename, codec):
        """
        please refer to http://shibafu3.hatenablog.com/entry/2016/11/13/151118
        :param fps: 動画のfps
        :param height: 動画の縦のサイズ
        :param width: 動画の横サイズ
        :param filename: 保存するファイルの名前
        :param codec: ビデオコーデック, opencvのドキュメントを参照
        """
        self.last_time = None
        self.fps = fps
        self.sec_par_frame = 1.0 / fps
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        self.writer = out

    def write_frame(self, frame):
        """
        動画に画像を数フレーム計算して書き出す
        :param frame: opencv形式の画像
        """
        now = time.monotonic()

        if self.last_time is None:
            self.writer.write(frame)
        else:
            time_diff = time.monotonic() - self.last_time
            frame_times = round(time_diff / self.sec_par_frame)

            if frame_times is 0:
                return

            for _ in range(frame_times):
                self.writer.write(frame)

        self.last_time = now
        return

    def release(self):
        """
        後処理、これをしないとバグる
        """
        self.writer.release()
        return
