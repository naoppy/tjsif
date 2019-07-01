import time


class VideoWriteHelper:
    def __init__(self, fps, codec='X264'):
        self.lasttime = None
        self.fps = fps
        self.sec_par_frame = 1.0 / fps
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter('output.avi', fourcc, fps, (height, width))
        self.writer = out

    def write_frame(self, frame):
        if self.lasttime is None:
            # TODO!!!!!!
            pass

        time_diff = self.lasttime - time.monotonic()
        time = time_diff / self.sec_par_frame

        self.lasttime = time.monotonic()
