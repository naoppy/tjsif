import time


class VideoWriteHelper:
    def __init__(self, fps, codec='X264'):
        self.last_time = None
        self.fps = fps
        self.sec_par_frame = 1.0 / fps
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter('output.avi', fourcc, fps, (height, width))
        self.writer = out

    def write_frame(self, frame):
        if self.last_time is None:
            self.writer.write(frame)
        else:
            time_diff = time.monotonic() - self.last_time
            frame_times = round(time_diff / self.sec_par_frame)
            for _ in range(frame_times):
                self.writer.write(frame)

        self.last_time = time.monotonic()

    def release(self):
        self.writer.release()
