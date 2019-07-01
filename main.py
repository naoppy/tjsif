# coding=utf-8
import cv2


def main():
    cap = cv2.VideoCapture(0)

    height = int(cap.get(3))
    width = int(cap.get(4))
    print("Camera Height:%d Width:%d" % (height, width))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (height, width))

    # prepare EdgeTPU
    engine, labels, threshold, top_k = prepare_edgetpu()

    # main func
    def edge_detect(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)

        start_time = time.monotonic()
        objs = engine.DetectWithImage(im_pil, threshold=threshold,
                                      keep_aspect_ratio=True, relative_coord=True,
                                      top_k=top_k)
        end_time = time.monotonic()
        # text_lines = [
        #     'Inference: %.2f ms' % ((end_time - start_time) * 1000),
        #     'FPS: %.2f fps' % (1.0 / (end_time - start_time)),
        #     '%d object found' % len(objs),
        # ]
        # print(' '.join(text_lines))
        person_list = get_person_list(objs, labels)
        for e in person_list:
            write_rect(image, e)
        return image

    while True:
        ret, frame = cap.read()

        processed_frame = edge_detect(frame)

        cv2.imshow('processed_frame', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
