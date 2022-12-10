import cv2
import numpy as np
import time


def webcam():
    # url = r"http://192.168.100.250:8080/video/mjpeg"
    video = cv2.VideoCapture(0, cv2.CAP_ANY)
    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer1 = cv2.VideoWriter('output_all.mov', fourcc, fps, (w, h))
    video_writer2 = cv2.VideoWriter('output_actions.mov', fourcc, fps, (w, h))

    cur_frame = None
    old_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        ok, img = video.read()
        img = cv2.resize(img, (w, h))
        cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        img = cv2.putText(img, current_time, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 155, 155), 4, cv2.LINE_8)
        cv2.imshow('img', img)
        video_writer1.write(img)

        frame_diff = cv2.absdiff(cur_frame, old_frame)
        ret, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            if cv2.contourArea(i) > 1000:
                video_writer2.write(img)
                break

        old_frame = cur_frame
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video.release()


if __name__ == '__main__':
    webcam()
