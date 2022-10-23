import math
import numpy as np
import cv2


def view_image(path):
    img = cv2.imread(path, 1)
    cv2.namedWindow('Display window', cv2.WINDOW_FREERATIO)
    cv2.imshow('Display window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_ANY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def save_as_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_ANY)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 30, (w, h))
    while True:
        ok, img = cap.read()
        video_writer.write(img)
        if not ok:
            break
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


def record_video():
    # Камера в режиме реального времени.
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

    # Камера с записью.
    video = cv2.VideoCapture(0)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))
    while True:
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def read_ip_write_to_file():
    video = cv2.VideoCapture("http://192.168.63.31:8080/video/mjpeg")
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))
    while True:
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


# Создание матрицы свертки
def convolution_matrix(size, sigma, a, b):
    gauss = [[0.0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            gauss[i][j] = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(
                -(((i - a) ** 2 + (j - b) ** 2) / (2 * (sigma ** 2))))
    sum = np.sum(gauss)
    gauss /= sum
    return gauss


# Умножение для размытия по Гауссу
def mult_for_gauss_blur(img, conv_matrix):
    n = len(conv_matrix)
    margin = n // 2
    for i in range(len(img) - margin * 2):
        for j in range(len(img[0]) - margin * 2):
            img[margin + i][margin + j] = np.sum(np.multiply(img[i:i + n, j:j + n], conv_matrix))
    return img


# Реализация размытия по Гауссу
def gaussian_blur(size_conv_matrix, sigma, a, b, path):
    conv_matrix = convolution_matrix(size_conv_matrix, sigma, a, b)
    gray_image = cv2.imread(path, 0)
    gray_image = mult_for_gauss_blur(gray_image, conv_matrix)
    cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Display window', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # lab1

    # 2
    """view_image(r'C:\\Users\\romAn\\image\\small_rabbit.jpg')
    """
    # 3
    """open_video(r'C:\\Users\\romAn\\video\\video1.mp4')
    """

    # 4
    """save_as_video(r'C:\\Users\\romAn\\video\\video1.mp4')
    """

    # 5
    """record_video()
    """

    # 6
    """read_ip_write_to_file()
    """

    # lab2

    # lab2
    # view_image(r'C:\\Users\\romAn\\image\\husky.jpg')
    conv_matrix_size = 21
    sigma = 0.3 * ((conv_matrix_size - 1) * 0.5 - 1) + 0.8
    gaussian_blur(conv_matrix_size, sigma, (conv_matrix_size - 1) // 2, (conv_matrix_size - 1) // 2, r'C:\\Users\\romAn\\image\\husky.jpg')
    """img = cv2.imread(r'C:\\Users\\romAn\\image\\husky.jpg', 0)
                img_blur = cv2.blur(img, (20, 20))
                cv2.namedWindow("Blur")
                cv2.imshow("Blur", img_blur)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    """
    