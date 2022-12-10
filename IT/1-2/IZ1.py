import numpy as np
import cv2
import math
import time

# Метод Канни

def round_angle(sum_gx, sum_gy):
    tangens = math.tan(sum_gy / sum_gx)
    fi = -1

    if (sum_gx > 0 and sum_gy < 0 and tangens < -2.414) or (sum_gx < 0 and sum_gy < 0 and tangens > 2.414):
        fi = 0
    elif (sum_gx > 0 and sum_gy < 0 and tangens < -0.414):
        fi = 1
    elif (sum_gx > 0 and sum_gy < 0 and tangens > -0.414) or (sum_gx > 0 and sum_gy > 0 and tangens < 0.414):
        fi = 2
    elif (sum_gx > 0 and sum_gy > 0 and tangens < 2.414):
        fi = 3
    elif (sum_gx > 0 and sum_gy > 0 and tangens > 2.414) or (sum_gx < 0 and sum_gy > 0 and tangens < -2.414):
        fi = 4
    elif (sum_gx < 0 and sum_gy > 0 and tangens < -0.414):
        fi = 5
    elif (sum_gx < 0 and sum_gy > 0 and tangens > -0.414) or (sum_gx < 0 and sum_gy < 0 and tangens < 0.414):
        fi = 6
    elif (sum_gx < 0 and sum_gy < 0 and tangens < 2.414):
        fi = 7
    return fi


def non_maximum_suppression(grad_lenght, grad_angle, img2):
    for i in range(3, len(grad_lenght) - 3):
        for j in range(3, len(grad_lenght[i]) - 3):
            img2[i][j] = 0
            if grad_angle[i][j] == 6 or grad_angle[i][j] == 2:
                if grad_lenght[i][j] > grad_lenght[i][j - 1] and grad_lenght[i][j] > grad_lenght[i][j + 1]:
                    img2[i][j] = 255
                else:
                    img2[i][j] = 0
            elif grad_angle[i][j] == 4 or grad_angle[i][j] == 0:
                if grad_lenght[i][j] > grad_lenght[i - 1][j] and grad_lenght[i][j] > grad_lenght[i + 1][j]:
                    img2[i][j] = 255
                else:
                    img2[i][j] = 0
            elif grad_angle[i][j] == 5 or grad_angle[i][j] == 1:
                if grad_lenght[i][j] > grad_lenght[i + 1][j - 1] and grad_lenght[i][j] > grad_lenght[i - 1][j + 1]:
                    img2[i][j] = 255
                else:
                    img2[i][j] = 0
            elif grad_angle[i][j] == 7 or grad_angle[i][j] == 2:
                if grad_lenght[i][j] > grad_lenght[i - 1][j - 1] and grad_lenght[i][j] > grad_lenght[i + 1][j + 1]:
                    img2[i][j] = 255
                else:
                    img2[i][j] = 0
    return img2


def double_filtration(max_grad_lenght, grad_lenght, img):
    low_level = max_grad_lenght // 15
    high_level = max_grad_lenght // 10

    for i in range(3, len(grad_lenght) - 3):
        for j in range(3, len(grad_lenght[i]) - 3):
            if img[i][j] == 255:
                if grad_lenght[i][j] < low_level:
                    img[i][j] = 0
                elif grad_lenght[i][j] < high_level:
                    img[i][j] = 0
                    for h in range(0, 8):
                        for g in range(0, 8):
                            if h != 4 and g != 4 and img[i - 4 + h][j - 4 + g] == 255:
                                img[i][j] == 255
    return img


def sobel():  # оператор Собеля
    gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    return gx, gy


def roberts():  # оператор Робертса
    gx = [[-1, 0], [0, 1]]
    gy = [[0, -1], [1, 0]]
    return gx, gy


def previtt():  # оператор Превитта
    gx = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    gy = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    return gx, gy


def picture1():
    start_time = time.time()
    img = cv2.imread(r'C:\Users\romAn\IZ1\turtles1.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img2, (5, 5))

    grad_lenght = [[0 for i in range(len(blur[j]))] for j in range(len(blur))]
    grad_angle = [[0 for i in range(len(blur[j]))] for j in range(len(blur))]

    gx, gy = sobel()
    max_grad_lenght = 0

    for i in range(len(gx) // 2, len(blur) - len(gx) % 2):
        for j in range(len(gx) // 2, len(blur[i]) - len(gx) % 2):
            sum_gx = 0
            sum_gy = 0

            for h in range(len(gx)):
                for g in range(len(gx)):
                    sum_gx += gx[h][g] * blur[h - len(gx) // 2 + i][g - len(gx) // 2 + j]
                    sum_gy += gy[h][g] * blur[h - len(gy) // 2 + i][g - len(gy) // 2 + j]
            grad_lenght_temp = math.sqrt(sum_gx ** 2 + sum_gy ** 2)
            if sum_gx == 0:
                sum_gx = 0.0000001

            fi = round_angle(sum_gx, sum_gy)

            grad_lenght[i][j] = grad_lenght_temp
            grad_angle[i][j] = fi
            max_grad_lenght = max(max_grad_lenght, grad_lenght_temp)

    img2 = non_maximum_suppression(grad_lenght, grad_angle, img2)
    img2 = double_filtration(max_grad_lenght, grad_lenght, img2)

    print('time = ' + str(time.time() - start_time))

    cv2.imshow('Display window', img2)
    cv2.waitKey(0)


# Метод математической морфологии

def erosion(img, struct_element):
    img2 = np.copy(img)
    for i in range(len(struct_element) // 2, len(img) - len(struct_element) // 2):
        for j in range(len(struct_element) // 2, len(img[i]) - len(struct_element) // 2):
            img_part = img[i - len(struct_element) // 2: i + 1 + len(struct_element) // 2,
                       j - len(struct_element) // 2: j + 1 + len(struct_element) // 2]
            img2[i][j] = np.min(img_part[struct_element])
    return img2


def dilate(img, struct_element):
    img2 = np.copy(img)
    for i in range(len(struct_element) // 2, len(img) - len(struct_element) // 2):
        for j in range(len(struct_element) // 2, len(img[i]) - len(struct_element) // 2):
            img_part = img[i - len(struct_element) // 2: i + 1 + len(struct_element) // 2,
                       j - len(struct_element) // 2: j + 1 + len(struct_element) // 2]
            img2[i][j] = np.max(img_part[struct_element])
    return img2


def razn(img1, img2):
    img3 = np.copy(img1)
    for i in range(len(img1)):
        for j in range(len(img1[i])):
            img3[i][j] -= img2[i][j]
    return img3


def picture2():
    start_time = time.time()
    img = cv2.imread(r'C:\Users\romAn\IZ1\turtles1.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    struct_element = np.array([
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True]], dtype=bool)

    img3 = erosion(img2, struct_element)
    # cv2.imshow('Display window', img3)
    # cv2.waitKey(0)

    img4 = dilate(img2, struct_element)
    # cv2.imshow('Display window', img4)
    # cv2.waitKey(0)

    # img5 = razn(img4, img3)
    # cv2.imshow('Display window', img5)
    # cv2.waitKey(0)

    print('time = ' + str(time.time() - start_time))

    img6 = razn(img2, img3)
    cv2.imshow('Display window', img6)
    cv2.waitKey(0)

    img7 = razn(img4, img2)
    cv2.imshow('Display window', img7)
    cv2.waitKey(0)


if __name__ == '__main__':
    picture2()
