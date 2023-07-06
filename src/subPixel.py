import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import os
import sys

from customFunction import showPic
from recognition import recognition
from scipy.interpolate import splprep, splev
from scipy.special import comb
from scipy.optimize import fsolve, leastsq
from collections import defaultdict, deque


from skimage import morphology
from collections import deque




M00 = np.array([0, 0.0287, 0.0686, 0.0807, 0.0686, 0.0287, 0,
                0.0287, 0.0815, 0.0816, 0.0816, 0.0816, 0.0815, 0.0287,
                0.0686, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816, 0.0686,
                0.0807, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816, 0.0807,
                0.0686, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816, 0.0686,
                0.0287, 0.0815, 0.0816, 0.0816, 0.0816, 0.0815, 0.0287,
                0, 0.0287, 0.0686, 0.0807, 0.0686, 0.0287, 0]).reshape((7, 7))

M11R = np.array([0, -0.015, -0.019, 0, 0.019, 0.015, 0,
                 -0.0224, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0224,
                 -0.0573, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0573,
                 -0.069, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.069,
                 -0.0573, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0573,
                 -0.0224, -0.0466, -0.0233, 0, 0.0233, 0.0466, 0.0224,
                 0, -0.015, -0.019, 0, 0.019, 0.015, 0]).reshape((7, 7))

M11I = np.array([0, -0.0224, -0.0573, -0.069, -0.0573, -0.0224, 0,
                 -0.015, -0.0466, -0.0466, -0.0466, -0.0466, -0.0466, -0.015,
                 -0.019, -0.0233, -0.0233, -0.0233, -0.0233, -0.0233, -0.019,
                 0, 0, 0, 0, 0, 0, 0,
                 0.019, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233, 0.019,
                 0.015, 0.0466, 0.0466, 0.0466, 0.0466, 0.0466, 0.015,
                 0, 0.0224, 0.0573, 0.069, 0.0573, 0.0224, 0]).reshape((7, 7))

M20 = np.array([0, 0.0225, 0.0394, 0.0396, 0.0394, 0.0225, 0,
                0.0225, 0.0271, -0.0128, -0.0261, -0.0128, 0.0271, 0.0225,
                0.0394, -0.0128, -0.0528, -0.0661, -0.0528, -0.0128, 0.0394,
                0.0396, -0.0261, -0.0661, -0.0794, -0.0661, -0.0261, 0.0396,
                0.0394, -0.0128, -0.0528, -0.0661, -0.0528, -0.0128, 0.0394,
                0.0225, 0.0271, -0.0128, -0.0261, -0.0128, 0.0271, 0.0225,
                0, 0.0225, 0.0394, 0.0396, 0.0394, 0.0225, 0]).reshape((7, 7))

M31R = np.array([0, -0.0103, -0.0073, 0, 0.0073, 0.0103, 0,
                 -0.0153, -0.0018, 0.0162, 0, -0.0162, 0.0018, 0.0153,
                 -0.0223, 0.0324, 0.0333, 0, -0.0333, -0.0324, 0.0223,
                 -0.0190, 0.0438, 0.0390, 0, -0.0390, -0.0438, 0.0190,
                 -0.0223, 0.0324, 0.0333, 0, -0.0333, -0.0324, 0.0223,
                 -0.0153, -0.0018, 0.0162, 0, -0.0162, 0.0018, 0.0153,
                 0, -0.0103, -0.0073, 0, 0.0073, 0.0103, 0]).reshape(7, 7)

M31I = np.array([0, -0.0153, -0.0223, -0.019, -0.0223, -0.0153, 0,
                 -0.0103, -0.0018, 0.0324, 0.0438, 0.0324, -0.0018, -0.0103,
                 -0.0073, 0.0162, 0.0333, 0.039, 0.0333, 0.0162, -0.0073,
                 0, 0, 0, 0, 0, 0, 0,
                 0.0073, -0.0162, -0.0333, -0.039, -0.0333, -0.0162, 0.0073,
                 0.0103, 0.0018, -0.0324, -0.0438, -0.0324, 0.0018, 0.0103,
                 0, 0.0153, 0.0223, 0.0190, 0.0223, 0.0153, 0]).reshape(7, 7)

M40 = np.array([0, 0.013, 0.0056, -0.0018, 0.0056, 0.013, 0,
                0.0130, -0.0186, -0.0323, -0.0239, -0.0323, -0.0186, 0.0130,
                0.0056, -0.0323, 0.0125, 0.0406, 0.0125, -0.0323, 0.0056,
                -0.0018, -0.0239, 0.0406, 0.0751, 0.0406, -0.0239, -0.0018,
                0.0056, -0.0323, 0.0125, 0.0406, 0.0125, -0.0323, 0.0056,
                0.0130, -0.0186, -0.0323, -0.0239, -0.0323, -0.0186, 0.0130,
                0, 0.013, 0.0056, -0.0018, 0.0056, 0.013, 0]).reshape(7, 7)


def zernike_detection(img, cnt):
    c_img = img
    g_N = 7
    ZerImgM00 = cv2.filter2D(c_img, cv2.CV_64F, M00)
    ZerImgM11R = cv2.filter2D(c_img, cv2.CV_64F, M11R)
    ZerImgM11I = cv2.filter2D(c_img, cv2.CV_64F, M11I)
    ZerImgM20 = cv2.filter2D(c_img, cv2.CV_64F, M20)
    ZerImgM31R = cv2.filter2D(c_img, cv2.CV_64F, M31R)
    ZerImgM31I = cv2.filter2D(c_img, cv2.CV_64F, M31I)
    ZerImgM40 = cv2.filter2D(c_img, cv2.CV_64F, M40)

    point_temporary_x = []
    point_temporary_y = []
    scatter_arr = cv2.findNonZero(ZerImgM00).reshape(-1, 2)
    for idx in scatter_arr:
        j, i = idx
        theta_temporary = np.arctan2(ZerImgM31I[i][j], ZerImgM31R[i][j])
        rotated_z11 = np.sin(theta_temporary) * ZerImgM11I[i][j] + np.cos(theta_temporary) * ZerImgM11R[i][j]
        rotated_z31 = np.sin(theta_temporary) * ZerImgM31I[i][j] + np.cos(theta_temporary) * ZerImgM31R[i][j]
        l_method1 = np.sqrt((5 * ZerImgM40[i][j] + 3 * ZerImgM20[i][j]) / (8 * ZerImgM20[i][j]))

        l_method2 = np.sqrt((5 * rotated_z31 + rotated_z11) / (6 * rotated_z11))

        l = (l_method1 + l_method2) / 2

        k = 3 * rotated_z11 / (2 * (1 - l_method2 ** 2) ** 1.5)

        # h = (ZerImgM00[i][j] - k * np.pi / 2 + k * np.arcsin(l_method2) + k * l_method2 * (1 - l_method2 ** 2) ** 0.5)
        # / np.pi
        # k_value = 20.0
        # l_value = 2 ** 0.5 / g_N
        k_value = 5.0
        l_value = 0.8 / g_N

        absl = np.abs(l_method2 - l_method1)

        if k >= k_value and absl <= l_value:
            y = i + g_N * l * np.sin(theta_temporary) / 2
            x = j + g_N * l * np.cos(theta_temporary) / 2
            point_temporary_x.append(x)
            point_temporary_y.append(y)
        else:
            continue

    return point_temporary_x, point_temporary_y


def radial_poly(rho, n, m):
    if (n - m) % 2 != 0:
        return 0
    else:
        s = int((n - m) / 2)
        radial = 0
        for k in range(s+1):
            c = (-1)**k * comb(n - k, s - k) * comb(n - 2*s + k, k)
            radial += c * rho**(n - 2*k)
        return radial


def zernike_moment(img, n, m, cx, cy, radius):
    moments = 0
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    x = x - cx
    y = y - cy
    rho = np.sqrt(x**2 + y**2) / radius
    phi = np.arctan2(y, x)
    mask = np.logical_and(rho <= 1, img > 0)
    rho = rho[mask]
    phi = phi[mask]
    img = img[mask]
    for i in range(len(img)):
        moments += img[i] * radial_poly(rho[i], n, m) * np.exp(-1j*m*phi[i])
    moments *= (n + 1) / np.pi
    return moments


def zernike_edge_detection(img, circle_center, circle_radius):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist=int(circle_radius*1.5), param1=50, param2=30, minRadius=int(circle_radius*0.5), maxRadius=int(circle_radius*1.5))
    edges = np.zeros(gray.shape, dtype=np.uint8)
    for circle in circles[0]:
        x, y, radius = circle.astype(np.int)
        roi = gray[y-radius:y+radius, x-radius:x+radius]
        moments = []
        for n in range(8):
            for m in range(-n, n+1, 2):
                moment = zernike_moment(roi, n, m, radius, radius, radius)
                moments.append((n, m, moment))
        moments = sorted(moments, key=lambda x: abs(x[2]), reverse=True)
        n, m, moment = moments[0]
        derivative = -m * moment.real / moment.imag
        x1 = int(x + radius * np.cos(-derivative))
        y1 = int(y + radius * np.sin(-derivative))
        x2 = int(x + radius * np.cos(np.pi - derivative))
        y2 = int(y + radius * np.sin(np.pi - derivative))
        edges = cv2.line(edges, (x1, y1), (x2, y2), 255, 1)
    return edges

# def third_order_moment(img, cnt):
#     moments = cv2.moments(img)
#     mu20 = moments['mu20']
#     mu02 = moments['mu02']
#     mu30 = moments['mu30']
#     mu03 = moments['mu03']
#     mu21 = moments['mu21']
#     mu12 = moments['mu12']
#
#     x_c = moments['m10'] / moments['m00']
#     y_c = moments['m01'] / moments['m00']
#
#     central_moment_30 = mu30 - 3 * x_g * mu20 + 2 * x_g ** 3
#     central_moment_03 = mu03 - 3 * y_g * mu02 + 2 * y_g ** 3
#     central_moment_21 = mu21 - 2 * x_g * mu11 - y_g * mu20 + 2 * x_g ** 2 * mu01 - x_g * y_g ** 2
#     central_moment_12 = mu12 - 2 * y_g * mu11 - x_g * mu02 + 2 * y_g ** 2 * mu10 - x_g ** 2 * y_g


def find_subpixel_edges(img, cnt):
    subX = []
    subY = []
    for point in cnt:
        x, y = point[0]
        mm = cv2.moments(img[y-1:y+1, x-1:x+1])
        mu20 = mm["mu20"]
        mu02 = mm["mu02"]
        mu11 = mm["mu11"]

        if mu20 * mu02 != 0:
            k = (mu11 * mu11 - mu20 * mu02) / (mu20 * mu02)

            subx = (x + 0.5 * (-1) ** (k > 0) * abs(mu02) ** 0.5)
            suby = (y + 0.5 * (-1) ** (k > 0) * abs(mu20) ** 0.5)
            subX.append(subx)
            subY.append(suby)

    return subX, subY


def conv(picture, nsqure, j, i):
    M = np.array([[0, 0.00913767235, 0.021840193, 0.025674188, 0.021840193, 0.00913767235, 0],
                  [0.00913767235, 0.025951560, 0.025984481, 0.025984481, 0.025984481, 0.025951560, 0.00913767235],
                  [0.021840193, 0.025984481, 0.025984481, 0.025984481, 0.025984481, 0.025984481, 0.021840193],
                  [0.025674188, 0.025984481, 0.025984481, 0.025984481, 0.025984481, 0.025984481, 0.025674188],
                  [0.021840193, 0.025984481, 0.025984481, 0.025984481, 0.025984481, 0.025984481, 0.021840193],
                  [0.00913767235, 0.025951560, 0.025984481, 0.025984481, 0.025984481, 0.025951560, 0.00913767235],
                  [0, 0.00913767235, 0.021840193, 0.025674188, 0.021840193, 0.00913767235, 0]])
    # 卷积模板

    result = picture[j-3,i-3]**nsqure * M[0, 0] + picture[j-2,i-3]**nsqure * M[0, 1] + picture[j-1,i-3]**nsqure * M[0, 2] + picture[j,i-3]**nsqure * M[0, 3] + picture[j+1,i-3]**nsqure * M[0, 4] + picture[j+2,i-3]**nsqure * M[0, 5] + picture[j+3,i-3]**nsqure * M[0, 6] + \
            picture[j-3,i-2]**nsqure * M[1, 0] + picture[j-2,i-2]**nsqure * M[1, 1] + picture[j-1,i-2]**nsqure * M[1, 2] + picture[j,i-2]**nsqure * M[1, 3] + picture[j+1,i-2]**nsqure * M[1, 4] + picture[j+2,i-2]**nsqure * M[1, 5] + picture[j+3,i-2]**nsqure * M[1, 6] + \
            picture[j-3,i-1]**nsqure * M[2, 0] + picture[j-2,i-1]**nsqure * M[2, 1] + picture[j-1,i-1]**nsqure * M[2, 2] + picture[j,i-1]**nsqure * M[2, 3] + picture[j+1,i-1]**nsqure * M[2, 4] + picture[j+2,i-1]**nsqure * M[2, 5] + picture[j+3,i-1]**nsqure * M[2, 6] + \
            picture[j-3,i]**nsqure * M[3, 0] + picture[j-2,i]**nsqure * M[3, 1] + picture[j-1,i]**nsqure * M[3, 2] + picture[j,i]**nsqure * M[3, 3] + picture[j+1,i]**nsqure * M[3, 4] + picture[j+2,i]**nsqure * M[3, 5] + picture[j+3,i]**nsqure * M[3, 6] + \
            picture[j-3,i+1]**nsqure * M[4, 0] + picture[j-2,i+1]**nsqure * M[4, 1] + picture[j-1,i+1]**nsqure * M[4, 2] + picture[j,i+1]**nsqure * M[4, 3] + picture[j+1,i+1]**nsqure * M[4, 4] + picture[j+2,i+1]**nsqure * M[4, 5] + picture[j+3,i+1]**nsqure * M[4, 6] + \
            picture[j-3,i+2]**nsqure * M[5, 0] + picture[j-2,i+2]**nsqure * M[5, 1] + picture[j-1,i+2]**nsqure * M[5, 2] + picture[j,i+2]**nsqure * M[5, 3] + picture[j+1,i+2]**nsqure * M[5, 4] + picture[j+2,i+2]**nsqure * M[5, 5] + picture[j+3,i+2]**nsqure * M[5, 6] + \
            picture[j-3,i+3]**nsqure * M[6, 0] + picture[j-2,i+3]**nsqure * M[6, 1] + picture[j-1,i+3]**nsqure * M[6, 2] + picture[j,i+3]**nsqure * M[6, 3] + picture[j+1,i+3]**nsqure * M[6, 4] + picture[j+2,i+3]**nsqure * M[6, 5] + picture[j+3,i+3]**nsqure * M[6, 6]
    return result


def func(x, A):
    return x - 0.5 * np.sin(2 * x) - A * np.pi


def getSubPixels(cnt, img, point, radius):
    X, Y = point
    roi = img[Y-radius:Y+radius, X-radius:X+radius]
    moments = cv2.moments(roi)
    x_c = X-radius + moments['m10'] / moments['m00']
    y_c = Y-radius + moments['m01'] / moments['m00']
    subX, subY = [], []
    # for point in cnt:
    #     i,j = point[0]
    for j in range(Y-radius, Y+radius):
        for i in range(X-radius, X+radius):
            m1 = conv(img, 1, j, i)
            m2 = conv(img, 2, j, i)
            sigma = np.sqrt(m2 - m1**2)
            print(sigma)
            if sigma>2:
                m3 = conv(img, 3, j, i)
                s = (m3+2*m1**3-m1*m2*3) / (sigma**3)
                p1 = (1+s*np.sqrt(1.0/(4+s**2)))/2
                p2 = 1-p1
                h1 = m1-sigma*np.sqrt(p2/p1)
                h2 = m1+sigma*np.sqrt(p1/p2)
                # print(abs(h1-h2))
                if abs(h1-h2)>sigma*2:
                    A = min(p1,p2)
                    x = fsolve(func, 1.42, args=(A,))
                    rou = np.cos(x)
                    if rou <= 0.2*2/7:
                        sin_o = y_c / np.sqrt(x_c**2 + y_c**2)
                        cos_o = x_c / np.sqrt(x_c**2 + y_c**2)
                        subX.append(i+rou*cos_o*7/2)
                        subY.append(j+rou*sin_o*7/2)
    return subX, subY



def grayMoment(cnt, img):
    subpixel_x, subpixel_y = [], []
    window_size = 7
    for point in cnt:
        x, y = point[0]
        if (x < window_size or y < window_size or x >= img.shape[1] - window_size or y >= img.shape[0] - window_size):
            continue
        window = img[y - window_size:y + window_size + 1, x - window_size:x + window_size + 1].astype(np.float32)
        m00 = np.sum(window)
        m10 = np.sum(np.arange(window_size * 2 + 1) * np.ones_like(window[0]) * window)
        m01 = np.sum(
            np.transpose([np.arange(window_size * 2 + 1)] * (window_size * 2 + 1)) * np.ones_like(window) * window)
        m20 = np.sum(np.power(np.arange(window_size * 2 + 1), 2) * np.ones_like(window[0]) * window)
        m02 = np.sum(np.power(np.transpose([np.arange(window_size * 2 + 1)] * (window_size * 2 + 1)), 2) * np.ones_like(
            window) * window)
        m11 = np.sum(np.transpose([np.arange(window_size * 2 + 1)] * (window_size * 2 + 1)) * np.arange(
            window_size * 2 + 1) * window)

        try:
            x_c, y_c = np.linalg.inv([[m20, m11], [m11, m02]]).dot([m10, m01])
            print(x_c, y_c)
            subpixel_x.append(x + x_c)
            subpixel_y.append(y + y_c)
        except np.linalg.LinAlgError as e:
            print('Could not calculate for pixel', x, y)

    # subpixel_corners = np.asarray(subpixel_corners, dtype=np.float32)
    return subpixel_x, subpixel_y


def getGrayMoment(gray, edges):
    # 计算亚像素梯度
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 定义系数矩阵
    coef_mat = np.matrix([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1 / 3, 1 / 6, 1 / 3, 0], [0.25, 1 / 3, 0.25, 1 / 4]])
    subX, subY = [], []
    # 遍历每个边缘点
    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            if edges[y, x] > 0:
                # 计算灰度矩
                m = np.zeros((4, 1))
                for p in range(4):
                    for q in range(4 - p):
                        px = x + p - 1
                        qy = y + q - 1
                        m[p + q] += coef_mat[p, q] * gray[qy, px]

                # 计算亚像素梯度
                Gx = dx[y, x]
                Gy = dy[y, x]

                # 构造函数 F
                F = lambda dx, dy: np.dot(m.transpose(),
                                          np.power(np.array([dx ** i * dy ** (3 - i) for i in range(4)]), 2))

                # 最小二乘法求解最小值点
                A = np.array([[F(1, 0), F(0, 1)], [Gx, Gy]])
                b = np.array([-F(1, 0) + m[1], -Gx])
                d = np.linalg.solve(A.astype(np.float32), b.astype(np.float32))


                # 更新边缘点坐标
                # edges[y, x] = 0
                # if abs(d[0]) <= 0.5 and abs(d[1]) <= 0.5:
                if abs(d[0]) <= 1 and abs(d[1]) <= 1:
                    subX.append(x+d[0])
                    subY.append(y+d[1])
    return subX, subY


def cal_gaussian(Z, X):
    A = np.zeros((len(X), 3))
    for i in range(3):
        A[:, i] = np.power(X, i)
    A_T = A.T
    Z_T = Z.T
    ATA = np.dot(A_T, A)
    b = np.dot(A_T, Z_T)
    if np.linalg.det(ATA)==0:
        return 0
    else:
        B = np.dot(np.linalg.inv(ATA), b)

        BB = -1*B[1]/(2*B[2])
        if BB>3:
            BB = 3
        elif BB<0.0001:
            BB = 0.0001
        return BB



def gaussian_fit(p):
    x, y = p
    # theta = sobely[y][x] / sobelx[y][x]
    theta = grad_angle[y][x]
    step = np.pi / 8
    if -1*step <= theta < 1*step:
        y_bar = [0] * 5
        x_bar = [x for x in range(-2, 3)]
    elif step <= theta < 3*step:
        y_bar = [y for y in range(2, - 3, -1)]
        x_bar = [x for x in range(-2, 3)]
    elif theta >= 3*step or -3*step > theta:
        x_bar = [0] * 5
        y_bar = [y for y in range(2, -3, -1)]
    else:
        y_bar = [y for y in range(-2, 3)]
        x_bar = [x for x in range(-2, 3)]
    # if (-1 / 3) <= theta < 1 / 3:
    #     y_bar = [0] * 5
    #     x_bar = [x for x in range(-2, 3)]
    # elif 1 / 3 <= theta < 3:
    #     y_bar = [y for y in range(2, - 3, -1)]
    #     x_bar = [x for x in range(-2, 3)]
    # elif theta >= 3 or -3 > theta:
    #     x_bar = [0] * 5
    #     y_bar = [y for y in range(2, -3, -1)]
    # else:
    #     y_bar = [y for y in range(-2, 3)]
    #     x_bar = [x for x in range(-2, 3)]

    # 获取五组数据的梯度幅值
    mag_y = [dy[j+y][i+x] for j,i in list(zip(y_bar, x_bar))]
    mag_x = [dx[j+y][i+x] for j,i in list(zip(y_bar, x_bar))]

    # 排序
    # print("mag:", mag_x, mag_y)
    subx = cal_gaussian(np.array(mag_x, dtype=np.float32), x_bar)
    suby = cal_gaussian(np.array(mag_y, dtype=np.float32), y_bar)
    # print(subx, suby)
    return x+subx, y+suby

# 高斯权重函数
def gaussian_weight(d, sigma=0.5):
    return np.exp(-d**2 / (2 * sigma**2))

# 根据边缘梯度方向计算投影长度
def calc_projection_length(x, y, x0, y0, theta):
    dx = x - x0
    dy = y - y0
    return dx * np.cos(theta) + dy * np.sin(theta)


def unSharpMask(img, sigma, amount, thresh):
    """
    图像锐化:  增强图像边缘信息，计算公式：Y = X+λZ  X为源图像，Z为校正因子，此处为高通滤波后的图像，λ为缩放因子
    :param img: 源图像
    :param radius: 高斯内核
    :param amount: λ
    :param thresh:  阈值
    :return:
    """
    lImg = cv2.GaussianBlur(img, (0,0), sigma)
    hImg = cv2.subtract(img, lImg)

    mask = hImg > thresh
    newVal = img + mask*amount*hImg/100
    newImg = np.clip(newVal, 0, 255)
    newImg = newImg.astype(np.uint8)
    # h, w = img.shape
    # for i in range(0, h):
    #     for j in range(0, w):
    #         val = hImg[i][j]
    #         if  val> thresh:
    #             newVal = img[i][j] + amount*val/100
    #             img[i][j] = 0 if newVal < 0 else (255 if newVal > 255 else newVal)
    return newImg


def gaussian_surface_fit(point):
        x, y = point

        # 取3x3邻域内的梯度幅值
        neighbors = img[y-1:y+1, x-1:x+1]
        gx = sobelx[y-1:y+1, x-1: x+1]
        gy = sobely[y-1: y+1, x-1:x+1]


        # 构造雅可比矩阵
        jacobian = np.array([[gx, gy]], dtype=np.float32)

        # 进行高斯曲面拟合
        hessian = np.zeros((2, 2), dtype=np.float32)
        cv2.cornerEigenValsAndVecs(neighbors, 3, hessian, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        print(hessian)
        A, B, C = hessian[:, 0]
        params = np.array([A, B, C], dtype=np.float32)
        J_inv = np.linalg.pinv(jacobian)
        delta = -J_inv.dot(params)

        # 求解极值点
        return x+delta[0], y+delta[1]


def delBranch(edges_points, edges):
    noBranch = edges.copy()
    # dx = [1,-1, 0, 0, 1, 1,-1,-1]
    # dy = [0, 0, 1,-1, 1,-1, 1,-1]
    # d = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # for point in edges_points:
    #     i, j = point
    #     a, b = 0, 0
    #     for k in range(len(d)):
    #         tx, ty = i + d[k][0], j + d[k][1]
    #         if isOverImg(tx, ty, edges.shape[1], edges.shape[0]) and edges[ty][tx]:
    #             if k < 4:
    #                 a += 1
    #             else:
    #                 b += 1
    #     if (a>2) or (b>2):  # 存在分支
    #         noBranch[j][i] = 0
    ## 优化：使用numpy进行计算，减少了时间开销0.03秒
    dx = np.array([1, -1, 0, 0, 1, 1, -1, -1])
    dy = np.array([0, 0, 1, -1, 1, -1, 1, -1])
    moves = np.column_stack((dx, dy))
    for point in edges_points:
        i, j = point
        tx = i + moves[:, 0]
        ty = j + moves[:, 1]

        valid_moves = (tx >= 0) & (tx < edges.shape[1]) & (ty >= 0) & (ty < edges.shape[0]) & edges[ty, tx]

        a = np.count_nonzero(valid_moves[:4])
        b = np.count_nonzero(valid_moves[4:])

        if (a > 2) or (b > 2):
            noBranch[j][i] = 0

    return noBranch

def isOverImg(x, y, width, height):
    return x>0 and y>0 and x<width and y<height

def dfs(point, visited):
    x, y = point
    curve_point = [point]
    visited[y][x] = True
    dx = [1, -1, 0, 0, 1, 1, -1, -1]
    dy = [0, 0, 1, -1, 1, -1, 1, -1]
    for i in range(0, 8):
        nx, ny = x+dx[i], y+dy[i]
        if isOverImg(nx, ny, visited.shape[1], visited.shape[0]) and no_branch[ny][nx] and not visited[ny][nx]:
            next_point = dfs([nx, ny], visited)
            curve_point += next_point
        else:
            return curve_point

def getCurve1(noBranch):
    xs, ys = np.where(noBranch > 0)
    no_branch_points = [[x, y] for x, y in list(zip(ys, xs))]
    visited = np.zeros((noBranch.shape[0], noBranch.shape[1]))
    c_list = []
    for point in no_branch_points:
        x, y = point
        if visited[y][x]==0:
            c = dfs(point, visited)
            c_list.append(c)
    return c_list



def dfs1(point, visited, groups, group):
    """
    深度优先搜索（DFS）函数
    :param point: 当前搜索的起始点
    :param visited: 记录每个点是否被访问过的矩阵
    :param groups: 存储不同组的列表
    :param group: 当前正在处理的组
    """
    que = deque()
    que.append(point)
    while len(que)>0:
        p = que.popleft()
        x, y = p
        if visited[y][x] == 1:
            continue
        visited[y][x] = 1
        group.append(p)
        # 遍历所有8个相邻点
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                # 跳过当前点和越界的点
                if (i == x and j == y) or \
                        i < 0 or i >= len(visited[0]) or j < 0 or j >= len(visited):
                    continue
                # 如果该相邻点未访问且是同一组的点，则加入栈中
                if visited[j][i] == 0 and no_branch[j][i] > 0 :
                    que.append([i, j])

def group_points(noBranch):
    """
    将点集合按8邻域连通分成不同的组，为了能准确地进行圆弧分割，必须从边缘端点开始搜索(曲线点集必须从端点开始)
    :param points: 点集合，形如[[x1, y1], [x2, y2], ...]
    :return: 分组结果，形如[[[x1, y1], ...], [[x2, y2], ...], ...]
    """
    # visited = [[0]*len(points[0]) for _ in range(len(points))]
    xs, ys = np.where(noBranch > 0)
    no_branch_points = [[x, y] for x, y in list(zip(ys, xs))]
    visited = np.zeros(edges.shape)
    groups = []
    for p in no_branch_points:
        i, j = p
        # 检测该点是否边缘端点
        neibor = noBranch[j-1:j+2, i-1:i+2]
        if np.sum(neibor) == 510:    # 不是端点
            if visited[j][i] == 0:
                group = []
                dfs1(p, visited, groups, group)
                groups.append(group)

    return groups




def collectCourve(noBranch):
    xs, ys = np.where(noBranch>0)
    no_branch_points = [[x,y] for x,y in list(zip(ys, xs))]
    curve_num, labels = cv2.connectedComponents(noBranch)
    curve_dic = {}
    for point in no_branch_points:
        x, y = point
        curve_index = str(labels[y][x])
        if curve_index in curve_dic:
            curve_dic[curve_index].append([point])
        else:
            curve_dic[curve_index] = [[point]]

    return list(curve_dic.values())


def splitCurveToArc(curve):
    epsilon = 10
    approx = cv2.approxPolyDP(curve, epsilon, False)
    return approx

# def getArc(curves):
#     for curve in curves:
#         if len(curve)>30:   # 过滤掉长度小于30的曲线







imgName = "marker0-20"

img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)

# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# # 计算梯度幅值和方向
# grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
# grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# cv2.imwrite("../images/process/0428/process/"+imgName+'-grad_mag.bmp', grad_mag)
# _, sobel_thresh = cv2.threshold(grad_mag, 25,255, cv2.THRESH_BINARY)
# cv2.imwrite("../images/process/0428/process/"+imgName+'-grad_thresh.bmp', sobel_thresh)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
enhance = clahe.apply(img)
blurred = cv2.GaussianBlur(enhance, (0, 0), 10)
# usm = cv2.addWeighted(enhance, 1.8, blurred, -0.3, 0)
usm = unSharpMask(enhance, 10, 60, 0)
cv2.imwrite("../images/process/0428/process/" + imgName + "-usm.bmp", usm)
blurred1 = cv2.GaussianBlur(usm, (0, 0), 2)
cv2.imwrite("../images/process/0428/process/" + imgName + "-blurred1.bmp", blurred1)

sobelx = cv2.Sobel(blurred1, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred1, cv2.CV_64F, 0, 1, ksize=3)
dx = np.abs(sobelx)
dy = np.abs(sobely)
# # 计算梯度幅值和方向
grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
e = cv2.Canny(blurred1, 20, 60)
cv2.imwrite("../images/process/0428/process/" + imgName + "-edges.bmp", e)
e[e==255] = 1
skeleton0 = morphology.skeletonize(e)
edges = skeleton0.astype(np.uint8)*255
cv2.imwrite("../images/process/0428/process/" + imgName + "-refine.bmp", edges)
rows, cols = np.where(edges==255)   # 像素级边缘
edges_points = [[x,y] for x,y in list(zip(cols, rows))]

mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

## 圆弧分割算法
### 1 切断分支
time1 = time.time()
no_branch = delBranch(edges_points, edges)
time2 = time.time()
### 2 边缘点连接
# line_g = getCurve1(no_branch)
curve_list = group_points(no_branch)




# curve_list = collectCourve(no_branch)
# time3 = time.time()
# ### 3 将每一条边缘分割成不同的圆弧段
curves_arcs = []
z = 0
for curve in curve_list:
    if len(curve)>30:
        z += 1
        epsilon = 5
        angle_max = 0
        curve_arcs = []
        curve = np.array(curve)
        approx = cv2.approxPolyDP(curve, epsilon, False)
        print("approx数量:",len(approx))
        if len(approx)<3:
            curve_arcs.append(curve)
            print(curve_arcs)
            curves_arcs.append(curve_arcs)
            continue
        first = approx[0][0]
        start = 0
        end = np.where((curve == first).all(axis=1))[0][0]
        sub_arc = curve[start:end + 1]
        start = end + 1
        for index in range(1, len(approx) - 1):
            p_0 = np.array(approx[index - 1][0])
            p_1 = np.array(approx[index][0])
            p_2 = np.array(approx[index + 1][0])
            l_0 = p_0 - p_1
            l_1 = p_1 - p_2
            angle = np.dot(l_0, l_1) / (np.linalg.norm(l_0) * np.linalg.norm(l_1))
            angle = np.degrees(np.arccos(angle))
            end = np.where((curve == p_1).all(axis=1))[0][0]
            if len(sub_arc):
                sub_arc = np.concatenate((sub_arc, curve[start:end + 1]))
            else:
                sub_arc = curve[start:end + 1]
            start = end + 1
            if angle > 60:
                curve_arcs.append(sub_arc)
                sub_arc = []
            if index == len(approx) - 2:
                end = np.where((curve == p_2).all(axis=1))[0][0]
                if len(sub_arc):
                    sub_arc = np.concatenate((sub_arc, curve[start:end + 1]))
                else:
                    sub_arc = curve[start:end + 1]
                start = end + 1
        if start < len(curve):
            sub_arc = np.concatenate((sub_arc, curve[start:len(curve)]))
        curve_arcs.append(sub_arc)
        curves_arcs.append(curve_arcs)
print(len(curves_arcs))
print("长度大于30的曲线数量：", z)
#### approx多边形逼近顶点可视化
# for i in range(0, len(approx)-1):
#     p1 = approx[i][0]
#     p2 = approx[i+1][0]
#     cv2.line(mask1, p1, p2, (0,255,0), 1)
#     mask1[p1[1],p1[0],:] = (255,0,0)
#     cv2.putText(mask1, str(i), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
#     if i==len(approx)-2:
#         cv2.putText(mask1, str(i+1), (p2[0], p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
#         mask1[p2[1], p2[0], :] = (255, 0, 0)

#####圆弧过滤与分割可视化
for curve_arcs in curves_arcs:
    arc_nums = len(curve_arcs)

    # 创建同样数量的颜色表
    colors = []
    for i in range(arc_nums):
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

    for arc_index in range(arc_nums):
        color = colors[arc_index]
        arc = curve_arcs[arc_index]
        for point in arc:
            i, j = point
            mask1[j,i,:] = color

cv2.imwrite("../images/process/0428/process/" + imgName + "-all_sub_arc.bmp", mask1)







# print(len(curve_arc))


# 在原图上标记连通域并可视化结果
# w, h = img.shape[0], img.shape[1]
# result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# p_sum = 0
# for index in range(0, len(curve_arc)):
#
#     color = colors[index]
#     line = curve_arc[index]
#     p_sum += len(line)
#     print(index, line)
#     for p in line:
#         i, j = p
#         result[j,i,:] = color
# cv2.imwrite("../images/process/0428/process/" + imgName + "-sub_arc.bmp", result)
# print(len(curve), p_sum)
#
#
#         # angle阈值
#         if angle_max < 60: # 不存在尖角
#             # 进行椭圆拟合
#

#
#
# cv2.imwrite("../images/process/0428/process/" + imgName + "-arc.bmp", mask1)




# for index in range(0, len(circleCnts)):
#     cnt = circleCnts[index]
#     arc = cv2.arcLength(cnt, False)
#     epsilon = 20
#     approx = cv2.approxPolyDP(cnt, epsilon, False)
#     cv2.polylines(mask1, [approx], isClosed=True, color=(0, 255, 0), thickness=1)
#
#     # print(approx)
#
# cv2.imwrite("../images/process/0428/process/" + imgName + "-approx.bmp", mask1)
# # 计算高斯滤波器大小
# ksize = int(6 * 1.4 + 1)
#
# # 使用cv2.GaussianBlur()函数进行高斯滤波
# blur = cv2.GaussianBlur(img, (0, 0), 0.5)
# # blur = cv2.GaussianBlur(img, (3,3), 0.2)
# cv2.imwrite("../images/process/0428/process/"+imgName+"-blur.bmp", blur)
# edges = cv2.Canny(blur, 40, 120)
# cv2.imwrite("../images/process/0428/process/"+imgName+"-edges.bmp", edges)
#
#
# sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
# dx = np.abs(sobelx)
# dy = np.abs(sobely)
# # # 计算梯度幅值和方向
# grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
#
# rows, cols = np.where(edges==255)   # 像素级边缘
# edges_points = [[x,y] for x,y in list(zip(cols, rows))]

# e_x, e_y = [], []
# time1 = time.time()
# circleCnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_TREE
# sub_x = []
# sub_y = []
# sub_points = []
# for point in edges_points:
#     subx, suby = gaussian_fit(point)
#     # print("sub:", subx, suby)
#     sub_x.append(subx)
#     sub_y.append(suby)

# 画布
# mask1 = np.zeros(edges.shape[:2], dtype=np.uint8)
# mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# #
# for index in range(len(circleCnts)):
# #     canvas = np.zeros(edges.shape[:2], dtype=np.uint8)
#     cnt = circleCnts[index]
#     length = cv2.arcLength(cnt, False)
#     if length > 50: # 过滤掉噪声
#         area = cv2.contourArea(cnt, False)
#         (x,y), (a,b), angle = cv2.fitEllipse(cnt)
#         cv2.putText(mask1, str(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
#         cv2.ellipse(mask1, (int(x), int(y)), (int(a/2), int(b/2)), angle, 0, 360, (0,0,255), 1)
#         x, y = int(x), int(y)
#         if abs(a-b)<10 and min(a,b)>20:
#             sub_points = []
#             for point in cnt:
#                 p = point[0]
#                 e_x.append(p[0])
#                 e_y.append(p[1])
            #     subx, suby = gaussian_fit(p)
            #     # subx, suby = gaussian_surface_fit(p)
            #     sub_points.append([subx, suby])
            #     sub_x.append(subx)
            #     sub_y.append(suby)
            # sub_points = np.array(sub_points, dtype=np.float32)
            # (sub_center_x, sub_center_y), (sub_a, sub_b), sub_angle = cv2.fitEllipse(sub_points)
            # print(index, sub_center_x, sub_center_y)

            # x_p, y_p = grayMoment(cnt, img)
            # sub_x.extend(x_p)
            # sub_y.extend(y_p)
#
#             #
#             # print(index, abs(a - b), min(a,b))
#             # cv2.ellipse(mask1, (int(x), int(y)), (int(a/2), int(b/2)), angle, 0, 360, (0,0,255), 1)
#             # cv2.putText(mask1, str(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#             radius = int(max(a,b))
#             roi = blur[y-radius: y+radius, x-radius:x+radius]
#             cnt_sub_x, cnt_sub_y = zernike_detection(roi, cnt)
#             cnt_sub_x = list(map(lambda i: i + (x-radius), cnt_sub_x))
#             cnt_sub_y = list(map(lambda j: j + (y-radius), cnt_sub_y))
#             sub_x.extend(cnt_sub_x)
#             sub_y.extend(cnt_sub_y)
            # moments = []
            # for n in range(8):
            #     for m in range(-n, n+1, 2):
            #         moment = zernike_moment(roi, n, m, radius, radius, radius)
            #         moments.append((n, m, moment))
            # moments = sorted(moments, key=lambda x: abs(x[2]), reverse=True)
            # n, m, moment = moments[0]
            # derivative = -m * moment.real / moment.imag
            # x1 = x + radius * np.cos(-derivative)
            # y1 = y + radius * np.sin(-derivative)
            # x2 = x + radius * np.cos(np.pi - derivative)
            # y2 = y + radius * np.sin(np.pi - derivative)
            # print((x1,y1), (x2,y2))
            # sub_x.append(x1)
            # sub_x.append(x2)
            # sub_y.append(y1)
            # sub_y.append(y2)
            # print(index)

# cv2.imwrite("../images/process/0428/process/"+imgName+"-circle.bmp", mask1)

# time2 = time.time()
# print(time2-time1)
# # showPic('edges', mask1)
#
# plt.imshow(img, cmap="gray")
# plt.scatter(cols, rows, s=10, marker="*")
# plt.scatter(e_x, e_y, s=10, marker="*")
# # # plt.scatter(subpixel_x, subpixel_y, s=10, marker="*")
# # # print("edges_points:", len(edges_points))
# # # print(len(sub_x), len(sub_y), min(sub_x), max(sub_x), min(sub_y), max(sub_y))
# # plt.scatter(sub_x, sub_y, s=10, marker="*", c="red")
# plt.show()

#
# rows, cols = np.where(edges==255)
# edge_points = list(zip(rows, cols))
#
# circleCnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
#
#
# sub_x = []
# sub_y = []
# time1 = time.time()
# # subpixel_x, subpixel_y = zernike_detection(edges)
#
# for cnt in circleCnts:
#     arr = cnt.reshape((len(cnt), 2))
#     tck, u = splprep(arr.T, k=3, s=0)
#     u_new = np.linspace(u.min(), u.max(), 3*len(arr))
#     x_new, y_new = splev(u_new, tck, der=0)
#     sub_x.extend(x_new)
#     sub_y.extend(y_new)
# time2 = time.time()
# print("Zernike矩亚像素边缘检测花费时间为：",time2-time1)
#
# plt.imshow(img, cmap="gray")
# plt.scatter(cols, rows, s=10, marker="*")
# # plt.scatter(subpixel_x, subpixel_y, s=10, marker="*")
# plt.scatter(sub_x, sub_y, s=10, marker="*")
# plt.show()




