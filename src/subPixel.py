import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from customFunction import showPic
from recognition import recognition
from scipy.interpolate import splprep, splev
from scipy.special import comb
from scipy.optimize import fsolve


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


imgName = "marker_7"

img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)

# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# # 计算梯度幅值和方向
# grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
# grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# cv2.imwrite("../images/process/0428/process/"+imgName+'-grad_mag.bmp', grad_mag)
# _, sobel_thresh = cv2.threshold(grad_mag, 25,255, cv2.THRESH_BINARY)
# cv2.imwrite("../images/process/0428/process/"+imgName+'-grad_thresh.bmp', sobel_thresh)

edges = cv2.Canny(img, 40, 120)
cv2.imwrite("../images/process/0428/process/"+imgName+"-edges.bmp", edges)
blur = cv2.GaussianBlur(img, (5,5), 0.2)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# # 计算梯度幅值和方向
grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

rows, cols = np.where(edges==255)   # 像素级边缘

time1 = time.time()
circleCnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
sub_x = []
sub_y = []
sub_points = []
# 画布
# mask1 = np.zeros(edges.shape[:2], dtype=np.uint8)
mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
for index in range(len(circleCnts)):
    canvas = np.zeros(edges.shape[:2], dtype=np.uint8)
    cnt = circleCnts[index]
    length = cv2.arcLength(cnt, False)
    if length > 50: # 过滤掉噪声
        area = cv2.contourArea(cnt, False)
        (x,y), (a,b), angle = cv2.fitEllipse(cnt)
        x, y = int(x), int(y)
        if abs(a-b)<10 and min(a,b)>20:
            # x_p, y_p = grayMoment(cnt, img)
            # sub_x.extend(x_p)
            # sub_y.extend(y_p)

            #
            # print(index, abs(a - b), min(a,b))
            # cv2.ellipse(mask1, (int(x), int(y)), (int(a/2), int(b/2)), angle, 0, 360, (0,0,255), 1)
            # cv2.putText(mask1, str(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            radius = int(max(a,b))
            roi = blur[y-radius: y+radius, x-radius:x+radius]
            cnt_sub_x, cnt_sub_y = zernike_detection(roi, cnt)
            cnt_sub_x = list(map(lambda i: i + (x-radius), cnt_sub_x))
            cnt_sub_y = list(map(lambda j: j + (y-radius), cnt_sub_y))
            sub_x.extend(cnt_sub_x)
            sub_y.extend(cnt_sub_y)
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

time2 = time.time()
print(time2-time1)

plt.imshow(img, cmap="gray")
plt.scatter(cols, rows, s=10, marker="*")
# plt.scatter(subpixel_x, subpixel_y, s=10, marker="*")

print(len(sub_x), len(sub_y), sub_x[0], sub_y[0])
plt.scatter(sub_x, sub_y, s=10, marker="*", c="red")
plt.show()

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




