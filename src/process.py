import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import time


def getInertiaMean(img):
    # 计算灰度共生矩阵及对应惯性矩
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    inertias = []
    for d in distances:
        for a in angles:
            glcm = greycomatrix(img, [d], [a], symmetric=True, normed=True)
            inertia = greycoprops(glcm, 'dissimilarity')[0, 0]
            inertias.append(inertia)
    # 计算12个方向得惯性矩均值
    inertia_mean = np.mean(inertias)
    return inertia_mean

def adaptiveThreshold(pecent_of_edge_pixel, grad_mag):
    h, w = grad_mag.shape[0], grad_mag.shape[1]
    target = w*h*pecent_of_edge_pixel
    total = 0
    thresh_high = np.max(grad_mag)
    while total<target:
        thresh_high -= 1
        total = np.count_nonzero(grad_mag >= thresh_high)
    thresh_low = 0.3*thresh_high
    return thresh_high, thresh_low

#### 基于灰度共生矩阵与阈值的关系计算阈值
# imgName = "marker0-20"
#
# img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
# aveF = getInertiaMean(img)
# derta = 0.8 + np.log(aveF+1)/10
# Th = 3*(np.log(aveF+1)+0.5)
# print(derta, Th)
# blur = cv2.GaussianBlur(img, (3,3), derta)
# edges = cv2.Canny(blur, Th, 2*Th)
# cv2.imwrite("../images/process/0428/glcm/" + imgName + "_blur.png", blur)
# cv2.imwrite("../images/process/0428/glcm/" + imgName + "_edges.png", edges)

# imgName = "marker0-20"
#
# img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
#
# blur = cv2.GaussianBlur(img, (0,0), 1)
# Th = 30
# edges = cv2.Canny(blur, Th, 3*Th)
# # cv2.imwrite("../images/process/0428/glcm/" + imgName + "_blur.png", blur)
# # cv2.imwrite("../images/process/0428/glcm/" + imgName + "_edges.png", edges)
# aveF = getInertiaMean(edges)
# print(aveF)

imgName = "marker0-20"

img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
blur = cv2.GaussianBlur(img, (0,0), 1)
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
grad_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# 计算类间方差最大化的阈值
threshold, _ = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th_low = max(0, int(threshold * 0.2))

print(th_low, threshold)
# 应用Canny算子
edges = cv2.Canny(blur, th_low, threshold)
cv2.imwrite("../images/process/0428/glcm/" + imgName + "_ostu-edges.png", edges)
# 显示结果
# cv2.imshow('Original', img)
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


