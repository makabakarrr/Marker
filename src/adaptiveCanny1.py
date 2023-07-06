import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

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

    return newImg


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

imgName = "marker_0"

img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)

### 分析局部直方图均衡化、滤波、usm对canny的影响
## 保持计算canny阈值的pecent不变
# clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
# enhance = clahe.apply(img)
# blurred = cv2.GaussianBlur(img, (0, 0), 1)
# cv2.imwrite("../images/process/0428/adaptiveCanny1/" + imgName + "_only-gaussian1.bmp", blurred)
# sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
# grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
#
# time1 = time.time()
# th, tl = adaptiveThreshold(0.001)
# time2 = time.time()
# print("计算阈值共耗时：", time2-time1)
# print("计算出的阈值为：", th, tl)
#
# edges = cv2.Canny(blurred, tl, th)
# cv2.imwrite("../images/process/0428/adaptiveCanny1/" + imgName + "_only-gaussian-edges1.bmp", edges)
# rows, cols = np.where(edges==255)   # 像素级边缘
# edges_points = [[x,y] for x,y in list(zip(cols, rows))]
# res = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
# for point in edges_points:
#     x,y = point
#     res[y,x,:] = (0,0,255)
# cv2.imwrite("../images/process/0428/adaptiveCanny1/" + imgName + "_only-gaussian_edges-ori1.png", res)
# plt.hist(grad_mag.ravel(), 256)
# plt.show()

# blurred = cv2.GaussianBlur(img, (0, 0), 1)
# blurred = cv2.bilateralFilter(img, 7, 7,7)
# cv2.imwrite("../images/process/0428/adaptiveCanny1/" + imgName + "_only-bilateral2.bmp", blurred)
# sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
# grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
#
# time1 = time.time()
# th, tl = adaptiveThreshold(0.001)
# time2 = time.time()
# print("计算阈值共耗时：", time2-time1)
# print("计算出的阈值为：", th, tl)
#
# edges = cv2.Canny(blurred, tl, th)
# cv2.imwrite("../images/process/0428/adaptiveCanny1/" + imgName + "_only-bilateral-edges2.bmp", edges)
# rows, cols = np.where(edges==255)   # 像素级边缘
# edges_points = [[x,y] for x,y in list(zip(cols, rows))]
# res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for point in edges_points:
#     x,y = point
#     res[y,x,:] = (0,0,255)
# cv2.imwrite("../images/process/0428/adaptiveCanny1/" + imgName + "_only-bilateral-ori2.png", res)
# plt.hist(grad_mag.ravel(), 256)
# plt.show()
