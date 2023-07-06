import cv2
import numpy as np
import time




def non_maximum_suppression(gradient, magnitude):
    height, width = magnitude.shape
    nms_magnitude = np.zeros((height, width))
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255
            qq, rr = 255,255
            if gradient[i][j] == 0:
                q = magnitude[i][j + 1]
                r = magnitude[i][j - 1]
            elif gradient[i][j] == 90:
                q = magnitude[i + 1][j]
                r = magnitude[i - 1][j]
            elif gradient[i][j] == 45:
                q = magnitude[i + 1][j + 1]
                r = magnitude[i - 1][j - 1]
            elif gradient[i][j] == 135:
                q = magnitude[i + 1][j - 1]
                r = magnitude[i - 1][j + 1]

            if magnitude[i][j] > q and magnitude[i][j] > r:
                nms_magnitude[i][j] = magnitude[i][j]
            else:
                nms_magnitude[i][j] = 0

    return nms_magnitude


def double_threshold(suppress, low_threshold, high_threshold):
    height, width = suppress.shape
    result = np.zeros((height, width), dtype=np.uint8)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(suppress >= high_threshold)
    zeros_i, zeros_j = np.where(suppress < low_threshold)

    weak_i, weak_j = np.where((suppress <= high_threshold) & (suppress >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    result[zeros_i, zeros_j] = 0

    res = hysteresis(result, weak, strong)

    return res

def hysteresis(img, weak, strong):
    height, width = img.shape
    _edge = np.zeros((height + 2, width + 2), dtype=np.uint8)
    _edge[1: height + 1, 1: width + 1] = img
    nn = np.array(((1, 1, 1), (1, 0, 1), (1, 1, 1)), dtype=np.uint8)
    for i in range(1, height+2):
        for j in range(1, width+2):
            if _edge[i, j] < weak or _edge[i,j]>strong:
                continue
            if np.max(_edge[i - 1:i + 2, j - 1:j + 2]*nn) >= strong:
                _edge[i,j] = 255
            else:
                _edge[i][j] = 0
    return _edge[1:height+1, 1:width+1]


def angle_quantization(angle):
     angle = angle / np.pi * 180
     angle[angle < -22.5] = 180 + angle[angle < -22.5]
     _angle = np.zeros_like(angle, dtype=np.uint8)
     _angle[np.where(angle <= 22.5)] = 0
     _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
     _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
     _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135
     _angle[np.where((angle > 157.5) & (angle <= 202.5))] = 180

     return _angle


def manualCanny(img, derta, tl, th):
    ## 1.滤波
    blur = cv2.GaussianBlur(img, (5,5), derta)
    cv2.imwrite("../images/process/0428/manualCanny/" + imgName + "-manual_blur.png", blur)
    ## 2.求梯度
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    # 梯度幅值
    # magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = abs(sobelx) + abs(sobely)
    magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
    # 梯度方向
    gradient = np.arctan2(sobely, sobelx)
    gradient = angle_quantization(gradient)
    ## 3.非极大值抑制
    suppress = non_maximum_suppression(gradient, np.array(magnitude))
    #### 计算阈值
    # th, tl = adaptiveThreshold(0.001, magnitude)

    ## 4.双阈值处理+边缘连接
    result = double_threshold(suppress, tl, th)

    return result


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
blur = cv2.GaussianBlur(img, (0,0), 1)

### 边缘提取
# res = manualCanny(img, 1.4, 30, 60)
# time2 = time.time()
# print(time2-time1)
# cv2.imwrite("../images/process/0428/manualCanny/" + imgName + "-ratio_edges.png", res)
## 使用比例计算阈值
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
grad_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# 使用比例计算阈值
# th, tl = adaptiveThreshold(0.0005, mag)
# 使用ostu计算阈值
th, _ = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
tl = max(0, int(th * 0.3))
time1 = time.time()
edges = manualCanny(img, 2, tl, th)
time2 = time.time()
print(time2-time1, th, tl)
# edges = cv2.Canny(blur, tl,th)
# cv2.imwrite("../images/process/0428/manualCanny/" + imgName + "-auto_gaussian_ostu_edges.png", edges)
cv2.imwrite("../images/process/0428/manualCanny/" + imgName + "-test_constrain.png", edges)

