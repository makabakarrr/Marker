import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

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


def adaptiveThreshold(pecent_of_edge_pixel):
    h, w = grad_mag.shape[0], grad_mag.shape[1]
    target = w*h*pecent_of_edge_pixel
    total = 0
    thresh_high = np.max(grad_mag)
    while total<target:
        thresh_high -= 1
        total = np.count_nonzero(grad_mag >= thresh_high)
    thresh_low = 0.3*thresh_high
    return thresh_high, thresh_low


# img1 = cv2.imread("../images/process/0428/marker_0-ad7.png", 0)
# img = cv2.imread("../images/process/0428/marker_0.png", 0)
# sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
# grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
# th, tl = adaptiveThreshold(0.001)
# edges = cv2.Canny(img1, tl, th)
# rows, cols = np.where(edges==255)   # 像素级边缘
# edges_points = [[x,y] for x,y in list(zip(cols, rows))]
# res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for point in edges_points:
#     x,y = point
#     res[y,x,:] = (0,0,255)
# cv2.imwrite("../images/process/0428/adaptiveCanny/marker_0_ADfilter_edges7.png", res)



imgName = "marker0-20"

img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
enhance = clahe.apply(img)
blurred = cv2.GaussianBlur(enhance, (0, 0), 2)
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-ori-gaussian.bmp", blurred)

sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

time1 = time.time()
th, tl = adaptiveThreshold(0.008)
time2 = time.time()
print(th, tl)
print(time2-time1)
edges = cv2.Canny(blurred, tl, th)
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-edges.bmp", edges)
plt.hist(grad_mag.ravel(), 256)
plt.show()
# rows, cols = np.where(edges==255)   # 像素级边缘
# edges_points = [[x,y] for x,y in list(zip(cols, rows))]
# res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for point in edges_points:
#     x,y = point
#     res[y,x,:] = (0,0,255)
# cv2.imwrite("../images/process/0428/adaptiveCanny/marker0-20_gaussian_edges1.png", res)
sys.exit()

sub_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

cnt, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
curves_arcs = []
for curve in cnt:
    if len(curve)>30:
        epsilon = 2
        angle_max = 0
        curve_arcs = []
        curve = np.array(curve)
        approx = cv2.approxPolyDP(curve, epsilon, False)
        ### approx可视化
        colors = []
        for i in range(len(approx)):
            colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        for ap_index in range(len(approx)):
            i, j = approx[ap_index][0]
            sub_mask[j,i,:] = colors[ap_index]


        if len(approx)<3:
            curve_arcs.append(curve)
            curves_arcs.append(curve_arcs)
            continue
        first = approx[0][0]
        start = 0
        end = np.where((np.squeeze(curve) == first).all(axis=1))[0][0]
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
            end = np.where((np.squeeze(curve) == p_1).all(axis=1))[0][0]
            if len(sub_arc):
                sub_arc = np.concatenate((sub_arc, curve[start:end + 1]))
            else:
                sub_arc = curve[start:end + 1]
            start = end + 1
            if angle > 50:
                curve_arcs.append(sub_arc)
                sub_arc = []
            if index == len(approx) - 2:
                end = np.where((np.squeeze(curve) == p_2).all(axis=1))[0][0]
                if len(sub_arc):
                    sub_arc = np.concatenate((sub_arc, curve[start:end + 1]))
                else:
                    sub_arc = curve[start:end + 1]
                start = end + 1
        if start < len(curve):
            sub_arc = np.concatenate((sub_arc, curve[start:len(curve)]))
        curve_arcs.append(sub_arc)
        curves_arcs.append(curve_arcs)

####圆弧过滤与分割可视化
for curve_arcs in curves_arcs:
    arc_nums = len(curve_arcs)

    # 创建同样数量的颜色表
    colors = []
    for i in range(arc_nums):
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

    for arc_index in range(arc_nums):
        color = colors[arc_index]
        arc = curve_arcs[arc_index]
        x, y, w, h = cv2.boundingRect(arc)
        if w<5 or h <5:
            continue
        for point in arc:
            i, j = point[0]
            mask1[j,i,:] = color
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-merge_sub_arc.bmp", mask1)
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-all_sub_arc.bmp", sub_mask)

## 对圆弧进行椭圆拟合
mask2 = mask1.copy()
center_arr = []
for curve_arcs in curves_arcs:
    for sub_arc in curve_arcs:
        _, _, w, h = cv2.boundingRect(sub_arc)
        if w<3 or h<3 or len(sub_arc)<5:  # 过滤掉
            continue
        (x,y), (a,b), angle = cv2.fitEllipse(sub_arc)
        # if abs(angle-90)<10 or angle%90<10:
        if abs(a-b)<15 and min(a,b)>20:
            # 根据内点数量进行二次筛选
            # 求内点数量
            k=0
            for point in sub_arc:
                x0, y0 = point[0]
                dist = np.sqrt((x0-x)**2+(y0-y)**2)
                r = (a+b)/4
                if r-2<dist<r+2:
                    k+=1
            # 内点数量小于圆弧点集长度的2/3
            if 2/3<(k/len(sub_arc)):
                cv2.ellipse(mask2, (int(x), int(y)), (int(a/2), int(b/2)), angle, 0, 360, (0,255,0), 1)
                center_arr.append([x,y])
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-fit_ellipse.bmp", mask2)
rows, cols = np.array(center_arr)[:,0], np.array(center_arr)[:, 1]

plt.imshow(img, cmap="gray")
plt.scatter(rows, cols, s=10, marker="*")
plt.show()


