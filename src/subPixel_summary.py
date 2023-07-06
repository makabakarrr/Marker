import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

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

def gaussian_fit(p, sobelx, sobely):
    x, y = p
    dx = abs(sobelx)
    dy = abs(sobely)
    # theta = sobely[y][x] / sobelx[y][x]
    theta = angle[y][x]
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

    # 获取五组数据的梯度幅值
    mag_y = [dy[j+y][i+x] for j,i in list(zip(y_bar, x_bar))]
    mag_x = [dx[j+y][i+x] for j,i in list(zip(y_bar, x_bar))]

    # 排序
    # print("mag:", mag_x, mag_y)
    subx = cal_gaussian(np.array(mag_x, dtype=np.float32), x_bar)
    suby = cal_gaussian(np.array(mag_y, dtype=np.float32), y_bar)
    # print(subx, suby)
    return x+subx, y+suby


def delBranch(edge_points, edges):
    noBranch = edges.copy()
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

# imgName = "marker0-20"
# imgName = "marker_0-1800"
# imgName = "marker_1-1000"
# imgName = "marker_1-1200"
# imgName = "marker_1-1500"
# imgName = "marker_1-1800"
# imgName = "marker_5-1000"
# imgName = "marker-6-1"
# imgName = "marker-6-4"
# imgName = "marker_1-2"
# imgName = "0-100-38-7"
# imgName = "pic-32-3"
img_list = ["marker0-20", "marker_0-1800", "marker_1-1000", "marker_1-1200", "marker_1-1500", "pic-32-3",
            "marker_1-1800", "marker_5-1000", "marker-6-1", "marker-6-4", "marker_1-2", "0-100-9", "0-100-38-7"]

for name in img_list:
    img = cv2.imread('../images/process/summary/img/'+name+'.png', 0)
    blur = cv2.GaussianBlur(img, (0,0), 1)
    cv2.imwrite("../images/process/summary/gaussianBlur/" + name + '-blur.png', blur)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    sobely= cv2.Sobel(blur, cv2.CV_64F, 0, 1)

    mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
    grad_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    th, _ = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(th)
    tl = max(0, int(th * 0.3))
    edges = cv2.Canny(blur, tl,th)
    rows, cols = np.where(edges==255)
    edges_points = [[x,y] for x,y in list(zip(cols, rows))]
    cv2.imwrite("../images/process/summary/gaussianOstuCanny/"+name+'-edges.png', edges)
    mask1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in edges_points:
        x, y = point
        mask1[y,x,:] = (0,0,255)
    cv2.imwrite("../images/process/summary/gaussianOstuCanny/00-" + name + '-mask.png', mask1)


# 对轮廓进行分类，将属于同一个圆的轮廓分为同一组
# exits_circle = []
# cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# for index in range(len(cnts)):
#     cnt = cnts[index]
#     if len(cnt)>20:
#         (cx, cy), (a, b), angle = cv2.fitEllipse(cnt)
#         if max(a,b)-min(a,b)<5:
#             exits_circle.append([cx, cy, r])

## 过滤掉伪边缘
# cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# subX, subY = [], []
# mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# filter_cnts = np.zeros(edges.shape)
# for index in range(len(cnts)):
#     cnt = cnts[index]
#     if len(cnt)>20:
#         (cx, cy), (a, b), angle = cv2.fitEllipse(cnt)
#         d = max(a,b)
#         # 求内点数量
#         perimeter = np.pi * d
#         nums = 0
#         for point in cnt:
#             x,y = point[0]
#             dist = abs(np.sqrt((x-cx)**2 + (y-cy)**2)-d/2)
#             if dist <= 1:
#                 nums += 1
#         k = nums / perimeter
#         if k > 0.8:
#             # cv2.circle(mask1, (int(cx), int(cy)), int(d / 2), (0, 0, 255), 1)
#             cv2.drawContours(filter_cnts, cnt, -1, (255,255,255), 1)
# cv2.imwrite("../images/process/0428/filterCnts/"+imgName+'-filter_cnts_result.png', filter_cnts)

# for point in edges_points:
#     subx, suby = gaussian_fit(point, sobelx, sobely)
#     subX.append(subx)
#     subY.append(subY)

# time2 = time.time()
# print(time2-time1)
# plt.show(img, cmap="gray")
# plt.scatter(cols, rows, s=10, marker="*")
# plt.scatter(subX, subY, s=10, marker="*")
# plt.show()

