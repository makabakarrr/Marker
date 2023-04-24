import cv2
import numpy as np
import math

from customFunction import showPic
from recognition import recognition
import matplotlib.pyplot as plt

from skimage import morphology
from collections import deque

# 根据梯度方向获取下一个边缘点
def get_next_edge_point(point, angle):
    k1 = np.array([np.cos(np.radians(angle + 90)), np.sin(np.radians(angle + 90))])
    k2 = np.array([np.cos(np.radians(angle - 90)), np.sin(np.radians(angle - 90))])
    next_point1 = np.round(np.array(point) + k1).astype(int)
    next_point2 = np.round(np.array(point) + k2).astype(int)
    return [next_point1, next_point2]

def get_next_edge_point1(point, gradient_angle):
    x, y = point
    basic_direction = round(gradient_angle / 45.0) % 4
    if basic_direction == 0:
        return (x, y + 1) if refine[x, y + 1] < 255 else (x, y - 1)
    elif basic_direction == 1:
        return (x - 1, y - 1) if refine[x - 1, y - 1] < 255 else (x + 1, y + 1)
    elif basic_direction == 2:
        return (x + 1, y) if refine[x + 1, y] < 255 else (x - 1, y)
    elif basic_direction == 3:
        return (x + 1, y - 1) if refine[x + 1, y - 1] < 255 else (x + 1, y - 1)



def detectEndPoints(cnt):
    que = deque()
    for point in cnt:
        x, y = point[0]
        neighborhood = refine[y - 1:y + 2, x - 1:x + 2]
        if np.sum(neighborhood) == 510:  # 3*3窗口内只有一个相邻像素为非零像素
            que.append([x, y])
    return que


def connectEdges(que, image):
    while len(que) > 0:
        point = que.popleft()  # 取出端点
        center_x, center_y = point
        cv2.circle(mask_canvas, (center_x, center_y), 1, (0,0,255),-1)
        next_points = get_next_edge_point(point, grad_angle[point[1]][point[0]])
        x1, y1 = next_points[1][0], next_points[1][1]
        next_point = next_points[0] if image[y1][x1] > 0 else next_points[1]
        x, y = next_point

        if not np.array_equal(point, next_point) and sobel_thresh[y][x] == 255 and image[y][x] == 0:
            image[y][x] = 255
            que.append(next_point)

def region_growing(img, seed):
    """
    :param img: 要处理的单通道图像
    :param seed: 包含种子点的二元组 (x,y) 坐标。从这个点开始向外扩散。
    :return: 区域生长后的掩码图像
    """

    # 创建空掩码图像，初始化为全零
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # 参数设定
    connectivity = 4  # 连通性
    flags = connectivity
    lo_diff = 3  # 差异阈值loDiff
    up_diff = 3  # 差异阈值upDiff

    # 生长区域
    cv2.floodFill(img, mask, seed, newVal=255, loDiff=lo_diff, upDiff=up_diff, flags=flags)

    # 返回掩码图像
    return mask[1:-1, 1:-1]


imgName = "1000/marker_03-1"

img = cv2.imread('../images/process/marker/'+imgName+'.bmp', 0)
## 图像增强
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
enhance = clahe.apply(img)
# cv2.imwrite("./images/process/0411-pm/cal-edges/" + imgName + "-enhance.bmp", enhance)

blurred = cv2.GaussianBlur(enhance, (0, 0), 1)
usm = cv2.addWeighted(enhance, 1.8, blurred, -0.3, 0)
# cv2.imwrite("./images/process/0411-pm/cal-edges/" + imgName + "-usm.bmp", usm)
blurred1 = cv2.GaussianBlur(usm, (0, 0), 1)

sobelx = cv2.Sobel(blurred1, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred1, cv2.CV_64F, 0, 1, ksize=3)
# # 计算梯度幅值和方向
grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
_, sobel_thresh = cv2.threshold(grad_mag, 20,255, cv2.THRESH_BINARY)
cv2.imwrite("../images/process/marker/process/"+imgName+'-grad_mag.bmp', grad_mag)
cv2.imwrite("./images/process/marker/process/"+imgName+'-grad_thresh.bmp', sobel_thresh)

edges = cv2.Canny(blurred1, 50, 150)
cv2.imwrite("../images/process/marker/process/"+imgName+"-edges1.bmp", edges)

edges[edges==255] = 1
skeleton0 = morphology.skeletonize(edges)
refine = skeleton0.astype(np.uint8)*255
cv2.imwrite("../images/process/marker/process/"+imgName+"-refine.bmp", refine)

circleCnts, _ = cv2.findContours(refine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
circularCnts, _ = cv2.findContours(sobel_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE

# 画布
mask1 = np.zeros(edges.shape[:2], dtype=np.uint8)
mask2 = np.zeros(edges.shape[:2], dtype=np.uint8)
mask_canvas = cv2.cvtColor(refine, cv2.COLOR_GRAY2BGR)
mask_canvas1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
mask_canvas2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


for index in range(len(circleCnts)):
    cnt = circleCnts[index]
    length = cv2.arcLength(cnt, False)
    area = cv2.contourArea(cnt, False)
    (x,y), (a,b), angle = cv2.fitEllipse(cnt)
    if area>5 or length>1.6*math.pi*a:  # 筛选圆形边缘（包含闭合的和断开的）
        cv2.drawContours(mask1, [cnt], -1, (255,255,255), 1)
        que = detectEndPoints(cnt)  # 对断开的边缘进行连接
        if len(que) >= 2:
            connectEdges(que, mask1)

for c in circularCnts:
    area = cv2.contourArea(c, False)
    if area > 400:
        (x,y), (a,b), _ = cv2.fitEllipse(c)
        print(a, b, abs(a-b))
        if abs(a-b)>10 and 30<b<400:    # 筛选圆环带边缘
            cv2.drawContours(mask1, [c], -1, (255,255, 255), -1)

#
cv2.imwrite("../images/process/marker/process/"+imgName+"-close-contours.bmp", mask1)
# cv2.imwrite("../images/process/marker/process/"+imgName+"-open-contours.bmp", mask_canvas2)
# cv2.imwrite("../images/process/marker/process/"+imgName+"-contours.bmp", mask_canvas1)
#
# # d_knernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# # e_knernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
# # cv2.dilate(mask1, d_knernal, mask1)
# # cv2.erode(mask1, e_knernal, mask1)
# # cv2.imwrite("./images/0411-pm/cal-edges/"+imgName+"-mask1-connect.bmp", mask1)
#
#
# contours_mask, hierarchy_mask = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
# # # 将轮廓按照层级进行分类
# contour_dict = {}
# for i, c in enumerate(contours_mask):
#     level = 0
#     next_index = hierarchy_mask[0][i][3]
#     while next_index != -1:
#         level += 1
#         next_index = hierarchy_mask[0][next_index][3]
#
#     if level not in contour_dict:
#         contour_dict[level] = [c]
#     else:
#         contour_dict[level].append(c)
# #
# colors_list = [(255, 255, 255), (0, 0, 0)]
# for k, v in contour_dict.items():
#     # 一条闭合的曲线有两个轮廓
#     if k%2:
#         cv2.drawContours(mask1, v, -1, colors_list[((k // 2) % 2)], -1)
# # cv2.imwrite("./images/process/0411-pm/cal-edges/" + imgName + "-mask1-fill.bmp", mask1)
# ret, thresh = cv2.threshold(mask1 + mask2, 0, 255, cv2.THRESH_OTSU)
#
# kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal, thresh)
# #
# cv2.imwrite("../images/process/marker/process/" + imgName + "-res.bmp", thresh)
# #
# recognition(thresh)


