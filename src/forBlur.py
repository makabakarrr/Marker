# import cv2
# import numpy as np
import time

import cv2
import matplotlib.pyplot as plt
import sys
import itertools

import numpy as np

from customFunction import *
from utils import calculateCrossPoint, getAngle, calculateAngle

from sklearn.cluster import DBSCAN

def extractCircles(cnts, no_branch):
    circle_canvas = cv2.cvtColor(no_branch, cv2.COLOR_GRAY2BGR)
    circle_info = []
    for cnt_index in range(len(cnts)):
        cnt = cnts[cnt_index]
        if len(cnt) > 20:
            (cx, cy), (a, b), angle = cv2.fitEllipse(cnt)
            radius = max(a, b) / 2
            # 求内点数量
            perimeter = 2 * np.pi * radius
            nums = 0
            test_points = []
            for point in cnt:
                x, y = point[0]
                dist = abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius)
                # if dist <= 1 and [x, y] not in test_points:
                if dist < 3 and [x, y] not in test_points:  # test_point 避免重复
                # if dist < 3:  # test_point 避免重复
                    nums += 1
                    test_points.append([x, y])
            k = nums / perimeter
            cv2.circle(circle_canvas, (int(cx), int(cy)), int(radius), (0,0,255), 1)
            # print("k:", k)
            if k > 0.8:  # 圆形轮廓
                # cv2.ellipse(filter_cnts, (int(cx), int(cy)), (int(a/2), int(b/2)), angle, 0, 360, (0,255,0),1)
                cv2.drawContours(filter_cnts, cnt, -1, (255, 255, 255), 1)
                circle_info.append([cx, cy, radius, cnt_index])
    cv2.imwrite("../images/process/summary/filterCnts/" + imgName + '-filter_cnts.png', filter_cnts)
    cv2.imwrite("../images/process/summary/filterCnts/" + imgName + '-fit_circle.png', circle_canvas)
    return circle_info


def classifyCircles(circle_info, cnts):
    circle_centers = np.array(circle_info)[:, 0:2]
    db = DBSCAN(eps=5, min_samples=1).fit(circle_centers)
    c_s = db.labels_
    start_template_circle_points = []  # 起始模板点的边缘点 [[起始模板点1 [圆形1 [点1], [点2]], [圆形2]], [起始模板点2]]
    start_template_circle_info = []
    other_template_circle_points = []  # 其他模板点的边缘点
    other_template_circle_info = []
    locate_circle_points = []   # 定位圆的边缘点
    locate_circle_info = []
    other_circle_points = []  # 编码圆
    other_circle_info = []
    all_radius = np.array(circle_info)[:, 2]
    sorted_all_radius = np.argsort(all_radius)
    max_radius = circle_info[sorted_all_radius[-1]][2]
    for c in range(np.max(c_s) + 1):
        cate_indexs = np.where(c_s == c)[0]  # 某一类的索引
        cate_circles = np.array(circle_info)[cate_indexs]
        cur_cate_radius = cate_circles[:, 2]
        sorted_radius = np.argsort(cur_cate_radius)
        sorted_circles = cate_circles[sorted_radius]  # 对该类的圆参数按半径从小到大进行排序
        radius_flag = 0  # 最小的半径值
        radius_category = 1
        all_edges_points = []
        single_cnt_points = []
        for circle_index in range(len(sorted_circles)):
            circle = sorted_circles[circle_index]
            if circle_index == 0:
                radius_flag = circle[2]
                single_cnt_points = [[p[0][0], p[0][1]] for p in cnts[int(circle[3])]]
            else:
                if circle[2]-radius_flag < 5:  # 同一个圆形
                    for p in cnts[int(circle[3])]:
                        i, j = p[0]
                        if [i, j] not in single_cnt_points:
                            single_cnt_points.append([i, j])
                else:
                    radius_flag = circle[2]
                    radius_category += 1
                    all_edges_points.append(single_cnt_points)
                    single_cnt_points = [[p[0][0], p[0][1]] for p in cnts[int(circle[3])]]
        all_edges_points.append(single_cnt_points)

        if radius_category == 1:  # 1种半径  定位圆和编码圆
            if max_radius-radius_flag < 5:   # 只有一种半径值，并且是标记点中半径最大的圆，标记点中半径最大的圆有四个模板点最外层的圆、中心定位圆
                locate_circle_points.append(all_edges_points)
                locate_circle_info.append(sorted_circles)
            else:
                other_circle_points.append(all_edges_points)
                other_circle_info.append(sorted_circles)
        elif radius_category == 2:  # 2种半径  起始模板点
            start_template_circle_points.append(all_edges_points)
            start_template_circle_info.append(sorted_circles)
        else:  # 其他类模板点
            other_template_circle_points.append(all_edges_points)
            other_template_circle_info.append(sorted_circles)
    return start_template_circle_points, start_template_circle_info, other_template_circle_points, other_template_circle_info, locate_circle_points, locate_circle_info, other_circle_points, other_circle_info


def calCenters(circle_Points):
    center_list = [] # 几组圆的中心坐标
    # 有几组圆
    for cate in circle_Points:
        # 每一组圆里有几个圆
        cur_cate_center = []
        for circle in cate:
            # 使用椭圆拟合的方式计算中心坐标点
            (x, y), (a,b), angle = cv2.fitEllipse(np.array(circle))
            cur_cate_center.append([x, y])
        center_mean = np.mean(np.array(cur_cate_center), axis=0)
        center_list.append(np.array(center_mean))
    return center_list


def calCentersByAverage(center_info):
    cate_center= []
    for cate in center_info:
        mean_val = np.array(np.mean(np.array(cate)[:, 0:2], axis=0).round(4))
        cate_center.append([mean_val[0], mean_val[1]])
    return cate_center


def gaussianFit(edge_points):
    for point in edge_points:
        x, y = point.ravel()

        # 获取3x3邻域图像块
        image_block = img[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2]

        # 计算图像块的梯度
        gradient_x = cv2.Sobel(image_block, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image_block, cv2.CV_64F, 0, 1, ksize=3)

        # 将梯度转换为向量形式
        gradient_x = gradient_x.flatten()
        gradient_y = gradient_y.flatten()

        # 组合梯度向量
        gradient_vector = np.vstack((gradient_x, gradient_y)).T

        # 进行线性拟合
        line_direction = cv2.fitLine(gradient_vector, cv2.DIST_L2, 0, 0.01, 0.01)

        # 提取拟合参数
        vx, vy, x0, y0 = line_direction

        # 计算亚像素边缘坐标点
        subpixel_x = x0[0] - vx[0] * y0[0] / vy[0]
        subpixel_y = y0[0] - vy[0] * x0[0] / vx[0]



def getDistCombinations(source_combinations, source_points, dist_points):
    src_n0, src_n1, src_n2, src_n3 = source_points
    n0, n1, n2, n3 = dist_points
    dist_combinations  = []
    for s in source_combinations:
        d = []
        for point in s:
            xx, yy = point
            if [xx, yy] == [src_n0[0], src_n0[1]]:
                d.append(n0)
                # d.append('n0')
            elif [xx, yy] == [src_n1[0], src_n1[1]]:
                d.append(n1)
                # d.append('n1')
            elif [xx, yy] == [src_n2[0], src_n2[1]]:
                d.append(n2)
                # d.append('n2')
            else:
                d.append(n3)
                # d.append('n3')
        dist_combinations.append(d)

    return np.array(dist_combinations)


def delBranch(edges, angle_thresh):
    sub_mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    sub_mask2 = edges.copy()
    cnts0, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts0:
        if len(cnt) > 20:
            epsilon = 1
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            ### approx可视化
            colors = []
            for i in range(len(approx)):
                colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            for ap_index in range(len(approx) - 1):
                i, j = approx[ap_index][0]
                ii, jj = approx[ap_index + 1][0]
                sub_mask1[j, i, :] = colors[ap_index]
                cv2.line(sub_mask1, [i, j], [ii, jj], colors[ap_index], 1)

            for index in range(1, len(approx) - 1):
                p_0 = np.array(approx[index - 1][0])
                p_1 = np.array(approx[index][0])
                p_2 = np.array(approx[index + 1][0])
                l_0 = p_0 - p_1
                l_1 = p_1 - p_2
                aa = np.dot(l_0, l_1) / (np.linalg.norm(l_0) * np.linalg.norm(l_1))
                A = np.degrees(np.arccos(aa))
                if A > angle_thresh:
                    xx, yy = p_1
                    sub_mask2[yy][xx] = 0
    cv2.imwrite("../images/process/summary/approxResult/" + imgName + "_cnt_approx.png", sub_mask1)
    cv2.imwrite("../images/process/summary/approxResult/" + imgName + "_cnt_break.png", sub_mask2)
    return sub_mask2



def decodedByGray(points, lp, side_length, img):
    canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # 根据灰度值进行解码----灰度值几乎无差别，区分不出来
    step = side_length / 10
    temp_dist = step * 2
    gray = 0.0
    # 随机取四个方向的模板
    for i in range(4):
        angle = random.randint(0, 360)
        x = int(lp[0] + temp_dist*np.cos(np.radians(angle)))
        y = int(lp[1] - temp_dist*np.sin(np.radians(angle)))
        cv2.circle(canvas, (x,y), 2, (0,0,255), -1)
        surrounding = img[y-5:y+6, x-5:x+6]
        gray_mean = np.mean(surrounding)
        gray += gray_mean
    gray /= 4
    int_str = ''
    for index in range(len(points)):
        p = points[index]
        i, j = int(p[0]), int(p[1])
        cv2.putText(canvas, str(index), (i, j), cv2.FONT_HERSHEY_SIMPLEX, 1, (247, 162, 17), 2)
        cv2.circle(canvas, (i, j), 2, (0, 255, 0), -1)
        p_surrounding = img[j-5:j+6, i-5:i+6]
        p_gray = np.mean(p_surrounding)
        if abs(p_gray - gray) > 5:
            int_str += '1'
        else:
            int_str += '0'
    cv2.imwrite("../images/process/summary/decode/" + imgName + '-pos.png', canvas)
    return int_str


def extractCircularImg(contours, side_length, height, width):
    dec_edges = np.zeros((height, width), np.uint8)
    # 提取小数部分的边缘信息
    length = int(side_length / 10 * 4)
    locate_radius = int(side_length / 10 * 2)
    for cnt in contours:
        if len(cnt) > 20:
            # 获取x的最大值和最小值
            x_max = np.max(cnt[:, 0, 0])
            x_min = np.min(cnt[:, 0, 0])
            # 获取y的最大值和最小值
            y_max = np.max(cnt[:, 0, 1])
            y_min = np.min(cnt[:, 0, 1])
            if lp[0] - length < x_min and x_max < lp[0] + length and lp[1] - length < y_min and y_max < lp[1] + length:
                # cv2.drawContours(dec_edges, cnt, -1, (255, 255, 255), 1)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(dec_edges, [box], 0, (255,255,255), -1)
                if lp[0] - locate_radius < x_min and x_max < lp[0] + locate_radius and lp[
                    1] - locate_radius < y_min and y_max < lp[1] + locate_radius:
                    cv2.drawContours(dec_edges, [box], 0, (0, 0, 0), -1)
                    # cv2.drawContours(dec_edges, cnt, -1, (0, 0, 0), 1)
    cv2.imwrite("../images/process/summary/decode/" + imgName + "_dec_rect.png", dec_edges)
    # cv2.imwrite("../images/process/summary/decode/" + imgName + "_dec_edges1.png", dec_edges)


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


imgName = "marker0-20"
img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
height, width = img.shape[0], img.shape[1]
bilateral_blur = cv2.bilateralFilter(img, -1, 8, 8)
cv2.imwrite("../images/process/summary/medianBlur/" + imgName + '-bilateral.png', bilateral_blur)
clache = cv2.createCLAHE(4, (8,8))
enhance = clache.apply(bilateral_blur)
# median_blur = cv2.medianBlur(enhance, 3)
usm = unSharpMask(enhance, 1.4, 200, 0)
cv2.imwrite("../images/process/summary/enhance/" + imgName + '-usm.png', usm)
blur = cv2.GaussianBlur(usm, (0,0), 1.75)
cv2.imwrite("../images/process/summary/gaussianBlur/" + imgName + '-blur.png', blur)
# cv2.imwrite("../images/process/summary/medianBlur/" + imgName + '-blur.png', median_blur)
## 图像增强
# clache = cv2.createCLAHE(4, (32,32))
# enhance = clache.apply(img)
# cv2.imwrite("../images/process/summary/enhance/" + imgName + '-clache.png', enhance)
#
# usm = unSharpMask(enhance, 1.4, 200, 0)
# cv2.imwrite("../images/process/summary/enhance/" + imgName + '-usm.png', usm)
#
# ## 图像平滑
# # blur1 = cv2.medianBlur(usm, 5)
# blur = cv2.GaussianBlur(usm, (0,0), 2)
# cv2.imwrite("../images/process/summary/gaussianBlur/" + imgName + '-blur.png', blur)
# cv2.imwrite("../images/process/summary/medianBlur/" + imgName + '-blur.png', median_blur)
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
sobely= cv2.Sobel(blur, cv2.CV_64F, 0, 1)

mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
grad_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
th, _ = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("th:", th)
tl = max(0, int(th * 0.2))
edges = cv2.Canny(blur, tl,th-20)
cv2.imwrite("../images/process/summary/gaussianOstuCanny/"+imgName+"_edges.png", edges)
# edges1 = cv2.Canny(median_blur, tl,th-20)
# cv2.imwrite("../images/process/summary/gaussianOstuCanny/"+imgName+"_edges1.png", edges1)
rows, cols = np.where(edges==255)
edges_points = [[x,y] for x,y in list(zip(cols, rows))]

# 去除圆弧相交的情况
no_branch = delBranch(edges, 80)

# 圆形检测
filter_cnts = np.zeros(edges.shape, dtype=np.uint8)
contours, _ = cv2.findContours(no_branch, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
circle_info = extractCircles(contours, no_branch)


# 圆形分类
start_template_points, start_template_info, other_template_points, other_template_info, \
locate_circle_points, locate_circle_info, other_circle_points, other_circle_info = classifyCircles(circle_info, contours)
for other_circle in other_circle_info:
   for c in other_circle:
       cnt_i = c[3]
       cv2.drawContours(no_branch, contours, int(cnt_i), (255,255,255), -1)
cv2.imwrite("../images/process/summary/decode/" + imgName + "_coded_circles.png", no_branch)
# 圆形分类可视化
test_edges = np.zeros((edges.shape[0], edges.shape[1], 3))
for template in locate_circle_points:
    for circle in template:
        for point in circle:
            xx, yy = point
            test_edges[yy][xx] = (125, 186, 94)
for template in start_template_points:
    for circle in template:
        for point in circle:
            xx, yy = point
            test_edges[yy][xx] = (14, 169, 250)
for template in other_template_points:
    for circle in template:
        for point in circle:
            xx, yy = point
            test_edges[yy][xx] = (17,17, 255)
for other_circle in other_circle_points:
    for circle in other_circle:
        for point in circle:
            xx, yy = point
            test_edges[yy][xx] = (247, 162, 17)
cv2.imwrite("../images/process/summary/templateCircle/"+imgName+"_circles_category.png", test_edges)

# 求出模板点、定位圆的中心坐标点
# template_circle_centers = calCenters(other_template_points)
start_circle_centers = calCentersByAverage(start_template_info)
template_circle_centers = calCentersByAverage(other_template_info)
locate_circle_centers = calCentersByAverage(locate_circle_info)
# print(template_circle_centers)
# print("locate_circle_centers", locate_circle_centers)
# 识别N1, N2, N3
## 设计坐标
# 根据设计图的特征，假设模板点坐标如下：
src_n0 = [50, 50]
src_n1 = [250, 50]
src_n2 = [50, 250]
src_n3 = [250, 250]
src_lp = [150, 150]
source_points = np.array([src_n0, src_n1, src_n2, src_n3], dtype=np.float32)
# 整数编码圆中心的设计坐标
code_points = generateCodePoints(src_n0, 200)  # 根据设计图的比例生成编码点的中心坐标列表
# band_angles = calculateBandAnglesByTemplatePoint(src_lp, src_n0, src_n1, src_n2, src_n3)    # 计算设计图上每个编码带的起始角度、终止角度
# 小数编码块中心的设计坐标
band_angles = calculateBandAngleByAverage(10)
bandCenters = getBandCenter(band_angles, src_lp, 200)  # 计算每个编码带中间位置的设计坐标
# 从source_points中任选3点的组合方式
source_combinations = np.array(list(itertools.combinations(source_points, 3)))
# print(source_points)
# print(source_combinations)

drawCircle = cv2.cvtColor(filter_cnts, cv2.COLOR_GRAY2BGR)
for n0 in start_circle_centers:
    lp, n1, n2, n3 = matchPoint(template_circle_centers, n0, locate_circle_centers)

    ## 可视化
    cv2.putText(drawCircle, 'n0', (int(n0[0]+50), int(n0[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (247, 162, 17), 2)
    cv2.circle(drawCircle, (int(n0[0]), int(n0[1])), 2, (247, 162, 17), -1)
    cv2.putText(drawCircle, 'n1', (int(n1[0]+50), int(n1[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 186, 94), 2)
    cv2.circle(drawCircle, (int(n1[0]), int(n1[1])), 2, (125, 186, 94), -1)
    cv2.putText(drawCircle, 'n2', (int(n2[0]+50), int(n2[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (17,17, 255), 2)
    cv2.circle(drawCircle, (int(n2[0]), int(n2[1])), 2, (17,17, 255), -1)
    cv2.putText(drawCircle, 'n3', (int(n3[0]+50), int(n3[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (14, 169, 250), 2)
    cv2.circle(drawCircle, (int(n3[0]), int(n3[1])), 2, (14, 169, 250), -1)
    cv2.imwrite("../images/process/summary/templateCircle/" + imgName + '-recognize.png', drawCircle)

    # 从n0, n1, n2, n3中找到与combinations组合中与src_x对应的点
    dist_points = np.array([n0, n1, n2, n3], dtype=np.float32)
    dist_combinations = getDistCombinations(source_combinations, source_points, dist_points)
    angle = 0.0
    M = np.zeros((2,3), np.float32)

    for i in range(len(source_combinations)):
        MM = calculateAffineMatrix(source_combinations[i], dist_combinations[i])
        M = np.add(M, MM)
        angle += np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

    transform_matrix = M / len(source_combinations)
    angle = angle / len(source_combinations)
    print("均值：", 135 - angle)
    M1 = calculateAffineMatrix(source_combinations[0], dist_combinations[0])
    angle1 = np.arctan2(M1[1, 0], M1[0, 0]) * 180 / np.pi
    print("使用1组数据", 135 - angle1)
    angle2 = getAngle(lp, n0)
    print("lp-n0:", angle2)

    ## 使用平均变换矩阵进行解码
    print("average_matrix:", transform_matrix)
    print("single_matrix:", M1)



    ## 整数部分解码
    side_length = np.sqrt((n0[0]-n1[0])**2+(n0[1]-n1[1])**2)
    int_points = cvtCodePoints1(code_points, transform_matrix)
    res_str = getCodeVal(no_branch, int_points)
    print("解码数据：", res_str)
    ## 小数部分解码
    dec_points = cvtCodePoints1(bandCenters, transform_matrix)
    circular_img = extractCircularImg(contours, side_length, height, width)


    ## 计算角度
    # angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
    # print("角度：", angle) # -35.98058

# 对模板点、定位圆进行亚像素边缘定位## 1)使用高斯拟合进行亚像素边缘检测



