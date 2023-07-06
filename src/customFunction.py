import math
import cv2
import numpy as np
import random

from utils import getKClosestPoints, getClosestPointToLine, calculateAngle, calculateAverageAngle


# 创建圆
def create_circle():
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    circle_list = []
    # height, width, n = mask.shape
    nums = random.randint(20, 50)  # 光斑的个数20-50之间
    # nums = 1
    for i in range(0, nums):
        v = random.randint(0, 255)
        r = random.randint(2, 30)
        template = np.zeros((2 * r, 2 * r), dtype=np.uint8)
        cv2.circle(template, (r, r), r, (v, v, v), -1)
        for w in range(0, 2 * r):
            for h in range(0, 2 * r):
                template[w, h] = int(template[w, h] * random.uniform(0.7, 1.5))
        circle_list.append(template)
    return circle_list


# 随机添加光斑
def add_circles(img):
    height, width = img.shape
    newImg = img.copy()
    circle_list = create_circle()
    for circle in circle_list:
        h, w = circle.shape
        x = random.randint(0, height - h)
        y = random.randint(0, width - w)

        for i in range(0, h):
            for j in range(0, w):
                # 防止溢出
                cur = int(newImg[x + i, y + j])
                newImg[x + i, y + j] = 255 if cur + circle[i, j] > 255 else cur + circle[i, j]
    return newImg


def showPic(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 提取圆形轮廓：
def getCircleCntsByArea(cnts, threshold):
    """
    提取圆形轮廓：根据性质（圆的最小外接圆面积与轮廓面积相近）筛选
    :param cnts: 轮廓列表
    :return: 圆形轮廓列表
    """
    res = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 200:
            _, radius = cv2.minEnclosingCircle(cnt)    # 最小外接圆
            s = math.pi * radius ** 2
            if math.isclose(area, s, abs_tol=threshold):
                res.append(cnt)

    return res


# 提取圆形轮廓：
def getCircleCntsByRoundness(cnts):
    """
    提取圆形轮廓：根据圆度（0.9）筛选
    :param cnts: 轮廓列表
    :return: 圆形轮廓列表
    """
    res = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 100:
            length = cv2.arcLength(cnt, True)
            k = 4 * math.pi * area / (length ** 2)

            if math.isclose(k, 0.9, abs_tol=0.1):
                res.append(cnt)
    return res


#  将圆形轮廓进行分类
def cateCircles(cnts, thresh):
    """
    对圆形轮廓进行分类， 中心坐标均由椭圆拟合得到
        1类模板点--中心到边缘的灰度值变化为255-0-255
        2类模板点--中心到边缘的灰度值变化为0-255
        非模板点---中心到边缘的灰度值无变化，均为255
    :param cnts: 圆形轮廓列表
    :param thresh: 二值图像
    :return: 三组轮廓列表及中心点坐标
    """
    model1 = []  # 模板点1
    center1 = []
    model2 = []  # 模板点2
    center2 = []
    model3 = []  # 非模板点---------中心定位圆和编码点
    center3 = []
    for i in range(0, len(cnts)):
        circle = cnts[i]
        # M = cv2.moments(circle)
        # cx = int(M['m10'] / M['m00'])  # 中心点坐标
        # cy = int(M['m01'] / M['m00'])  # 中心点坐标
        (x, y), _, _ = cv2.fitEllipse(circle)
        cx, cy = int(x), int(y)
        y_coordinate = 0    # 记录圆形轮廓的下边缘点的y坐标
        for point in circle:
            p = point[0]
            if p[0] == cx and p[1] > y_coordinate:
                y_coordinate = p[1]

        pixel_cate = 0  # 记录圆心到下边缘点的灰度值变化次数 255-0-255变化次数为2  0-255变化次数为1、255不变
        for j in range(cy, y_coordinate-1): # y_coordinate-1为了消除模板点白色圆环内边界的影响
            if thresh[j + 1, cx] != thresh[j, cx]:
                pixel_cate += 1
        if thresh[cy, cx] == 255 and pixel_cate == 2:  # 1类模板点--N0
            model1.append(circle)
            center1.append([x, y])
        elif thresh[cy, cx] == 0 and pixel_cate == 1:  # 2类模板点--N1,N2,N3
            model2.append(circle)
            center2.append([x, y])
        elif thresh[cy, cx] == 255 and pixel_cate == 0:  # 非模板点--编码圆、定位圆、方向标识圆
            model3.append(circle)
            center3.append([x, y])
    return model1, center1, model2, center2, model3, center3

#  将圆形轮廓进行分类
def categoryCircles(cnts, thresh):
    """
    对圆形轮廓进行分类， 中心坐标均由椭圆拟合得到
        1类模板点--中心到边缘的灰度值变化为255-0-255
        2类模板点--中心到边缘的灰度值变化为0-255
        非模板点---中心到边缘的灰度值无变化，均为255
    :param cnts: 圆形轮廓列表
    :param thresh: 二值图像
    :return: 三组轮廓列表及中心点坐标
    """
    model1 = []  # 模板点1
    center1 = []
    model2 = []  # 模板点2
    center2 = []
    model3 = []  # 非模板点---------中心定位圆和编码点
    center3 = []
    for i in range(0, len(cnts)):
        circle = cnts[i]
        # M = cv2.moments(circle)
        # cx = int(M['m10'] / M['m00'])  # 中心点坐标
        # cy = int(M['m01'] / M['m00'])  # 中心点坐标
        (x, y), _, _ = cv2.fitEllipse(circle)
        area = cv2.contourArea(circle)
        cx, cy = int(x), int(y)
        y_coordinate = 0    # 记录圆形轮廓的下边缘点的y坐标
        for point in circle:
            p = point[0]
            if p[0] == cx and p[1] > y_coordinate:
                y_coordinate = p[1]

        pixel_cate = 0  # 记录圆心到下边缘点的灰度值变化次数 255-0-255变化次数为2  0-255变化次数为1、255不变
        for j in range(cy, y_coordinate-1): # y_coordinate-1为了消除模板点白色圆环内边界的影响
            if thresh[j + 1, cx] != thresh[j, cx]:
                pixel_cate += 1
        if area>4000 and pixel_cate == 3:  # 1类模板点--N0
            model1.append(circle)
            center1.append([x, y])
        elif area>4000 and pixel_cate == 5:  # 2类模板点--N1,N2,N3
            model2.append(circle)
            center2.append([x, y])
        elif (area>4000 or area<1000) and pixel_cate == 1:  # 非模板点--编码圆、定位圆、方向标识圆
            model3.append(circle)
            center3.append([x, y])
    return model1, center1, model2, center2, model3, center3


# 提取中心定位圆
def getLocateCircles(cnts, centers):
    """
    将定位圆从非模板点列表中分离出来：比较半径的长度
    Args:
        cnts: 轮廓列表
        centers: 中心坐标列表

    Returns: 定位圆轮廓列表， 定位圆中心坐标列表

    """
    locate_circle = []
    locate_center = []
    radius_list = []
    for i in range(0, len(cnts)):
        cnt = cnts[i]
        _, radius = cv2.minEnclosingCircle(cnt)
        radius_list.append(radius)
    target = min(radius_list)   # 最小半径
    for i in range(len(cnts)-1, -1, -1):
        cnt = cnts[i]
        _, radius = cv2.minEnclosingCircle(cnt)
        if radius > 2 * target:
            locate_circle.append(cnt)
            locate_center.append(centers[i])
            del cnts[i]
            del centers[i]

    return locate_circle, locate_center


# 寻找标识方向的圆
def getDirectionCircle(centers, locate_center):
    """
    寻找标识方向的圆轮廓与圆心：从cnts中筛选距离定位圆中心最近的圆
    Args:
        cnts: 编码圆与方向圆中心坐标列表
        locate_center: 定位圆中心坐标

    Returns: 方向圆的中心坐标

    """
    return getKClosestPoints(locate_center, centers, 1)[0]



# 匹配模板点与定位圆
def matchPoint(center1, N0, locate_center):
    """
    根据N0获取同一个标记中的中心点lp，模板点N1、N2、N3
    :param center1: 1类模板点坐标列表
    :param N0: N0的坐标
    :param locate_center: 中心圆心坐标列表
    :return: lp, n1, n2, n3
    """
    lp = getKClosestPoints(N0, locate_center, 1)[0]  # 距离N0最近的定位圆为该标记点的定位圆
    locate_center.remove(lp)  # 从locate_center中移除该点
    template_points = getKClosestPoints(lp, center1, 3)  # 从1类模板点中取出距离lp最近的3个点
    n3 = getClosestPointToLine(template_points, N0, lp)    # 距离直线N0-lp最近的点为N3
    template_points.remove(n3)
    p1 = np.array(template_points[0])
    p2 = np.array(template_points[1])
    # x_right = np.array([lp[0], N0[1]])
    v1 = np.array(lp) - p1
    v2 = np.array(lp) - p2
    level = np.array(lp) - np.array(N0)
    # level = np.array(lp) - x_right
    theta1 = calculateAngle(v1, level)
    theta2 = calculateAngle(v2, level)
    # print("向量夹角：", np.degrees(theta1), np.degrees(theta2))
    if theta1 < theta2:
        n2 = template_points[0]
        n1 = template_points[1]
    else:
        n2 = template_points[1]
        n1 = template_points[0]
    return lp, n1, n2, n3


# 计算投影变换矩阵
def calculatePerspectiveMatrix(src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M


# 计算仿射变换矩阵
def calculateAffineMatrix(src_points, dst_points):
    M = cv2.getAffineTransform(src_points, dst_points)
    return M


# 生成编码点的坐标列表：
def generateCodePoints(startPoint, side):
    """

    :param startPoint: 左上顶点坐标---靠近0号编码点的顶点
    :param side: 两个模板中心点之间的距离
    :return: 编码圆中心坐标列表
    """
    points = []
    start_x, start_y = startPoint[0], startPoint[1]
    x, y = start_x, start_y
    step1 = side / 10 * 3   # 编码圆中心到模板点中心的步长
    step2 = side / 10 * 2   # 编码圆中心之间的距离
    y += step1
    x += step1
    for i in range(0, 3):
        points.append([x, start_y])
        if i < 2:
            x += step2
        else:
            x += step1

    for j in range(0, 3):
        points.append([x, y])
        if j < 2:
            y += step2
        else:
            y += step1

    x -= step1
    for m in range(0, 3):
        points.append([x, y])
        if m < 2:
            x -= step2
        else:
            x -= step1

    y -= step1
    for n in range(0, 3):
        points.append([x, y])
        if n < 2:
            y -= step2
        else:
            y -= step1

    return points


def cvtCodePoints(codePoints, matrix):
    """
    将设计坐标转换为图像坐标
    :param codePoints: 坐标列表
    :param matrix: 变换矩阵
    :return: 变换后的坐标列表
    """
    res = []
    for point in codePoints:
        u = point[0]
        v = point[1]
        x = (matrix[0][0] * u + matrix[0][1] * v + matrix[0][2]) / (matrix[2][0] * u + matrix[2][1] * v + matrix[2][2])
        y = (matrix[1][0] * u + matrix[1][1] * v + matrix[1][2]) / (matrix[2][0] * u + matrix[2][1] * v + matrix[2][2])
        # x = (matrix[0][0] * u + matrix[1][0] * v + matrix[2][0]) / (matrix[0][2] * u + matrix[1][2] * v + matrix[2][2])
        # y = (matrix[0][1] * u + matrix[1][1] * v + matrix[2][1]) / (matrix[0][2] * u + matrix[1][2] * v + matrix[2][2])
        res.append([x, y])
    return res


def cvtCodePoints1(codePoints, matrix):
    """
    将设计坐标转换为图像坐标
    :param codePoints: 坐标列表
    :param matrix: 变换矩阵
    :return: 变换后的坐标列表
    """
    res = []
    for point in codePoints:
        u = point[0]
        v = point[1]
        x = round(matrix[0][0] * u + matrix[0][1] * v + matrix[0][2], 4)
        y = round(matrix[1][0] * u + matrix[1][1] * v + matrix[1][2], 4)
        res.append([x, y])
    return res


def getCodeVal(thresh, points):
    """
    遍历points，读取灰度值，进行解码
    :param thresh: 二值图像
    :param points: 读取位置---list
    :return: 二进制序列
    """
    res = []
    for point in points:
        i, j = int(point[1]), int(point[0])
        surrounding = thresh[i-1:i+2, j-1:j+2]
        if np.min(surrounding)==0:
            res.append('0')
        else:
            res.append('1')
    return "".join(res)



def calculateBandAnglesByTemplatePoint(lp, n0, n1, n2, n3):
    """
    根据定位圆中心点与四个模板中心点的几何关系计算每个编码环带的起始角度和终止角度----以lp为原点，正右方向为0°
    Args:
        lp: 定位圆中心坐标
        n0: N0模板中心坐标
        n1: N1模板中心坐标
        n2: N2模板中心坐标
        n3: N3模板中心坐标

    Returns:    编码环带的起始角度、终止角度列表

    """
    lp, n0, n1, n2, n3 = np.array(lp), np.array(n0), np.array(n1), np.array(n2), np.array(n3)
    angles = []  # 每个向量与x正方向的夹角
    bandAngles = []  # 每个环带的起始角度和终止角度
    level = np.array([lp[0] + 10, lp[1]]) - lp
    vectors = [n1 - lp, n3 - lp, n2 - lp, n0 - lp, n1 - lp, n3 - lp]
    # 计算每个向量与x正方向的夹角
    for vector in vectors:
        angles.append(math.degrees(calculateAngle(level, vector)))
    j = 0
    # print(angles)
    for i in range(0, 8):
        if i % 2 == 0:
            start_angle = calculateAverageAngle(angles[j], angles[j + 1])
            end_angle = angles[j + 1]
            bandAngles.append([start_angle, end_angle])
            j += 1
        else:
            start_angle = angles[j]
            end_angle = calculateAverageAngle(angles[j], angles[j + 1])
            bandAngles.append([start_angle, end_angle])

    return bandAngles


def calculateBandAngleByAverage(n):
    """
    按设计图获取设计坐标系中每段环带的起始角度与终止角度
    Args:
        n: 圆环等分个数

    Returns: 编码环带的起始角度、终止角度列表

    """
    angles = []
    step = 360 / n
    for i in range(0, n):
        angles.append([step*i, step*(i+1)])
    return angles




def getBandCenter(bandAngles, lp, side):
    """
    计算每个编码环带的中心坐标
    Args:
        bandAngles: 编码环带的起始角度、终止角度列表
        lp: 定位圆中心坐标
        side: 模板点之间的距离

    Returns: 编码环带中心坐标列表
    """
    # length = side/10*3.5
    length = side/10*3
    bandCenter = []
    for band in bandAngles:
        start = band[0]
        # end = band[1] if band[0] < 315 else 360 + band[1]
        end = band[1]

        middle = math.radians((start + end) / 2)
        average_x = lp[0] + length * math.cos(middle)
        average_y = lp[1] + length * math.sin(middle)
        bandCenter.append([average_x, average_y])
    return bandCenter


def isSameCircle(circle1, circle2, dist_threshold, radius_threshold):
    # 计算两个圆心之间的距离
    dist = math.sqrt((circle1[0]-circle2[0])**2 + (circle1[1]-circle2[1])**2)
    # 计算两个圆半径之差
    radius_diff = abs(circle1[2] - circle2[2])
    # 判断是否代表同一个圆
    if (dist <= dist_threshold) and (radius_diff <= radius_threshold):
        return True
    else:
        return False


def getRemainPoint(source_combination, source_points, dist_points):

    src_n0, src_n1, src_n2, src_n3 = [[x,y] for x,y in source_points]
    n0, n1, n2, n3 = [[x,y] for x,y in dist_points]
    s_points = [[x, y] for x, y in source_combination]
    source_remain_point = src_n0
    dist_remain_point = n0

    if src_n0 not in s_points:
        print("组合里没有src_n0")
        source_remain_point = src_n0
        dist_remain_point = n0
    elif src_n1 not in s_points:
        print("组合里没有src_n1")
        source_remain_point = src_n1
        dist_remain_point = n1
    elif src_n2 not in s_points:
        print("组合里没有src_n2")
        source_remain_point = src_n2
        dist_remain_point = n2
    else:
        print("组合里没有src_n3")
        source_remain_point = src_n3
        dist_remain_point = n3

    return source_remain_point, dist_remain_point


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
            elif [xx, yy] == [src_n1[0], src_n1[1]]:
                d.append(n1)
            elif [xx, yy] == [src_n2[0], src_n2[1]]:
                d.append(n2)
            else:
                d.append(n3)
        dist_combinations.append(d)
