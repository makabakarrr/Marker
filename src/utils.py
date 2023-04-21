import math
import numpy as np


# 获取距离某一点最近的k个点：
def getKClosestPoints(target, points, k):
    """
    从坐标点列表中获取距离某一目标点最近的k个点
    :param target:  目标点
    :param points:  坐标点列表
    :param k:   最近的k个
    :return:    返回距目标点最近的k个坐标点列表
    """
    res = []
    if len(points) >= k:
        point_list = []
        for point in points:
            point_list.append([point[0], point[1], math.sqrt(pow(point[0]-target[0], 2)+pow(point[1]-target[1], 2))])

        point_list.sort(key=lambda x: x[2])

        for i in range(0,k):
            res.append([point_list[i][0], point_list[i][1]])
    return res


# 计算点到直线的距离--向量法
def getDistance(target, line_point1, line_point2):
    """
    向量法求距离
    :param target:  list-----目标点的坐标
    :param line_point1:     list----直线向量起点坐标
    :param line_point2:     list----直线向量终点坐标
    :return: target到line的距离---float
    """
    l_p1 = np.array(line_point1)
    l_p2 = np.array(line_point2)
    p = np.array(target)
    vec1 = l_p1 - p
    vec2 = l_p2 - p
    dist = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(l_p1 - l_p2)
    return dist


# 获取距离直线最近的点
def getClosestPointToLine(points, line_point1, line_point2):
    """
    获取距离直线最近的点
    :param points: 点坐标列表
    :param line_point1: 直线上的点---用来确定直线
    :param line_point2: 直线上的点
    :return: 最近的点的坐标
    """
    p_list = []
    for point in points:
        dist = getDistance(point, line_point1, line_point2)
        p_list.append([point[0], point[1], dist])
    p_list.sort(key=lambda x: x[2])
    return [p_list[0][0], p_list[0][1]]



def calculateAngle(v1, v2):
    """
    计算两个向量的夹角
    :param v1: 向量1
    :param v2: 向量2
    :return: 夹角---弧度值
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    theta = np.arctan2(det, dot) if np.arctan2(det, dot) > 0 else 2 * np.pi + np.arctan2(det, dot)
    return theta


def calculateAverageAngle(angle1, angle2):
    """
    计算angle1与angle2的平均角度：
    :param angle1:角度1
    :param angle2:角度1
    :return:平均角度
    """
    if angle1 > 270 and angle2 < 90:
        angle2 += 360
    elif angle2 > 270 and angle1 < 90:
        angle1 += 360
    return ((angle1 + angle2) / 2) % 360



def calculateCrossPoint(firstPoints, secondPoints):
    """
    计算两条直线的交点
    :param firstPoints: 第一条直线上两个点的坐标
    :param secondPoints: 第二条直线上两点的坐标
    :return: 交点
    """
    x1, y1 = firstPoints[0]
    x2, y2 = firstPoints[1]
    x3, y3 = secondPoints[0]
    x4, y4 = secondPoints[1]
    x, y = 0, 0
    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0

    return [x, y]



def getAngle(point1, point2):
    """
    计算两点所在直线的方位角
    Args:
        point1: 坐标
        point2: 坐标

    Returns: 角度--------逆时针

    """
    angle = 0.0
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 > y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 < y1:
        angle = math.atan(-dy / dx)
    elif  x2 > x1 and  y2 > y1 :
        angle = math.pi*2 - math.atan(dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi - math.atan(dy / dx)
    elif  x2 < x1 and y2 > y1 :
        angle = math.pi + math.atan(dy / -dx)
    return (angle * 180 / math.pi)

