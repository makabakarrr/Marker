# author: jiangqin
# date: 2023-01-30 14:09
import cv2
import numpy as np
import re
import math
import datetime
import os

from customFunction import showPic







def decToBin(num, length):
    """
    将十进制num转化为长度为length的二进制
    Args:
        num: 十进制数据
        length: 二进制长度

    Returns: 二进制字符串

    """
    bin_str = ''
    while num > 0:
        bin_str += str(num % 2)
        num = num // 2
    # 补零
    zero_length = length - len(bin_str)
    bin_str += '0' * zero_length

    return bin_str[::-1]



class Marker:
    def __init__(self, distance, angle, diameter=32, side=600):
        '''
        Args:
            diameter: 编码圆点的直径
            side: 标记点的边长
            distance: 编码距离
            angle: 编码角度
        '''
        self.diameter = diameter
        self.side = side
        self.distance = distance
        self.angle = angle
        self.n0_x = int((self.side - 13 * self.diameter) / 2 + 1.5 * self.diameter)  # 模板点N0的坐标
        self.n0_y = int((self.side - 13 * self.diameter) / 2 + 1.5 * self.diameter)
        self.locate_x, self.locate_y = self.n0_x + 5*self.diameter, self.n0_y + 5*self.diameter # 定位圆中心坐标
        self.locate_radius = int(1.5 * self.diameter)    # 中心定位圆的半径

    def encodeDistance(self):
        """
        对距离进行编码
        Returns: 分别返回整数部分和小数部分的编码数据:integer, decimal

        """
        int_num, dec_num = '0', '0'
        # 处理字符串：
        # 正则校验：判断字符串是否由数字和小数点组成
        if (bool(re.match('^[0-9.]+$', self.distance)) == False):
            raise Exception("Data format error: string must contain only 0-9 or.")

        # 判断是否有小数点：有小数点则将字符串切割成两部分；没有则代表距离为整数
        a0 = self.distance.count('.')  # 小数点的数量
        if a0 > 1:
            raise Exception("Data format error: '.' must be at most one!")
        elif a0 == 0:
            int_left = int(self.distance)
            if int_left > 999:
                raise Exception("Data format error: the integer must be 0 to 3 digits!")
            else:
                int_num = decToBin(int_left, 12)
        else:
            a1 = self.distance.find('.')  # 小数点的位置
            str_left = self.distance[0:a1]
            str_right = self.distance[a1 + 1:]
            if str_left.isdigit() and str_right.isdigit():
                int_left = int(str_left)
                int_right = int(str_right)
                if int_left > 999:
                    raise Exception("Data format error: the integer must be 0 to 3 digits!")
                elif int_right > 999:
                    raise Exception("Data format error: the decimal must be 0 to 3 places!")
                else:
                    int_num = decToBin(int_left, 12)
                    dec_num = decToBin(int_right, 10)
            else:
                raise Exception("Data format error: Incomplete data!")
        return int_num, dec_num

    def getIntCenter(self):
        """
        获取整数部分编码圆点的中心坐标
        Returns: 整数部分编码圆点的中心坐标列表

        """
        startx, starty = self.n0_x, self.n0_y
        d = self.diameter
        center_list = []

        for i in range(0, 4):
            if i % 3 == 0:
                startx += 3 * d
            else:
                startx += 2 * d
            if i < 3:
                center_list.append([startx, starty])
        for j in range(0, 4):
            if j % 3 == 0:
                starty += 3 * d
            else:
                starty += 2 * d
            if j < 3:
                center_list.append([startx, starty])
        for i in range(0, 4):
            if i % 3 == 0:
                startx -= 3 * d
            else:
                startx -= 2 * d
            if i < 3:
                center_list.append([startx, starty])
        for j in range(0, 4):
            if j % 3 == 0:
                starty -= 3 * d
            else:
                starty -= 2 * d
            if j < 3:
                center_list.append([startx, starty])

        return center_list

    def drawTemplateNode(self, image):
        """
        绘制四个模板点

        Returns:

        """
        x0, y0 = self.n0_x, self.n0_y  # N0的中心坐标
        x1, y1 = x0 + 10 * self.diameter, y0  # N1的中心坐标
        x2, y2 = x0, y0 + 10 * self.diameter  # N2的中心坐标
        x3, y3 = x0 + 10 * self.diameter, y0 + 10 * self.diameter  # N3的中心坐标

        # 绘制N0
        cv2.circle(image, (x0, y0), int(1.5 * self.diameter), (255, 255, 255), -1)
        cv2.circle(image, (x0, y0), self.diameter, (0, 0, 0), -1)

        # 绘制N1
        cv2.circle(image, (x1, y1), int(1.5 * self.diameter), (255, 255, 255), -1)
        cv2.circle(image, (x1, y1), self.diameter, (0, 0, 0), -1)
        cv2.circle(image, (x1, y1), int(self.diameter / 2), (255, 255, 255), -1)

        # 绘制N2
        cv2.circle(image, (x2, y2), int(1.5 * self.diameter), (255, 255, 255), -1)
        cv2.circle(image, (x2, y2), self.diameter, (0, 0, 0), -1)
        cv2.circle(image, (x2, y2), int(self.diameter / 2), (255, 255, 255), -1)

        # 绘制N3
        cv2.circle(image, (x3, y3), int(1.5 * self.diameter), (255, 255, 255), -1)
        cv2.circle(image, (x3, y3), self.diameter, (0, 0, 0), -1)
        cv2.circle(image, (x3, y3), int(self.diameter / 2), (255, 255, 255), -1)

    def drawCodePoints(self, image):
        """
        绘制编码图案
        """
        # 数据编码------数据切割、校验、格式化
        integer, decimal = self.encodeDistance()

        # 1 绘制整数部分的编码点
        int_center = self.getIntCenter()
        index_int = 0
        while integer.find('1', index_int) != -1:
            index_int = integer.find('1', index_int)
            center = int_center[index_int]
            cv2.circle(image, center, self.diameter // 2, (255, 255, 255), -1)
            index_int += 1

        # 2 绘制小数部分的编码点------圆环带到中心点的距离为2.5d-3.5d
        index_dec = decimal.find('1')
        while index_dec != -1:
            cv2.ellipse(image, (self.locate_x, self.locate_y), (int(3.5 * self.diameter), int(3.5 * self.diameter)), 0, 36 * index_dec, 36 * (index_dec + 1), (255, 255, 255),
                        -1)
            index_dec = decimal.find('1', index_dec + 1)

        cv2.circle(image, (self.locate_x, self.locate_y), int(2.5 * self.diameter), (0, 0, 0), -1)

    def drawDirection(self, image):
        """
        绘制标识方向的圆A （A的直径为d，圆A与定位圆O相邻，即圆心距|AO|=2d）
        """
        # 计算圆A的中心坐标x,y
        x0, y0 = self.locate_x, self.locate_y  # 定位圆的中心坐标
        y = y0 + int(1.5 * self.locate_radius * math.sin(math.radians(self.angle)))  # 绘制圆点时，中心坐标就有了误差，再后续的方向识别过程中只能精确到整数
        x = x0 + int(1.5 * self.locate_radius * math.cos(math.radians(self.angle)))
        cv2.circle(image, (x, y), int(1 / 3 * self.locate_radius), (255, 255, 255), -1)

    def identifyAngle(self, image):
        M = cv2.getRotationMatrix2D([self.locate_x, self.locate_x], 45 + self.angle, 1.0)
        cv2.warpAffine(image, M, (self.side, self.side), image)

    def create(self):
        canvas = np.zeros((self.side, self.side), np.uint8)
        # 绘制模板点
        self.drawTemplateNode(canvas)
        # 绘制编码点
        self.drawCodePoints(canvas)
        # 标识方向
        self.identifyAngle(canvas)
        # 绘制定位圆
        cv2.circle(canvas, [self.locate_x, self.locate_y], self.locate_radius, (255, 255, 255), -1)
        return canvas

if __name__ == "__main__":
    distance = input("请输入编码距离：")
    angle = float(input("请输入编码角度："))
    fileName = input("请输入文件名称：")
    dirPath = "./images/generate/{}".format(datetime.date.today().strftime("%y%m%d"))
    folder = os.path.exists(dirPath)
    if not folder:
        os.mkdir(dirPath)

    marker = Marker(distance, angle)
    image = marker.create()

    savePath = dirPath + '/' + fileName + '.bmp'

    cv2.imwrite(savePath, image)
    print('Create and Save successfully!')
