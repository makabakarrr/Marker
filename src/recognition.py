import cv2

from customFunction import *
from utils import calculateCrossPoint, getAngle



def recognition(thresh):

    ## 轮廓检测：
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ## 筛选圆形轮廓
    # circle_cnts = getCircleCntsByArea(contours, 300)    # 通过外接圆面积筛选
    circle_cnts = getCircleCntsByRoundness(contours)  # 通过圆度筛选
    drawCircle = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(drawCircle, circle_cnts, -1, (0, 0, 255), 2)
    # showPic("circle", drawCircle)

    # 对圆形轮廓进行分类
    cate1, center1, cate2, center2, cate3, center3 = cateCircles(circle_cnts, thresh)

    if len(center2)<1 or len(center3)<1 or len(center1)<3:
        print("没有检测到标记点！")
        return

    # 从非模板点中将定位圆的信息分离出来,分离后cate3、center3内值包含编码圆和方向圆的信息
    locate_circle, locate_center = getLocateCircles(cate3, center3)
    if len(locate_center)<1:
        print("没有检测到定位圆")
        return

    # 根据设计图的特征，假设模板点坐标如下：
    src_n0 = [50, 50]
    src_n1 = [250, 50]
    src_n2 = [50, 250]
    src_n3 = [250, 250]
    src_lp = [150, 150]
    source_points = np.array([src_n0, src_n1, src_n2, src_n3], np.float32)
    code_points = generateCodePoints(src_n0, 200)  # 根据设计图的比例生成编码点的中心坐标列表
    # band_angles = calculateBandAnglesByTemplatePoint(src_lp, src_n0, src_n1, src_n2, src_n3)    # 计算设计图上每个编码带的起始角度、终止角度
    band_angles = calculateBandAngleByAverage(10)
    bandCenters = getBandCenter(band_angles, src_lp, 200)  # 计算每个编码带中间位置的设计坐标


    for i in range(0, len(center2)):  # 有几个N0说明有几个标记点
        N0 = center2[i]
        lp, n1, n2, n3 = matchPoint(center1, N0, locate_center)  # 查找与N0属于同一个标记点的模板点与定位圆
        cv2.putText(drawCircle, 'n1', (int(n1[0]), int(n1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.putText(drawCircle, 'n2', (int(n2[0]), int(n2[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        dst_points = np.array([N0, n1, n2, n3], np.float32)  # 四个模板点的图像坐标
        # 根据变换矩阵求取图像坐标再进行解码
        M = calculatePerspectiveMatrix(source_points, dst_points)  # 计算投影变换矩阵  设计坐标→图像坐标
        ## 1 整数部分的解码
        # a 根据变换矩阵和编码点的设计坐标计算编码点的图像坐标
        transform_code_points = cvtCodePoints(code_points, M)
        # b 在预处理后的图像中进行解码
        int_value = getCodeVal(thresh, transform_code_points)
        ## 2 小数部分的解码
        # a 坐标转换
        transform_band_points = cvtCodePoints(bandCenters, M)
        for index in range(0, len(transform_band_points)):
            x, y = transform_band_points[index]
            cv2.putText(drawCircle, str(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
        showPic("circular", drawCircle)
        # b 解码
        dec_value = getCodeVal(thresh, transform_band_points)

        print('第{}个标记点的解码结果为:{}.{}'.format(i + 1, int(int_value, 2), str(int(dec_value, 2)).zfill(3)))
        [x_o2, y_o2] = calculateCrossPoint([N0, n3], [n1, n2])
        average_x = (x_o2 + lp[0]) / 2
        average_y = (y_o2 + lp[1]) / 2
        print('第{}个标记点的中心坐标为:{}'.format(i + 1, [average_x, average_y]))

        ## 方向识别
        # 取n1-n3的中点
        angle = getAngle(lp, N0)
        print("方向为：", angle)


        ## 计算对准相机中心平台需要移动的距离
        move_x = 81/51.36*(average_y-1071)
        move_y = 81/51.36*(average_x-1210)
        print("平台X方向移动{}μm,Y方向移动{}μm".format(move_x, move_y))

        # showPic("text", drawCircle)
