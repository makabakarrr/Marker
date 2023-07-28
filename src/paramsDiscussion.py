import cv2
import numpy as np
import itertools

from utils import calculateCrossPoint
from customFunction import *

from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit





class DesignedMarker:
    def __init__(self, img_points, processed_img, lp):
        # 解码用的图像
        self.process_obj = processed_img

        # 标记点在设计坐标系中的坐标
        src_n0 = [50, 50]
        src_n1 = [250, 50]
        src_n2 = [50, 250]
        src_n3 = [250, 250]
        src_lp = [150, 150]
        self.source_template_points = np.array([src_n0, src_n1, src_n2, src_n3], dtype=np.float32)  # 模板圆的中心点
        self.code_points = generateCodePoints(src_n0, 200)  # 整数编码圆中心的设计坐标
        band_angles = calculateBandAngleByAverage(10)
        self.band_points = getBandCenter(band_angles, src_lp, 200)  # 计算每个编码带中间位置的设计坐标


        # 标记点在图像中的坐标点
        self.img_template_points = img_points
        self.img_combinations = []  # 与source_combination对应的图像坐标的组合
        self.img_n0, self.img_n1, self.img_n2, self.img_n3 = [[p[0], p[1]] for p in img_points]
        self.img_lp = lp
        self.img_int_points = []  # 标记点编码圆中心的图像坐标
        self.img_dec_points = []  # 标记点圆环带中心的图像坐标

        # 标记点设计坐标系与图像坐标系之间的转换矩阵
        self.transform_matrix = np.zeros((2, 3), np.float32)

        # 标记点标识的信息
        self.angle = 0.0
        self.distance = 0.0
        self.center_x, self.center_y = 0.0, 0.0

    def calTransformMatrix(self):
        source_combinations = np.array(
            list(itertools.combinations(self.source_template_points, 3)))  # 从source_points中任选3点的组合方式
        dist_combinations = getDistCombinations(source_combinations, self.source_template_points, self.img_template_points)
        max_dist = 10.0

        # 取误差最小的为变换矩阵
        for i in range(len(source_combinations)):
            MM = calculateAffineMatrix(source_combinations[i], dist_combinations[i])
            s_remain_p, d_remain_p = getRemainPoint(source_combinations[i], self.source_template_points, self.img_template_points)
            cal_dist_x = round(MM[0][0] * s_remain_p[0] + MM[0][1] * s_remain_p[1] + MM[0][2], 4)
            cal_dist_y = round(MM[1][0] * s_remain_p[0] + MM[1][1] * s_remain_p[1] + MM[1][2], 4)
            dist_err = np.sqrt((cal_dist_x - d_remain_p[0]) ** 2 + (cal_dist_y - d_remain_p[1]) ** 2)

            if dist_err < max_dist:
                max_dist = dist_err
                self.transform_matrix = MM


    def calAngle(self):
        angle = 135 - np.arctan2(self.transform_matrix[1,0], self.transform_matrix[0,0])*180/np.pi
        angle = angle if angle>0 else 360+angle
        self.angle = angle
        print("标记点标识的角度为：{}".format(angle))


    def calCenter(self):
        [x_o2, y_o2] = calculateCrossPoint([self.img_n0, self.img_n3], [self.img_n1, self.img_n2])
        self.center_x = (x_o2 + self.img_lp[0]) / 2
        self.center_y = (y_o2 + self.img_lp[1]) / 2
        print("标记点的中心位置：", [self.center_x, self.center_y])


    def calDistance(self):
        ## 整数部分解码
        side_length = np.sqrt((self.img_n0[0] - self.img_n1[0]) ** 2 + (self.img_n0[1] - self.img_n1[1]) ** 2)
        self.img_int_points = cvtCodePoints1(self.code_points, self.transform_matrix)
        int_str = getCodeVal(self.process_obj, self.img_int_points)
        ## 小数部分解码
        dec_points = cvtCodePoints1(self.band_points, self.transform_matrix)
        dec_value = getCodeVal(self.process_obj, dec_points)

        print('标记点的解码结果为:{}.{}'.format(int(int_str, 2), str(int(dec_value, 2)).zfill(3)))


    def recognize(self):
        self.calTransformMatrix()
        self.calAngle()
        self.calCenter()
        self.calDistance()





class MarkerDetector:
    def __init__(self, imgName, img):
        # 初始化
        # 初始化图像相关变量
        self.imgName = imgName  # 图像文件名称
        self.save_prefix = "../images/process/summary/" # 保存路径前缀
        self.img = img  # 图像
        self.height, self.width = img.shape

        self.sobelx, self.sobely = np.zeros((self.height, self.width)), np.zeros((self.height, self.width))
        self.sobel_angle, self.sobel_thresh = np.zeros((self.height, self.width), np.float64), np.zeros((self.height, self.width), np.uint8)    # 图像梯度
        self.edges, self.no_branch = np.zeros((self.height, self.width), np.uint8), np.zeros((self.height, self.width), np.uint8)    # 边缘图像、边缘分割后的图像
        self.edges_points = []  # 图像中边缘点


        self.circle_info = []   # 图像中的圆形参数列表
        self.circle_edge_points = []    # 图像中的每个圆的边缘点列表
        self.circular_edges = []    # 图像中圆环带边缘

        self.radius_kmeans_labels = []  # 图像中的半径聚类标签-----3类
        self.position_labels = []   # 图像中圆的位置聚类标签-----随标记点的编码而变

        self.simple_circle_tree = []    # 图像中的圆形--按位置存储，（每个位置上每种半径的圆只有一个）

        self.start_template_info = []   # 图像中的N0模板圆中心点坐标列表
        self.other_template_info = []   # 图像中的N1、N2、N3模板圆的中心点坐标列表
        self.locate_circle_info = []    # 图像中的定位圆中心坐标列表
        self.code_circle_info = []  # 图像中的编码圆中心坐标列表

        # 画布
        self.colorful_canvas = np.zeros((self.height, self.width, 3))
        self.black_canvas = np.zeros((self.height, self.width), np.uint8)
        self.white_canvas = np.ones((self.height, self.width, 3))*255
        self.process_res = np.zeros((self.height, self.width))


    def preProcess(self):
        self.blur = cv2.GaussianBlur(self.img, (0,0), 1)


    def getEdges(self):
        self.sobelx = cv2.Sobel(self.blur, cv2.CV_64F, 1, 0)
        self.sobely = cv2.Sobel(self.blur, cv2.CV_64F, 0, 1)


        mag, angle = cv2.cartToPolar(self.sobelx, self.sobely, angleInDegrees=True)
        grad_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        th, _ = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tl = max(0, int(th * 0.3))
        self.edges = cv2.Canny(self.blur, tl, th - 10)
        rows, cols = np.where(self.edges == 255)
        self.edges_points = [[x, y] for x, y in list(zip(cols, rows))]
        _, self.sobel_thresh = cv2.threshold(grad_mag, 10, 255, cv2.THRESH_BINARY)
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cv2.morphologyEx(self.sobel_thresh, cv2.MORPH_CLOSE, kernal, self.sobel_thresh)
        cv2.imwrite(self.save_prefix+"/sobelThresh/"+self.imgName+"_sobel-thresh1.png", self.sobel_thresh)

        self.sobel_angle = np.arctan(self.sobely, self.sobelx)


    def delBranch(self, angle_thresh):
        sub_mask1 = self.colorful_canvas.copy()

        cnts0, _ = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # for cnt in cnts0:
        for index in range(len(cnts0)):
            cnt = cnts0[index]
            if len(cnt) > 10:

                cv2.drawContours(sub_mask1, [cnt], -1, (255, 255, 255), 1)
                cv2.drawContours(self.no_branch, [cnt], -1, (255, 255, 255), 1)

                epsilon = 1
                approx = cv2.approxPolyDP(cnt, epsilon, False)
                ### approx可视化
                colors = []
                for i in range(len(approx)):
                    colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
                for ap_index in range(len(approx) - 1):
                    i, j = approx[ap_index][0]
                    ii, jj = approx[ap_index + 1][0]
                    # sub_mask1[j, i, :] = colors[ap_index]
                    cv2.line(sub_mask1, [i, j], [ii, jj], colors[ap_index], 1)

                for index in range(1, len(approx) - 1):
                    p_0 = np.array(approx[index - 1][0])
                    p_1 = np.array(approx[index][0])
                    p_2 = np.array(approx[index + 1][0])
                    l_0 = p_0 - p_1
                    l_1 = p_1 - p_2

                    aa = np.dot(l_0, l_1) / (np.linalg.norm(l_0) * np.linalg.norm(l_1))
                    aa = 1 if aa > 1.0 else aa
                    aa = -1 if aa < -1.0 else aa
                    A = np.degrees(np.arccos(aa))

                    if A > angle_thresh:
                        xx, yy = p_1
                        self.no_branch[yy][xx] = 0

        # cv2.imwrite(self.save_prefix + "approxResult/" + self.imgName + "_cnt_approx.png", sub_mask1)
        # cv2.imwrite(self.save_prefix + "approxResult/" + self.imgName + "_cnt_break.png", sub_mask2)


    def extractCircles(self, max_dist=2, min_k=0.3):
        filter_cnts = self.black_canvas.copy()
        circle_canvas = cv2.cvtColor(self.no_branch, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(self.no_branch, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt_index in range(len(contours)):
            cnt = contours[cnt_index]
            if len(cnt) > 10:
                (cx, cy), (a, b), angle = cv2.fitEllipse(cnt)
                radius = max(a, b) / 2
                if min(a, b) < 20 or max(a, b) > 90:
                    continue
                # 求内点数量
                perimeter = 2 * np.pi * radius
                nums = 0
                test_points = []
                for point in cnt:
                    x, y = point[0]
                    if [x, y] not in test_points:
                        test_points.append([x, y])

                for point in test_points:
                    x, y = point
                    dist = abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius)
                    if dist < max_dist:  # test_point 避免重复
                        nums += 1

                k = nums / perimeter
                # cv2.circle(circle_canvas, (int(cx), int(cy)), int(radius), (0, 0, 255), 1)
                # cv2.putText(circle_canvas, str(radius), (int(cx)+int(radius), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 1)
                if k > min_k:  # 圆形轮廓
                    self.circle_info.append([cx, cy, radius, len(self.circle_edge_points), k])  # [中心坐标x, 中心坐标y, 对应的边缘点索引号, k值]
                    cv2.circle(circle_canvas, (int(cx), int(cy)), int(radius), (0, 0, 255), 1)
                    for p in test_points:
                        i, j = p
                        filter_cnts[j][i] = 255
                    self.circle_edge_points.append(test_points)

        # cv2.imwrite(self.save_prefix + "filterCnts/" + self.imgName + '-filter_cnts.png', filter_cnts)
        # cv2.imwrite(self.save_prefix + "filterCnts/" + self.imgName + '-fit_circle.png', circle_canvas)


    def groupCircles(self, dist_threshold=1.5, radius_threshold=1.5):
        circles = self.circle_info
        num_circles = len(circles)
        groups = [[] for i in range(num_circles)]
        # 设定类别id
        group_id = 0
        for i in range(num_circles):
            # 如果该圆已经被分入某一个类别，则跳过
            if groups[i]:
                continue
            # 新建一个类别
            groups[i].append(group_id)

            # 将与该圆代表同一个圆的其他圆分入该类别
            for j in range(i + 1, num_circles):
                if isSameCircle(circles[i], circles[j], dist_threshold, radius_threshold):
                    groups[j].append(group_id)
                    # groups[j] = group_id
            group_id += 1
        groups_list = np.array(groups).flatten()
        new_circle_info = []
        new_circle_points = []
        for i in range(0, np.max(groups_list) + 1):
            cate_indexs = np.where(groups_list == i)[0]
            cate_circle = []
            # 边缘点去重
            circle_points = []
            for index in cate_indexs:
                cate_circle.append(self.circle_info[index])
                cate_edge_points = self.circle_edge_points[index]
                for point in cate_edge_points:
                    x, y = point
                    if [x, y] not in circle_points:
                        circle_points.append([x, y])
            # 同一组点，求均值
            cate_average_x, cate_average_y, cate_average_radius, cate_sum_k = np.mean(
                np.array(cate_circle)[:, 0]), np.mean(np.array(cate_circle)[:, 1]), np.mean(
                np.array(cate_circle)[:, 2]), np.sum(np.array(cate_circle)[:, 4])
            new_circle_info.append(
                [cate_average_x, cate_average_y, cate_average_radius, len(new_circle_points), cate_sum_k])
            new_circle_points.append(circle_points)

        self.circle_info = new_circle_info
        self.circle_edge_points = new_circle_points


    def classifyArcByRadius(self):
        new_circle_info = [0] * len(self.circle_info)
        data = np.array(self.circle_info, dtype=np.float32)[:, 2]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        retval, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        for cate in range(0, np.max(labels) + 1):
            cate_index = np.where(labels == cate)[0]
            for index in cate_index:
                circle = self.circle_info[index]
                circle.append(cate)
                new_circle_info[index] = circle
        self.radius_kmeans_labels = labels
        self.circle_info = new_circle_info


    def classifyArcByPosition(self):
        circle_centers = np.array(self.circle_info)[:, 0:2]
        db = DBSCAN(eps=20, min_samples=1).fit(circle_centers)
        self.position_labels = db.labels_


    def simplifyCircles(self):
        # 保证每个位置里每一种半径的圆只有一个
        for c in range(np.max(self.position_labels) + 1):
            cate_indexs = np.where(self.position_labels == c)[0]  # 某一个位置上的圆形参数的索引
            cate_circles = np.array(self.circle_info)[cate_indexs]  # 提取某一位置上的圆形参数列表
            circle_radius_labels = cate_circles[:, 5]   # 获取某一个位置上所有圆形的半径分类标签
            sorted_labels = np.argsort(circle_radius_labels)    # 半径类别标签排序 0，1， 2
            sorted_circles = cate_circles[sorted_labels]    # 按0，1，2的标签顺序对当前位置上的所有圆形进行排序
            current_label = int(sorted_circles[0][5])
            cur_pos_circle = []
            cur_radius_circles = []
            for circle_index in range(0, len(sorted_circles)):  # 保留当前位置上的某个半径类别里k值最大的圆
                circle = sorted_circles[circle_index]
                if int(circle[5]) != current_label:
                    k_list = [circle[4] for circle in cur_radius_circles]
                    cur_radius_k = np.argmax(np.array(k_list))
                    cur_pos_circle.append(cur_radius_circles[cur_radius_k])
                    cur_radius_circles = []
                    current_label = int(circle[5])
                cur_radius_circles.append(circle)
            k_list = [circle[4] for circle in cur_radius_circles]
            cur_radius_k = np.argmax(np.array(k_list))
            cur_pos_circle.append(cur_radius_circles[cur_radius_k])
            self.simple_circle_tree.append(cur_pos_circle)


    def classifyCircle(self, rate1, rate2, rate3):
        other_circle_info = []

        for position in self.simple_circle_tree:
            length = len(position)
            r_list = np.array(position)[:, 2]
            if length == 3:
                r1, r2, r3 = np.sort(r_list)
                ratio1 = r2 / r1
                ratio2 = r3 / r1
                if abs(ratio1-rate2/rate1)<0.1 and abs(ratio2-rate3/rate1)<0.1:
                    self.other_template_info.append(position)

            elif length == 2:
                r1, r2 = np.sort(r_list)
                ratio1 = r2 / r1
                if abs(ratio1 - rate3/rate2) < 0.1:
                    self.start_template_info.append(position)
                else:
                    other_circle_info.append(position[0])
            else:
                other_circle_info.append(position[0])

        circle_radius = np.array(other_circle_info)[:, 2]
        radius_max = np.max(circle_radius)

        for circle in other_circle_info:
            if abs(circle[2] - radius_max) > 5:
                self.code_circle_info.append(circle)
            else:
                self.locate_circle_info.append([circle])


    def showClassifyCircle(self):
        classify_canvas = self.colorful_canvas.copy()
        for pos in self.start_template_info:
            for circle in pos:
                circle_edges = self.circle_edge_points[int(circle[3])]
                for p in circle_edges:
                    x, y = p
                    classify_canvas[y, x, :] = (14, 169, 250)  # 橘色
        for pos in self.other_template_info:
            for circle in pos:
                circle_edges = self.circle_edge_points[int(circle[3])]
                for p in circle_edges:
                    x, y = p
                    classify_canvas[y, x, :] = (17, 17, 255)  # 红色
        for pos in self.code_circle_info:
            circle_edges = self.circle_edge_points[int(pos[3])]
            for p in circle_edges:
                x, y = p
                classify_canvas[y, x, :] = (247, 162, 17)  # 蓝色
        for pos in self.locate_circle_info:
            for circle in pos:
                circle_edges = self.circle_edge_points[int(circle[3])]
                for p in circle_edges:
                    x, y = p
                    classify_canvas[y, x, :] = (125, 186, 94)  # 绿色
        cv2.imwrite(self.save_prefix + "/templateCircle/" + self.imgName + '-category.png', classify_canvas)


    def calBySubEdges(self):
        self.start_template_info = getCenterBySubEdges(self.start_template_info, self.circle_edge_points, self.sobelx, self.sobely, self.sobel_angle)
        self.other_template_info = getCenterBySubEdges(self.other_template_info, self.circle_edge_points, self.sobelx, self.sobely, self.sobel_angle)
        self.locate_circle_info = getCenterBySubEdges(self.locate_circle_info, self.circle_edge_points, self.sobelx, self.sobely, self.sobel_angle)


    def calCenters(self):
        self.start_circle_centers = calCentersByAverage(self.start_template_info)
        self.template_circle_centers = calCentersByAverage(self.other_template_info)
        self.locate_circle_centers = calCentersByAverage(self.locate_circle_info)


    def extractCircularEdges(self, center, side_length):
        center_x, center_y = center
        circular_range = self.black_canvas.copy()

        # 提取小数部分的边缘信息
        length = int(side_length / 10 * 3.6)
        locate_radius = int(side_length / 10 * 2.4)
        cv2.circle(circular_range, (int(center_x), int(center_y)), int(length), (255, 255, 255), -1)
        cv2.circle(circular_range, (int(center_x), int(center_y)), int(locate_radius), (0, 0, 0), -1)
        self.circular_edges = cv2.bitwise_and(self.sobel_thresh, circular_range)


    def drawTemplates(self):
        canvas = self.process_res
        drawDonut(canvas, self.start_template_info)
        drawDonut(canvas, self.other_template_info)
        drawDonut(canvas, self.locate_circle_info)

        # self.process_res = cv2.bitwise_or(locate_canvas, cv2.bitwise_or(start_canvas, other_template_canvas))
        cv2.imwrite(self.save_prefix + "/templateCircle/" + self.imgName + "_template.png", canvas)
        cv2.imwrite(self.save_prefix + "/processRes/" + self.imgName + "_template.png", canvas)


    def drawIntPart(self):
        canvas = self.process_res
        for other_circle in self.code_circle_info:
            cv2.circle(canvas, (int(other_circle[0]), int(other_circle[1])), int(other_circle[2]), (255, 255, 255), -1)
        cv2.imwrite(self.save_prefix +  "/decode/"+ self.imgName + "_coded_circles.png", canvas)


    def drawDecPart(self):
        contours, _ = cv2.findContours(self.circular_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 50:
                continue
            cv2.drawContours(self.process_res, [cnt], -1, (255, 255, 255), -1)
        cv2.imwrite(self.save_prefix + "/decode/" + self.imgName + "_circular.png", self.process_res)


    def drawProcessImg(self):
        self.drawTemplates()
        self.drawIntPart()
        self.drawDecPart()

    def detect(self):
        templates = []
        lps = []

        for n0_index in range(0, len(self.start_circle_centers)):
            n0 = self.start_circle_centers[n0_index]
            if len(self.template_circle_centers) < 3 or len(self.locate_circle_centers) < 1:
                print("未检测到标记点")
                continue
            lp, n1, n2, n3 = matchPoint(self.template_circle_centers, n0, self.locate_circle_centers)
            dist_points = np.array([n0, n1, n2, n3], dtype=np.float32)
            templates.append(dist_points)
            lps.append(lp)
            side_length = np.sqrt((n0[0] - n1[0]) ** 2 + (n0[1] - n1[1]) ** 2)
            self.extractCircularEdges(lp, side_length)
            self.drawProcessImg()

        return templates, lps, self.process_res


    def detectMarkerByEdges(self):
        self.preProcess()
        self.getEdges()
        self.delBranch(60)
        self.extractCircles(2, 0.3)
        self.groupCircles()
        self.classifyArcByRadius()
        self.classifyArcByPosition()
        self.simplifyCircles()
        self.classifyCircle(1, 2, 3)  # 实际图像中的模板圆半径比值  1：2：3   加工条件变化后会导致比值变化
        self.showClassifyCircle()
        self.calCenters()

        return self.detect()

    def detectMarkerBySubEdges(self):
        self.preProcess()
        self.getEdges()
        self.delBranch(60)
        self.extractCircles(2, 0.3)
        self.groupCircles()
        self.classifyArcByRadius()
        self.classifyArcByPosition()
        self.simplifyCircles()
        self.classifyCircle(1, 2, 3)  # 实际图像中的模板圆半径比值  1：2：3   加工条件变化后会导致比值变化
        self.showClassifyCircle()
        self.calBySubEdges()
        self.calCenters()

        return self.detect()

            ## 计算对准相机中心平台需要移动的距离
            # move_x = round(97*0.545/83 * (average_y - 1057), 4)
            # move_y = round(97*0.545/83 * (average_x - 960), 4)
            # move_x = round(95*0.53/83 * (average_y - 1071), 4)
            # move_y = round(95*0.53/83 * (average_x - 1214), 4)
            # move_x = round(95 * 0.536 / 82 * (lp[1] - 1067), 6)
            # move_y = round(95 * 0.536 / 82 * (lp[0] - 1214), 6)
            # print("平台X方向移动{}μm,Y方向移动{}μm".format(move_x, move_y))


if __name__ == "__main__":
    imgName = "marker_0"
    img = cv2.imread('../images/process/0428/' + imgName + '.png', 0)
    dm = MarkerDetector(imgName, img)
    templates, lps, img_res = dm.detectMarkerBySubEdges()
    for index in range(0, len(templates)):
        marker_template = templates[index]
        marker_lp = lps[index]
        m = DesignedMarker(marker_template, img_res, marker_lp)
        m.recognize()




