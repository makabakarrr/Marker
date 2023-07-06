import numpy as np

from customFunction import *

from sklearn.cluster import DBSCAN


class DesignedMarker:
    def __init__(self):
        src_n0 = [50, 50]
        src_n1 = [250, 50]
        src_n2 = [50, 250]
        src_n3 = [250, 250]
        src_lp = [150, 150]
        self.source_points = np.array([src_n0, src_n1, src_n2, src_n3], dtype=np.float32)
        self.int_points = generateCodePoints(src_n0, 200)
        band_angles = calculateBandAngleByAverage(10)
        self.dec_points = getBandCenter(band_angles, src_lp, 200)  # 计算每个编码带中间位置的设计坐标


class MarkerDetector:
    def __init__(self, imgName, img):
        self.imgName = imgName
        self.save_prefix = "../images/process/summary/"
        self.img = img
        self.height, self.width = img.shape

        self.sobelx, self.sobely, self.sobel_angle = np.zeros((self.height, self.width), np.float64)
        self.edges, self.no_branch = np.zeros((self.height, self.width))
        self.edges_points = []
        self.colorful_canvas = np.zeros((self.height, self.width, 3))
        self.black_canvas = np.zeros((self.height, self.width))
        self.white_canvas = np.ones((self.height, self.width, 3))*255

        self.circle_info = []
        self.circle_edge_points = []

        self.radius_kmeans_labels = []
        self.dbscan_labels = []

        self.simple_circle_tree = []



    def preProcess(self):
        self.blur = cv2.GaussianBlur(self.img, (0,0), 1)


    def getEdges(self):
        self.sobelx = cv2.Sobel(self.blur, cv2.CV_64F, 1, 0)
        self.sobely = cv2.Sobel(self.blur, cv2.CV_64F, 0, 1)
        self.sobel_angle = np.arctan(self.sobely, self.sobelx)

        mag, angle = cv2.cartToPolar(self.sobelx, self.sobely, angleInDegrees=True)
        grad_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        th, _ = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tl = max(0, int(th * 0.3))
        self.edges = cv2.Canny(self.blur, tl, th - 10)
        rows, cols = np.where(self.edges == 255)
        self.edges_points = [[x, y] for x, y in list(zip(cols, rows))]


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
        self.dbscan_labels = db.labels_


    def simplifyCircles(self):
        for c in range(np.max(self.radius_kmeans_labels) + 1):
            cate_indexs = np.where(self.radius_kmeans_labels == c)[0]  # 某一类的索引
            cate_circles = np.array(self.circle_info)[cate_indexs]  # 提取某一类内的圆参数
            circle_radius_labels = cate_circles[:, 5]
            sorted_labels = np.argsort(circle_radius_labels)
            sorted_circles = cate_circles[sorted_labels]
            current_label = int(sorted_circles[0][5])
            cur_pos_circle = []
            cur_radiu_circles = []
            for circle_index in range(0, len(sorted_circles)):
                circle = sorted_circles[circle_index]
                if int(circle[5]) != current_label:
                    k_list = [circle[4] for circle in cur_radiu_circles]
                    cur_radius_k = np.argmax(np.array(k_list))
                    cur_pos_circle.append(cur_radiu_circles[cur_radius_k])
                    cur_radiu_circles = []
                    current_label = int(circle[5])
                cur_radiu_circles.append(circle)
            k_list = [circle[4] for circle in cur_radiu_circles]
            cur_radius_k = np.argmax(np.array(k_list))
            cur_pos_circle.append(cur_radiu_circles[cur_radius_k])
            self.simple_circle_tree.append(cur_pos_circle)





