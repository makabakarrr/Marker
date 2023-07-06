import cv2
import numpy as np
import random

def ransac_fit(data, inliers_theshold, max_iterations, dist_threshold):
    best_center, best_radius = None, 0
    iterations = 0
    best_inliers_nums = 0
    while iterations < max_iterations:
        # 选择三个随机点
        nums = len(data)
        # nums_step = nums // 3
        # sample_idx1 = random.sample(range(nums_step), 1)[0]  # 在数据中随机选 3 个点
        # sample_idx2 = random.sample(range(nums_step, 2 * nums_step), 1)[0]  # 在数据中随机选 3 个点
        # sample_idx3 = random.sample(range(2 * nums_step, nums), 1)[0]  # 在数据中随机选 3 个点
        # p1, p2, p3 = data[sample_idx1], data[sample_idx2], data[sample_idx3]
        sample_idx = random.sample(range(nums), 3)
        sample_data = [data[i] for i in sample_idx]
        print("sample_data:", sample_data)
        p1, p2, p3 = sample_data[0], sample_data[1], sample_data[2]

        # 估计外接圆心和半径
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # 计算两点之间的距离
        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)

        # 计算圆心坐标
        s = (a + b + c) / 2
        radius = a * b * c / (4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))
        center_x = ((y2 - y1) * c ** 2 + (y1 - y3) * b ** 2 + (y3 - y2) * a ** 2) / (2 * (x2 - x1) * c + 2 * (x1 - x3) * b + 2 * (x3 - x2) * a)
        center_y = ((x2 - x1) * c ** 2 + (x1 - x3) * b ** 2 + (x3 - x2) * a ** 2) / (2 * (y2 - y1) * c + 2 * (y1 - y3) * b + 2 * (y3 - y2) * a)

        # 计算所有数据点到圆心的距离
        distances = np.abs(np.sqrt((np.array(data)[:, 0] - center_x) ** 2 + (np.array(data)[:, 1] - center_y) ** 2) - radius)

        # 计算内点和外点的数量
        inliers = np.array(data)[distances < dist_threshold]
        n_inliers = len(inliers)
        print("n_inliers:", (center_x, center_y, radius), n_inliers)
        # 记录当前最佳拟合圆的参数
        if n_inliers > inliers_theshold:
            best_center = (center_x, center_y)
            best_radius = radius
            break
        else:
            if n_inliers > best_inliers_nums:
                best_inliers_nums = n_inliers
                best_center = (center_x, center_y)
                best_radius = radius


        iterations += 1

    return best_center, best_radius


# 定义模型函数
def circle_model(points):
    """
        计算由三个点确定的圆的参数（圆心坐标 x, y 和半径 r）
        :param points: 三个点的坐标 [x1, y1; x2, y2; ...]
        :return: 圆的参数，格式为 [x, y, r]，如果三点共线则返回 None
        """
    # 计算行列式
    p1, p2, p3 = points
    A = np.array([p1, p2, p3])

    # 计算各点间距离的平方
    distances = np.sum(A ** 2, axis=1)

    # 使用矩阵计算圆心和半径
    center = np.linalg.solve(2*A, distances)
    # center = np.linalg.solve(A, -1*distances)
    radius = np.sqrt(np.sum((p1[:2] - center[:2]) ** 2))
    # print("radius:", p1, center)

    return np.array([center[0], center[1], radius], dtype=np.float32)


def is_inlier(point, circle, d):
    """
    判断一个点是否为内点(inlier)
    :param point: 要判断的点，用二元组(x, y)表示
    :param circle: 一个元组(center, r)，表示圆心坐标和半径
    :param d: 内点阈值
    :return: 如果点在圆内，则返回True，否则返回False
    """

    distance = abs(np.sqrt((point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2) - circle[2])
    # print("distance:", distance)
    return distance <= d


def ransac_circle_fit(data, iter_num, inlier_thresh, dist_threshold):
    """
    使用 RANSAC 算法进行圆拟合
    :param data: 数据数组，每行一个点的坐标 [x1, y1; x2, y2; ...]
    :param iter_num: 迭代次数
    :param inlier_thres: 阈值，判断是否为内点的阈值
    :return: 拟合出来的圆的参数：圆心坐标 (x, y) 和半径 r
    """
    iterations = 0

    best_circle = None
    best_inliers = []
    best_length = 0

    while iterations < iter_num:
        # 随机选取n个点作为样本
        nums = len(data)
        nums_step = nums // 3
        # sample_idx1 = random.sample(range(nums_step), 1)[0]  # 在数据中随机选 3 个点
        # sample_idx2 = random.sample(range(nums_step, 2*nums_step), 1)[0]  # 在数据中随机选 3 个点
        # sample_idx3 = random.sample(range(2*nums_step, nums), 1)[0]  # 在数据中随机选 3 个点
        # p1, p2, p3 = data[sample_idx1], data[sample_idx2], data[sample_idx3]
        # A = np.array(sample_data)
        sample_idx = random.sample(range(nums_step), 3)
        sample_data = [data[i] for i in sample_idx]
        # print("sample_data:", sample_data)
        p1, p2, p3 = sample_data[0], sample_data[1], sample_data[2]

        A = np.hstack((np.array([p1, p2, p3]), np.ones((3, 1))))

        if np.linalg.det(A) == 0:
            continue
        # 根据选定的样本估计圆形模型
        circle = circle_model(A)
        # print("circle", circle)
        # 找出所有符合模型的点(inliers)
        inliers = []
        for point in data:
            if is_inlier(point, circle, dist_threshold):
                inliers.append(point)

        # 如果当前内点数量比历史最优解更好，那么更新最优解
        # print(len(inliers), len(best_inliers))

        if len(inliers) > inlier_thresh:
            best_circle = circle
            best_length = len(inliers)
            break
        else:
            if len(inliers) > len(best_inliers):
                best_circle = circle
                best_length = len(inliers)
                # best_inliers = inliers
        iterations += 1

    return best_circle



