import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# 根据梯度幅值求canny的高阈值
def adaptiveThreshold(pecent_of_edge_pixel):
    h, w = grad_mag.shape[0], grad_mag.shape[1]
    target = w*h*pecent_of_edge_pixel
    total = 0
    thresh_high = np.max(grad_mag)
    while total<target:
        thresh_high -= 1
        total = np.count_nonzero(grad_mag >= thresh_high)
    thresh_low = 0.3*thresh_high
    return thresh_high, thresh_low

# 计算两个点之间的距离和中点坐标
def midpoint(p, q):
    return ((p[0]+q[0])/2, (p[1]+q[1])/2)


# 定义模型函数
def circle_model(points):
    """
        计算由三个点确定的圆的参数（圆心坐标 x, y 和半径 r）
        :param points: 三个点的坐标 [x1, y1; x2, y2; ...]
        :return: 圆的参数，格式为 [x, y, r]，如果三点共线则返回 None
        """
    # 将样本值转换为numpy数组，以便进行线性代数运算
    # (x1, y1), (x2, y2), (x3, y3) = points
    # A = np.array([[2 * x1, 2 * y1, 1],
    #               [2 * x2, 2 * y2, 1],
    #               [2 * x3, 2 * y3, 1]])
    # B = np.array([x1 ** 2 + y1 ** 2,
    #               x2 ** 2 + y2 ** 2,
    #               x3 ** 2 + y3 ** 2])
    # params = np.linalg.solve(A, B)
    # cx = params[0]
    # cy = params[1]
    # r = np.sqrt(cx ** 2 + cy ** 2 + params[2])
    #
    # return np.array([cx, cy, r], dtype=np.float32)
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



# 定义误差函数
def circle_error(model, data):
    (cx, cy, r) = model
    errors = []
    for point in data:
        # 计算每个点到圆心的距离
        d = np.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2)
        error = abs(d - r)
        errors.append(error)
    return np.mean(errors)


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


def ransac_circle_fit(data, iter_num, inlier_thresh):
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
        sample_idx1 = random.sample(range(nums_step), 1)[0]  # 在数据中随机选 3 个点
        sample_idx2 = random.sample(range(nums_step, 2*nums_step), 1)[0]  # 在数据中随机选 3 个点
        sample_idx3 = random.sample(range(2*nums_step, nums), 1)[0]  # 在数据中随机选 3 个点
        p1, p2, p3 = data[sample_idx1], data[sample_idx2], data[sample_idx3]
        # A = np.array(sample_data)

        A = np.hstack((np.array([p1, p2, p3]), np.ones((3, 1))))

        if np.linalg.det(A) == 0:
            continue
        # 根据选定的样本估计圆形模型
        circle = circle_model(A)
        # print("circle", circle)
        # 找出所有符合模型的点(inliers)
        inliers = []
        for point in data:
            if is_inlier(point, circle, inlier_thresh):
                inliers.append(point)

        # 如果当前内点数量比历史最优解更好，那么更新最优解
        # print(len(inliers), len(best_inliers))
        if len(inliers) > len(best_inliers):
            best_circle = circle
            best_length = len(inliers)
            # best_inliers = inliers
        if len(inliers) > iter_num:
            best_circle = circle
            best_length = len(inliers)
            break
        iterations += 1

    return best_circle, best_length



imgName = "marker_0"

img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
enhance = clahe.apply(img)
blurred = cv2.GaussianBlur(img, (0, 0), 1)
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-blur.bmp", blurred)
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
# time1 = time.time()
th, tl = adaptiveThreshold(0.001)
# time2 = time.time()

edges = cv2.Canny(blurred, tl, th)
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-edges.bmp", edges)

sub_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
mask1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# for i in range(len(cnt)):
#     c = cnt[i]
#     if len(c)>400:
#         print(i)

# test_cnt = cnt[200]

rows, cols = [], []
circles = []
circle_center_arr = []
time1 = time.time()
for index in range(len(cnts)):
# index = 34
    cnt = cnts[index]
    if len(cnt)<60 or len(cnt)>1000:
        continue
    test_points = []
    for p in cnt:
        x, y = p[0]
        if [x,y] not in test_points:
            test_points.append([x, y])
    # print(test_points)
    nums = len(test_points)
    ransac_threshold = 1
    n_inliers = int(nums*3/4)
    max_iters = int(nums / 4)
    min_error = 1e9
    best_model = None
    best_inliers = None
    # 迭代 RANSAC
    circle, best_length = ransac_circle_fit(test_points, iter_num=n_inliers, inlier_thresh=ransac_threshold)
    # print("best circle:", circle, index)
    # print("best length:", best_length, n_inliers, nums)
    perimater = np.pi * 2 * circle[2]
    if best_length >= n_inliers and best_length>perimater/2:
        circles.append(circle)
        circle_center_arr.append([circle[0], circle[1]])
    # print(np.array(test_points)[:,0])
    rows.extend(np.array(test_points)[:,0])
    cols.extend(np.array(test_points)[:,1])

time2 = time.time()
# print(circles)
print(time2-time1)

db = DBSCAN(eps=1, min_samples=4).fit(circle_center_arr)
c_s = db.labels_
print(c_s)
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for c in range(np.max(c_s)+1):
#     indexs = np.where(c_s==c)[0]
#     length = len(indexs)
#     if length>1:
#         # 求均值
#         sum_x = 0
#         sum_y = 0
#         sum_a = 0
#         sum_b = 0
#         for i in indexs:
#             sum_x += ell[i][0]
#             sum_y += ell[i][1]
#             sum_a += ell[i][2]
#             sum_b += ell[i][3]
#         center_x = sum_x / length
#         center_y = sum_y / length
#         center_r = (sum_a/length+sum_b/length)/4
#         print(center_x, center_y)
#         cv2.circle(result, (int(center_x), int(center_y)), int(center_r), (0,0,255),1)
#     else:
#         # 判断该圆是否噪声
#         i = indexs[0]
#         x,y,a,b,inner = ell[i]
#         c = 2 * np.pi * np.sqrt((a ** 2 + b ** 2) / 2)
#         if inner/c > 0.5:
#             cv2.circle(result, (int(x), int(y)), int((a+b)/4), (0, 0, 255), 1)
for c in circles:
    # print(c)
    cv2.circle(result, (int(c[0]), int(c[1])), int(c[2]), (0,0,255), 1)
cv2.imwrite("../images/process/0428/adaptiveCanny/" + imgName + "-circles.bmp", result)
plt.imshow(img, cmap="gray")
plt.scatter(rows, cols, s=10, marker="*")
plt.scatter(np.array(circles)[:,0], np.array(circles)[:,1], s=10, marker="*")
plt.show()
