import cv2
import openpyxl
import numpy as np
import itertools

from customFunction import calculateAffineMatrix, cvtCodePoints1, getRemainPoint

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

    return np.array(dist_combinations)


filePath = "../documents/0706_location_center.xlsx"
wb = openpyxl.load_workbook(filePath)
sheet = wb['Sheet1']


source_points = np.array([[10, 10],[10,30], [34, 30], [34, 10]], dtype=np.float32)
target_point = [[22, 10], [16, 10], [16, 15], [16, 20]]
target_real = []
dist_points = []

for row in range(2, 6):
    d = sheet.cell(row=row, column=4)
    e = sheet.cell(row=row, column=5)

    dist_points.append([round(d.value, 4), round(e.value, 4)])


source_combinations = np.array(list(itertools.combinations(source_points, 3)))
dist_combinations = getDistCombinations(source_combinations, source_points, np.array(dist_points, dtype=np.float32))


angle = 0.0
transform_matrix = np.zeros((2,3), np.float32)
max_dist = 10.0

for i in range(len(source_combinations)):
    MM = calculateAffineMatrix(source_combinations[i], dist_combinations[i])
    s_remain_p, d_remain_p = getRemainPoint(source_combinations[i], source_points, dist_points)
    cal_dist_x = round(MM[0][0] * s_remain_p[0] + MM[0][1] * s_remain_p[1] + MM[0][2], 4)
    cal_dist_y = round(MM[1][0] * s_remain_p[0] + MM[1][1] * s_remain_p[1] + MM[1][2], 4)
    dist_err = np.sqrt((cal_dist_x-d_remain_p[0])**2 + (cal_dist_y-d_remain_p[1])**2)
    print(dist_err, 135 - np.arctan2(MM[1, 0], MM[0, 0]) * 180 / np.pi)
    if dist_err<max_dist:
        max_dist = dist_err
        angle = 135 - np.arctan2(MM[1, 0], MM[0, 0]) * 180 / np.pi
        transform_matrix = MM
angle = angle if angle>0 else 360+angle
targets = cvtCodePoints1(target_point, transform_matrix)
print(targets)

