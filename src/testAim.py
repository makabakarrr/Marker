import cv2
import numpy as np

# canvas = np.zeros((600, 600))
# width = 1
# height = 80
# start_rows = (600-width) // 2
# start_cols = (600-height) // 2
# cv2.rectangle(canvas, (start_cols, start_rows), (start_cols+height-1, start_rows+width-1), (255,255,255), -1)
# cv2.rectangle(canvas, (start_rows, start_cols), (start_rows+width-1, start_cols+height-1), (255,255,255), -1)
# cv2.imwrite("../images/generate/230710/cross_1px.bmp", canvas)

# img = cv2.imread("../images/process/location/cross_1_0_0.bmp", 0)
# canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.rectangle(canvas, (977, 950),(977, 958), (0,0,255), -1)
# cv2.rectangle(canvas, (973, 954),(981, 954), (0,0,255), -1)
# cv2.imwrite("../images/process/location/cross_1_0_0_marker.bmp", canvas)

m=1
R=96
def six_variable_function(xx1):
    y = 0.0
    x1, x2, x3, x4, x5, x6 = 86.3, 83.9, 0, 3.54, 0.00125, np.float(xx1)
    x3, x4 = np.radians(x3), np.radians(x4)
    for l in range(350, 851):
        l = round(l*0.001, 4)
        temp = np.clip(m*l/x5 - np.sin(x3), -1, 1)
        y1 = np.arcsin(temp)  # 角度的单位为弧度---β(λq)
        y2 = x2 / (np.cos(x4-y1))
        y3 = np.cos(x3)**2 / x1 + np.cos(y1)**2 / y2 - (np.cos(x3)+np.cos(y1)) / R
        y4 = (y3 + m*l/x5 * x6)**2
        # print("y1", l, x5, m*l/x5, np.sin(x3), m*l/x5 - np.sin(x3))
        # print("y1y2y3y4", y1, y2, y3, y4)
        y += y4

    return y
print(six_variable_function(0.00003))
