import cv2
import numpy as np
import time

import matplotlib.pyplot as plt

imgName = "marker_0"
img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# 计算梯度幅值和方向
grad_mag, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

edges = cv2.Canny(img, 40, 120)
indices = np.where(edges != 0)
coordinates = zip(indices[0], indices[1])

sub_x_points = []
sub_y_points = []
time1 = time.time()
for x, y in coordinates:
    a = grad_angle[y][x]
    theta = sobely[y][x] / sobelx[y][x]
    # if 0<= theta < step:
    if (-1/3)<= theta < 1/3:
        x1, y1 = x+1, y
        x2, y2 = x-1, y
    elif 1/3 <= theta < 3:
        x1, y1 = x + 1, y+1
        x2, y2 = x - 1, y-1
    elif theta>=3 or -3<=theta:
        x1, y1 = x, y-1
        x2, y2 = x, y+1
    else:
        x1, y1 = x - 1, y - 1
        x2, y2 = x + 1, y + 1

    grad_mag0 = grad_mag[x, y]
    grad_mag1 = grad_mag[x1, y1]
    grad_mag2 = grad_mag[x2, y2]
    dist = np.sqrt((x-x1)**2+(y-y1)**2)

    # sub_x = x + ((grad_mag1-grad_mag2)/(grad_mag1+grad_mag2-2*grad_mag0))*dist/2*np.cos(a)
    # sub_y = y + ((grad_mag1-grad_mag2)/(grad_mag1+grad_mag2-2*grad_mag0))*dist/2*np.sin(a)
    sub_x = x + (sobelx[y1][x1]*dist + sobelx[y2][x2]*dist)/(sobelx[y1][x1] + sobelx[y2][x2] + sobelx[y][x])
    sub_y = y + (sobely[y1][x1]*dist + sobely[y2][x2]*dist)/(sobely[y1][x1] + sobely[y2][x2] + sobely[y][x])

    sub_x_points.append(sub_x)
    sub_y_points.append(sub_y)

time2 = time.time()
print(time2-time1)

plt.imshow(img, cmap="gray")
plt.scatter(indices[1], indices[0], s=10, marker="*")
# plt.scatter(subpixel_x, subpixel_y, s=10, marker="*")
print(sub_x_points[0], sub_y_points[1])
print(len(indices[0]), len(sub_x_points))
plt.scatter(sub_y_points, sub_x_points, s=10, marker="*", c="red")
plt.show()



