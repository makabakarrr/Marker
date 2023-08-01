import cv2
import numpy as np

img = cv2.imread("../images/generate/230708/line_8_1.bmp", 0)
M = cv2.getRotationMatrix2D([199, 199], -2.3, 1.0)
rotate = cv2.warpAffine(img, M, (400, 400), flags=cv2.INTER_NEAREST)

ret, thresh = cv2.threshold(rotate, 200, 255, cv2.THRESH_BINARY)
cv2.imwrite("../images/generate/230708/line_8_1_rotate_2.bmp", rotate)
cv2.imwrite("../images/generate/230708/line_8_1_rotate_2_thresh.bmp", thresh)