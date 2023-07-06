import cv2
from customFunction import showPic

# 读取图像
imgName = "marker0-20"
img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
# enhance = clahe.apply(img)
# blur = cv2.GaussianBlur(img, (3,3), 1)

# 多尺度高斯金字塔处理
img_gaussian_pyramid = [img]
for i in range(4):
    pyr = cv2.pyrDown(img_gaussian_pyramid[i])
    img_gaussian_pyramid.append(pyr)
    cv2.imwrite("../images/process/0428/multiScale/" + imgName + "_down" + str(i) + ".png", pyr)

img_resize_start = img_gaussian_pyramid[4]
img_resize = [img_gaussian_pyramid[4]]
for j in range(3, -1, -1):
    temp = cv2.resize(cv2.pyrUp(img_resize_start), img_gaussian_pyramid[j].shape[:2][::-1])
    cv2.imwrite("../images/process/0428/multiScale/" + imgName + "_resize" + str(j) + ".png", temp)
    img_resize.append(temp)
    img_resize_start = img_gaussian_pyramid[j]

# Canny边缘检测
edges = []
for i in range(5):
    edge = cv2.Canny(img_gaussian_pyramid[i], 20, 50)
    edges.append(edge)
    cv2.imwrite("../images/process/0428/multiScale/" + imgName + "_edge" + str(i) + ".png", edge)


# 多尺度边缘检测结果合并
multi_scale_edges = edges[4]
for i in range(3, -1, -1):
    # img_upsampling = cv2.pyrUp(multi_scale_edges)
    img_upsampling = cv2.resize(multi_scale_edges, edges[i].shape[:2][::-1])
    cv2.imwrite("../images/process/0428/multiScale/" + imgName + "_up" + str(i) + ".png", img_upsampling)
    multi_scale_edges = cv2.addWeighted(edges[i], 0.5, img_upsampling, 0.5, 0)

res = []
for j in range(5):
    p = img_gaussian_pyramid[j]
    q = img_resize[4-j]
    cv2.imwrite("../images/process/0428/multiScale/"+imgName+"_res"+str(j)+".png", p-q)

# 显示结果
# showPic('Multi-Scale Canny Edges', multi_scale_edges)
cv2.imwrite("../images/process/0428/multiScale/"+imgName+"_mul.png", multi_scale_edges)



