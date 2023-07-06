import cv2

# 读取图像
img = cv2.imread('test.jpg')

# 构建高斯金字塔
gaussian_pyramid = [img]
for i in range(3):
    img = cv2.pyrDown(img)
    gaussian_pyramid.append(img)

# 构建拉普拉斯金字塔
laplacian_pyramid = [gaussian_pyramid[3]]
for i in range(3, 0, -1):
    img = cv2.pyrUp(gaussian_pyramid[i])
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], img)
    laplacian_pyramid.append(laplacian)

# 边缘检测
edges = []
for laplacian in laplacian_pyramid:
    edges.append(cv2.Canny(laplacian, 100, 200))

# 边缘链接
for i in range(3):
    x = cv2.pyrUp(edges[i + 1])
    edges[i] = cv2.bitwise_or(edges[i], x)

# 重构图像
reconstructed_image = edges[0]
for i in range(1, 4):
    reconstructed_image = cv2.pyrUp(reconstructed_image)
    reconstructed_image = cv2.bitwise_or(reconstructed_image, edges[i])

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
