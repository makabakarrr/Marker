import cv2
import numpy as np

def generate_kernel(sigma1, sigma2, theta, loc):
    # 当sigma1=sigma2是各向同性，反之为各向异性，theta是旋转角度，loc是高斯核坐标(-loc,loc)
    thetaMatrix = np.dot(np.matrix([[sigma1 ** 2, 0], [0, sigma2 ** 2]]), np.identity(2))
    # 旋转矩阵
    rotationMatrix = np.matrix([[np.cos(theta * np.pi / 180), -1 * np.sin(theta * np.pi / 180)],
                                [np.sin(theta * np.pi / 180), np.cos(theta * np.pi / 180)]])
    # 协方差矩阵
    covMatrix = np.dot(np.dot(rotationMatrix, thetaMatrix), rotationMatrix.transpose())
    # 高斯核在loc位置的值
    k_value = np.exp(-0.5 * np.dot(np.dot(loc.transpose(), np.linalg.inv(covMatrix)), loc))
    return k_value


def Anisotropic_Gaussian(img, size=10, sigma1=1, sigma2=2, theta=30):
    # 生成高斯核坐标 ，size是高斯核的尺寸+1除2
    X, Y = np.meshgrid(np.linspace(-size, size, size * 2 + 1), np.linspace(-size, size, size * 2 + 1))
    coors = np.concatenate((X[:, :, None], Y[:, :, None]), axis=-1)
    # 生成高斯核
    kernel = np.zeros((size * 2 + 1, size * 2 + 1))
    for i in range(0, size * 2 + 1):
        for j in range(0, size * 2 + 1):
            kernel[i, j] = generate_kernel(sigma1, sigma2, theta, coors[j, i])
    kernel = kernel / kernel.sum()  # 归一化一下

    # 对图像进行高斯滤波，这里默认sigma1=1,sigma=2，是个各向异性滤波
    img_res = cv2.filter2D(img, None, kernel=kernel)
    return img_res


# 读取图像
# imgName = "marker0-20"
# img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)
# blur = cv2.GaussianBlur(img, (5,5), 1.5)
# res1 = cv2.GaussianBlur(img, (5,5), 0.75)
# res2 = cv2.GaussianBlur(img, (5,5), 1)
# res3 = cv2.GaussianBlur(img, (5,5), 1.5)
# res4 = cv2.GaussianBlur(img, (5,5), 2)
# # cv2.imwrite("../images/process/0428/anisotropy/"+imgName+".png", res)
# edges1 = cv2.Canny(res1, 20, 50)
# edges2 = cv2.Canny(res2, 20, 50)
# edges3 = cv2.Canny(res3, 20, 50)
# edges4 = cv2.Canny(res4, 20, 50)
# edges = cv2.Canny(blur, 20, 50)
# cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_edges3x3.png", edges1)
# cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_edges5x5.png", edges2)
# cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_edges7x7.png", edges3)
# cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_edges9x9.png", edges4)
# # cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_edges5.png", edges1)
# result1 = cv2.bitwise_or(edges1, edges2)
# result2 = cv2.bitwise_or(result1, edges3)
# result = cv2.bitwise_or(result2, edges4)
# cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_result_multiScale.png", result)
# cv2.imwrite("../images/process/0428/anisotropy/"+imgName+"_edges.png", edges)
imgName = "marker0-20"
img = cv2.imread('../images/process/0428/marker0-20.png', 0)
blur = cv2.blur(img, (5,5))
cv2.imwrite("../images/process/0428/blur/"+imgName+"-1.png", blur)

