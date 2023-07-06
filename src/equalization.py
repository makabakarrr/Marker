import cv2


imgName = "marker_34"
img = cv2.imread('../images/process/0428/'+imgName+'.png', 0)

eq = cv2.equalizeHist(img)
cv2.imwrite("../images/process/0428/process/" + imgName + "-equalizaHist.png", eq)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
enhance = clahe.apply(img)
cv2.imwrite("../images/process/0428/process/"+imgName+"-enhance.png", enhance)
