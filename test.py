import cv2

image = cv2.imread("1.jpg", 1)

image=image+100
image=cv2.equalizeHist(image)
cv2.imshow("1",image)

cv2.waitKey(0)
