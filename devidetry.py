import cv2
import matplotlib.pyplot as  plt
import numpy


def showImage(Image):
    plt.imshow(cv2.cvtColor(Image, cv2.COLOR_BGR2RGB))
    plt.show()


#
# image = cv2.imread("images/flower2.jpg")
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # 高斯滤波平滑处理
# # cv2.imshow("gray1", gray)
# gray2 = cv2.GaussianBlur(gray, (53, 53), 0, 0)
# #  处理阴影
# # plt.imshow(cv2.cvtColor(gray2, cv2.COLOR_BGR2RGB))
# # plt.show()
# showImage(gray2)
# gray2 = cv2.Canny(gray2, 100, 100)
# # cv2.imshow("gray2", gray2)
#
# gray3 = cv2.addWeighted(gray2, 0.5, gray, 0.5, 0)
# # cv2.imshow("gray3", gray3)
#
# plt.hist(gray3.ravel(), 256)
# plt.show()
# # grayX = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
# # grayY = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
# #
# # absX = cv2.convertScaleAbs(grayX)  # 转回uint8
# # absY = cv2.convertScaleAbs(grayY)
# # gray = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# ret, mask = cv2.threshold(gray3, 45, 255, cv2.THRESH_BINARY)
# # cv2.imshow("mask", mask)
# showImage(mask)
# h, w = mask.shape[:2]
# mask3 = numpy.zeros([h + 2, w + 2], numpy.uint8)
#
# cv2.floodFill(mask, mask3, (100, 120), 255, 100, 50, cv2.FLOODFILL_FIXED_RANGE)
# plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
# plt.show()
#
# (binary, contours) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for i in range(len(binary)):
#     mask2 = cv2.drawContours(image, binary, i, (255, 0, 255), 3, 3)
#
# plt.imshow(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
# plt.show()
# gray4 = cv2.bitwise_and(gray, gray, mask=mask)
# gray4 = cv2.bitwise_not(gray4)
# cv2.imshow("gray4", gray4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# import numpy
# import matplotlib.pyplot as  plt
#
#


image = cv2.imread("images/flower2.jpg")
#  转换成灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 做高斯模糊 降低图像的噪声
guassGray = cv2.GaussianBlur(gray, (53, 53), 0, 0)
showImage(guassGray)
# 边缘检测
borderGray = cv2.Canny(guassGray, 100, 100)
gray2 = cv2.addWeighted(gray, 0.5, borderGray, 0.5, 0)

# 二值化
ret, binaryGray = cv2.threshold(gray2, 45, 255, cv2.THRESH_BINARY)

# roi = cv2.bitwise_not(binaryGray)
target = cv2.bitwise_and(image, image, mask=binaryGray)
showImage(target)

borders, other = cv2.findContours(binaryGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

showImage(binaryGray)

for i in range(len(borders)):
    cv2.drawContours(image, borders, i, (255, 0, 255), 1, 1)
showImage(image)
