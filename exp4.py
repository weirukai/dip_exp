import cv2
import matplotlib.pyplot as  plt
import numpy as np

'''
绘制直方图
'''
# image = cv2.imread("images/flower3.jpg", cv2.IMREAD_ANYCOLOR)
# plt.hist(image.ravel(), 256, [0, 256])
# plt.show()
'''
多通道直方图
'''
# image = cv2.imread("images/flower3.jpg", cv2.IMREAD_ANYCOLOR)
# color = ('blue', 'green', 'red')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([image], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#
#     # plt.hist(image[:, :, i].ravel(), 256, [0, 256])
#     plt.xlim([0, 256])
# plt.show()
'''
累计直方图
'''

# img = cv2.imread('images/flower.tif', 0)
# # flatten() 将数组变成一维
# array = img.flatten()
# hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# # 计算累积分布图  Return the cumulative sum of the elements along the given axis.
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max() / cdf.max()
#
# plt.plot(cdf_normalized, color='b')
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
# plt.show()

'''
5 灰度图像均衡化
'''

# img = cv2.imread('images/Fig5.jpg', 0)
# equ = cv2.equalizeHist(img)
# # hstack 可以水平拼接图片
# res = np.hstack((img, equ))
# cv2.imwrite('hist.jpg', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''
HSI空间亮度均衡化
'''
# img = cv2.imread('images/Fig6.png', 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# hue, light, saturation = cv2.split(img)
# light2 = cv2.equalizeHist(light)
# img2 = cv2.merge([hue, light2,saturation])
#
# img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
# img2 = cv2.cvtColor(img2, cv2.COLOR_HLS2BGR)
# cv2.imshow("1", img)
# cv2.imshow("2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
图像的三通道分别做直方图的均衡化
'''

#
# def equalHistChannel(img, channelNo):
#     channels = cv2.split(img)
#     channels[channelNo] = cv2.equalizeHist(channels[channelNo])
#     return cv2.merge(channels)
#
#
# if __name__ == '__main__':
#     img = cv2.imread("images/Fig6.png")
#     cv2.imshow("0", img)
#     for i in range(0, 3):
#         cv2.imshow(str(i+1), equalHistChannel(img, i))
#     cv2.waitKey(0)

'''

直方图的规定化
'''

img1 = cv2.imread('images/Fig7A.jpg')
img2 = cv2.imread('images/Fig7B.jpg')

img_hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # bgr转hsv
img_hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

color = ('h', 's', 'v')

for i, col in enumerate(color):
    # histr = cv2.calcHist([img_hsv1], [i], None, [256], [0, 256])
    hist1, bins = np.histogram(img_hsv1[:, :, i].ravel(), 256, [0, 256])
    hist2, bins = np.histogram(img_hsv2[:, :, i].ravel(), 256, [0, 256])
    cdf1 = hist1.cumsum()  # 灰度值0-255的累计值数组
    cdf2 = hist2.cumsum()
    cdf1_hist = hist1.cumsum() / cdf1.max()  # 灰度值的累计值的比率
    cdf2_hist = hist2.cumsum() / cdf2.max()

    # diff_cdf = [[0 for j in range(256)] for k in range(256)]  # diff_cdf 里是每2个灰度值比率间的差值
    diff_cdf=np.zeros((256, 256))
    for j in range(256):
        for k in range(256):
            diff_cdf[j][k] = abs(cdf1_hist[j] - cdf2_hist[k])

    lut = [0 for j in range(256)]  # 映射表
    for j in range(256):
        min = diff_cdf[j][0]
        index = 0
        for k in range(256):  # 直方图规定化的映射原理
            if min > diff_cdf[j][k]:
                min = diff_cdf[j][k]
                index = k
        lut[j] = ([j, index])

    h = int(img_hsv1.shape[0])
    w = int(img_hsv1.shape[1])
    for j in range(h):  # 对原图像进行灰度值的映射
        for k in range(w):
            img_hsv1[j, k, i] = lut[img_hsv1[j, k, i]][1]

hsv_img1 = cv2.cvtColor(img_hsv1, cv2.COLOR_HSV2BGR)  # hsv转bgr
hsv_img2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2BGR)

cv2.namedWindow('firstpic', 0)
cv2.resizeWindow('firstpic', 670, 900)
cv2.namedWindow('targetpic', 0)
cv2.resizeWindow('targetpic', 670, 900)
cv2.namedWindow('defpic', 0)
cv2.resizeWindow('defpic', 670, 900)

cv2.imshow('firstpic', img1)
cv2.imshow('targetpic', img2)
# cv2.imshow('img1', img_hsv1)
cv2.imshow('defpic', hsv_img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
