import cv2
import numpy as np

'''
扩展缩放
'''
# img = cv2.imread('images/flowerx.png')
# # 下面的None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
# # 因此这里为None
# res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# # OR
# # 直接设置输出图像的尺寸，所以不用设置缩放因子
# height, width = img.shape[:2]
# res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
# while (1):  # 注意缩进
#     cv2.imshow('res', res)
#     cv2.imshow('img', img)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

'''
变换 ，简单旋转45
'''

# image = cv2.imread("images/flower.tif", 0)
# rows, cols = image.shape
# M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.6)
# # cv2.imshow("1", M)
# # 第三个参数是输出图像的尺寸中心
# dst = cv2.warpAffine(image, M, (cols, rows))
# cv2.imwrite('before.jpg', dst)
# cv2.imshow('after', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
仿射变换
'''
# img = cv2.imread('images/news.jpg')
# rows, cols, ch = img.shape
# pts1 = np.float32([[225, 150], [670, 17], [225, 920]])
# pts2 = np.float32([[225, 150], [755, 150], [225, 920]])
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('Input', img)
# cv2.imshow('Output', dst)
# cv2.imwrite('getAffineTransformImg.jpg', dst)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

'''
透视变换
'''

# img = cv2.imread('images/news.jpg')
# rows, cols, ch = img.shape
# pts1 = np.float32([[225, 150], [670, 17], [225, 920], [680, 825]])
# pts2 = np.float32([[225, 150], [755, 150], [225, 920], [755, 920]])
# M = cv2.getPerspectiveTransform(pts1, pts2)
# dst = cv2.warpPerspective(img, M, (1000, 1000))
# cv2.imshow('Input2', img)
# cv2.imshow('Output2', dst)
# cv2.imwrite('getPerspectiveTransformImg.jpg', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''
思考题
'''

Point = []
rectangle = [(0, 0), (0, 300), (400, 0), (400, 300)]


def record(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        Point.append((x, y))


def affineTransform(img):
    pts1 = np.float32(Point)
    pts2 = np.float32(rectangle)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (400, 300))
    return dst


if __name__ == '__main__':
    image = cv2.imread("images/news.jpg")
    cv2.imshow("input", image)
    rows, cols = image.shape[:2]

    cv2.setMouseCallback("input", record)
    while 1:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
        if len(Point) == 4:
            dst = affineTransform(image)
            break

    cv2.imshow("output", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
