import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
cv.filter2D函数进行滤波
'''
# img = cv2.imread('images/lena.bmp')
# kernel = np.ones((5, 5), np.float32) / 25
# # cv.filter2D(src, dst, kernel, anchor=(-1, -1))
# # ddepth –desired depth of the destination image;
# # if it is negative, it will be the same as src.depth();
# # the following combinations of src.depth() and ddepth are supported:
# # src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
# # src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
# # src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
# # src.depth() = CV_64F, ddepth = -1/CV_64F
# # when ddepth=-1, the output image will have the same depth as the source.
# dst = cv2.filter2D(img, -1, kernel)
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()


'''
cv2.blur函数进行均值滤波 
'''
# img = cv2.imread('images/lena.bmp')
# blur3 = cv2.blur(img, (3, 3))
# blur5 = cv2.blur(img, (5, 5))
# blur7 = cv2.blur(img, (7, 7))
# plt.subplot(221), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222), plt.imshow(blur3), plt.title('Blurred3*3')
# plt.xticks([]), plt.yticks([])
# plt.subplot(223), plt.imshow(blur5), plt.title('Blurred 5*5')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224), plt.imshow(blur7), plt.title('Blurred 7*7')
# plt.xticks([]), plt.yticks([])
# plt.show()

'''
中值滤波
'''

# img = cv2.imread("images/lena.bmp")
# median3 = cv2.medianBlur(img, 3)
# median5 = cv2.medianBlur(img, 5)
# median7 = cv2.medianBlur(img, 7)
# plt.subplot(221), plt.imshow(img), plt.title("Original")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(222), plt.imshow(median3), plt.title("median3*3")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(223), plt.imshow(median5), plt.title("median5*5")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(224), plt.imshow(median7), plt.title("median7*7")
# plt.xticks([])
# plt.yticks([])
# plt.show()


'''
高斯模糊
'''
# img = cv2.imread("images/lena.bmp")
# median3 = cv2.GaussianBlur(img, (3, 3), 0)
# median5 = cv2.GaussianBlur(img, (5, 5), 0)
# median7 = cv2.GaussianBlur(img, (7, 7), 0)
# plt.subplot(221), plt.imshow(img), plt.title("Original")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(222), plt.imshow(median3), plt.title("Gaussian3*3")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(223), plt.imshow(median5), plt.title("Gaussian5*5")
# plt.xticks([])
# plt.yticks([])
# plt.subplot(224), plt.imshow(median7), plt.title("Gaussian7*7")
# plt.xticks([])
# plt.yticks([])
# plt.show()

'''
空间域图像锐化
'''

# img = cv2.imread('images/moon.jpg', 0)
# # cv2.CV_64F 输出图像的深度（数据类型），可以使用-1, 与原图像保持一致 np.uint8
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# # 参数 1,0 为只在 x 方向求一阶导数，最大可以求 2 阶导数。
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# # 参数 0,1 为只在 y 方向求一阶导数，最大可以求 2 阶导数。
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# cv2.imshow("1",sobelx)
# cv2.waitKey(0)
# plt.show()

'''


傅里叶变换  频率域图像变换
'''

# img = cv2.imread("images/messi5.jpg", cv2.IMREAD_GRAYSCALE)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# cv2.imshow("1", 20*np.log(np.abs(f)))
# cv2.waitKey(0)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()


# img = cv2.imread('images/messi5.jpg', 0)
# f = np.fft.fft2(img)
# #  移动到图像的中心位置
# fshift = np.fft.fftshift(f)
# rows, cols = img.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
# # 重新变换回图像的左上角位置
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# # 取绝对值
# img_back = np.abs(img_back)
# plt.subplot(131), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(img_back, cmap='gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()

'''
图像匹配
'''

img = cv2.imread('images/lena.bmp', 0)
img2 = img.copy()
template = cv2.imread('images/eyes.png', 0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    # exec 语句用来执行储存在字符串或文件中的 Python 语句。
    # 例如，我们可以在运行时生成一个包含 Python 代码的字符串，然后使用 exec 语句执行这些语句。
    # eval 语句用来计算存储在字符串中的有效 Python 表达式
    method = eval(meth)
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
