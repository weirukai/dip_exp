import cv2
import matplotlib.pyplot as  plt
import numpy as np


def OTSU(img_gray):
    max_g = 0
    suitable_th = 0
    th_begin = 0
    th_end = 256
    for threshold in range(th_begin, th_end):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        # 前景像素点的所占的比例
        w0 = float(fore_pix) / img_gray.size
        #   前景像素点的平均灰度值
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        # 背景像素所占的比例
        w1 = float(back_pix) / img_gray.size
        # 背景像素点的平均灰度值
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        #  最大类间方差
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold

    return suitable_th


if __name__ == '__main__':
    img = cv2.imread("images/flower2.jpg", cv2.IMREAD_GRAYSCALE)
    threshold = OTSU(img)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    plt.imshow(img)
    plt.show()
