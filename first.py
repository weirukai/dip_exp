# # import cv2
# # import matplotlib.pyplot as plt
# #
# # image = cv2.imread("images/test.png", 0)
# # # image = cv2.imread("images/diamond2.jpg", cv2.IMREAD_GRAYSCALE)
# # # print(image.__class__.__name__)
# #
# # if __name__ == '__main__':
# #     x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
# #     y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
# #
# #     absX = cv2.convertScaleAbs(x)  # 转回uint8
# #     absY = cv2.convertScaleAbs(y)
# #     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# #     # dst = cv2.addWeighted(x, 0.5, y, 0.5, 0)
# #
# #     # cv2.imshow("test",cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# #     # cv2.waitKey(0)
# #
# #     plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# #     plt.show()

'''

1.图像的读取与保存
'''
import numpy as np
import cv2

img = cv2.imread('images/test.png', 1)  # 试着修改 0 为 1.
cv2.imshow("image", img)

k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv2.imwrite('test2.jpg', img)
    cv2.destroyAllWindows()

'''
2.读取视频文件
'''
# import numpy as np
# import cv2
# cap = cv2.VideoCapture('images/vtest.avi')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


'''
3读取视频文件，显示视频，保存视频文件
'''
import numpy as np

# import cv2
# import matplotlib.pyplot as plt
#
# cap = cv2.VideoCapture(0)  # 摄像头编号。
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 注意编码器
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.flip(frame, 1)
#         # write the flipped frame
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         plt.show()
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

'''
绘制文字
'''
#
# import cv2
#
# img = cv2.imread('images/test.png', 1)  # 试着修改 0 为 1.
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'OpenCV', (10, 100), font, 3, (255, 255, 255), 2)
# cv2.imshow("image", img)
#
# k = cv2.waitKey(0)
# if k == 27:  # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'):  # wait for 's' key to save and exit
#     cv2.imwrite('test2.jpg', img)
#     cv2.destroyAllWindows()
