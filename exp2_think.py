import cv2
import numpy


def processGetPeople(image, low, high):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_back = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    imageGray = cv2.equalizeHist(imageGray)
    imageGray = cv2.GaussianBlur(imageGray, (5, 5), 0)
    cv2.imshow("0", imageGray)
    ret, mask_front = cv2.threshold(imageGray, low, high, cv2.THRESH_BINARY)

    # people = cv2.bitwise_and(imageGray, cv2.bitwise_not(mask_front))
    cv2.imshow("1", mask_front)
    return cv2.bitwise_not(mask_front)


def imageFuse(image1, image2, mask):
    height, width, channels = image2.shape
    center = (int(height / 2), int(width / 2))
    # image1 = cv2.bitwise_and(image1, mask)
    # mask=numpy.ones(image1.shape,image1.dtype)
    image1 = image1[100:400, 200:640]
    mask = mask[100:400, 200:640]
    image2 = cv2.seamlessClone(image1, image2, mask, center, cv2.MIXED_CLONE)
    cv2.imshow("3", image2)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    video = cv2.VideoCapture("images/vtest.avi")
    high = 255
    low = 60

    while cap.isOpened():

        ret, image = cap.read()
        ret2, image_video = video.read()
        if ret and ret2:
            mask = processGetPeople(image, low, high)
            imageFuse(image, image_video, mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(low, high)
            cv2.destroyAllWindows()
            break
        elif key == ord('w'):
            if high <= 245:
                high += 10
        elif key == ord('s'):
            high -= 10
        elif key == ord('a'):
            if low >= 10:
                low -= 10
        elif key == ord('d'):
            low += 10
