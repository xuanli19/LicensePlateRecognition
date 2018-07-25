import cv2
import numpy as np
import math

from os.path import dirname, join, basename
import sys
from glob import glob
from sklearn.externals import joblib
bin_n = 16 * 16  # Number of bins

clf = joblib.load("dect.model")

def pic_handle(img, num):
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()
    # 变成灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # binary = cv2.adaptiveThreshold(sobel, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,5)

    # cv2.imwrite(r'C:\Users\lx\Desktop\binary_process1.jpg',binary)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    # cv2.imwrite(r'C:\Users\lx\Desktop\binary_process2.jpg', dilation2)

    # 图形中对象的检测
    image, contours, _ = cv2.findContours(dilation2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite('C:/Users/lx/Desktop/handle/' + str(num) + '/rectangle.jpg', img1)
    region = []
    angle_map = {}
    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积不合适的都筛选掉
        if (area < 8000 or area > 34000):
            continue

        # 轮廓近似
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 应该根据边的比列来判断是否是车牌

        # 计算 围住的矩形 的 高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # height = math.sqrt( (box[0][1]-box[3][1])*(box[0][1]-box[3][1])+(box[0][0]-box[3][0])* (box[0][0]-box[3][0]) )
        # width = math.sqrt( (box[0][1]-box[1][1])*(box[0][1]-box[1][1])+(box[0][0]-box[1][0])* (box[0][0]-box[1][0]) )

        ratio = float(width) / float(height)

        if ratio > 2.4 or ratio < 1.0:
            continue
        if (angle > -80 and angle < -10):
            continue
        if (angle < -80):
            angle = -(angle + 90)
        else:
            angle = -angle
        # 算出来angle就是需要顺时针旋转的角度

        angle_map[str(box)] = angle
        region.append(box)

    # 用绿线画出这些找到的轮廓
    for box in region:


        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)
        x1 = box[xs_sorted_index[0], 0] + 20
        if (x1 <= 0):
            x1 = 0

        x2 = box[xs_sorted_index[3], 0] - 20
        y1 = box[ys_sorted_index[0], 1] + 20
        if (y1 <= 0):
            y1 = 0
        y2 = box[ys_sorted_index[3], 1] - 20
        img_plate = img3[y1:y2, x1:x2]
        img_platecopy = img_plate.copy()
        cv2.imwrite('C:/Users/lx/Desktop/handle/' + str(num) + '/res' + str(x1) + str(y1) + '.jpg', img_plate)

        ## 将其灰度化，然后指定图像尺寸
        im_gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(im_gray, (160,80), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('C:/Users/lx/Desktop/other2/grey' + str(x1) + str(y1)+str(x2)+str(y2) + '.jpg', res)
        cv2.imwrite('C:/Users/lx/Desktop/handle/' + str(num) + '/grey' + str(x1) + str(y1)+str(x2)+str(y2)+'.jpg',res)
        img_gray = cv2.imread('C:/Users/lx/Desktop/other2/grey' + str(x1) + str(y1)+str(x2)+str(y2) + '.jpg')
        sta =clf.predict(   [np.array(img_gray).flatten()/255-1]   )
        # print(sta)
        if( sta==[0]  ):
            continue
        cv2.drawContours(img2, [box], 0, (0, 255, 0), 2) # 判断是车牌再框住它
        # 倾斜纠正
        # ret, binary = cv2.threshold(gray1, 95, 255, cv2.THRESH_BINARY)
        (h, w) = img_platecopy.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -1 * angle_map[str(box)], 1.0)
        rotated = cv2.warpAffine(img_plate, M, (w, h), flags=cv2.INTER_CUBIC
                                 , borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite('C:/Users/lx/Desktop/handle/' + str(num) + '/res' + str(x1) + str(y1) + 'binary.jpg', rotated)
        # 分割字符
        ## 车牌按照论文中的连通域方法来处理 TODO
        img_plate = rotated
        gaussian = cv2.GaussianBlur(img_plate, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

        gray_lap = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
        dst = cv2.convertScaleAbs(gray_lap)
        # cv2.imshow('dst', dst)
        # cv2.waitKey(0)
        gray1 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        im_at_mean = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 5)
        # ret, im_at_mean = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow('im_at_mean', im_at_mean)
        # cv2.waitKey(0)
        cv2.imwrite(
            r'C:/Users/lx/Desktop/handle/' + str(num) + '/rotadect1' + str(x) + str(y) + str(w) + str(h) + '.jpg',
            im_at_mean)

        shape = im_at_mean.shape
        image, contours, _ = cv2.findContours(im_at_mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(binary, contours, -1, (0,255, 0 ), 2)
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            bili = h * 1.0 / w
            if h < 8 or w < 8:
                continue
            if h * 4 < shape[0] or h * 1.2 > shape[0] or w * 6 > shape[1] or w * 15 < shape[1]:
                continue
            if bili < 1.2 or bili > 5:
                continue
            img_plate = cv2.rectangle(img_plate, (x, y), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        # print('C:/Users/lx/Desktop/handle/' + str(num) + '/rotadect' + str(x)+str(y)+str(w)+str(h) + '.jpg')
        cv2.imwrite(
            r'C:/Users/lx/Desktop/handle/' + str(num) + '/rotadect' + str(x) + str(y) + str(w) + str(h) + '.jpg',
            img_plate)
        # cv2.imshow('rotated', img_plate)
        # cv2.waitKey(0)

    cv2.imwrite(r'C:/Users/lx/Desktop/handle/' + str(num) + '/img.jpg', img2)
    cv2.imwrite('C:/Users/lx/Desktop/final/image' + str(num) + '.jpg', img2)
