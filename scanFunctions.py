import cv2 as cv
import numpy as np

def nothing():
    pass

def initTrackbars(window_info):
    """
    初始化多个窗口和对应的滑动条。

    :param window_info: 一个列表，每个元素是一个字典，包含窗口名称和滑动条信息。
                        滑动条信息是一个列表，每个元素是一个元组，包含滑动条名称、初始值、最大值。
    """
    for info in window_info:
        window_name = info["window_name"]
        trackbars = info["trackbars"]
        # 创建窗口
        cv.namedWindow(window_name)
        # 调整窗口大小
        cv.resizeWindow(window_name, 360, 240)
        for trackbar_name, initial_value, max_value in trackbars:
            # 创建滑动条
            cv.createTrackbar(trackbar_name, window_name, initial_value, max_value, nothing)

def valueTrackbars(window_info):
    """
    从多个窗口中获取滑动条的值。

    :param window_info: 一个列表，每个元素是一个字典，包含窗口名称和滑动条信息。
                        滑动条信息是一个列表，每个元素是一个字符串，表示滑动条名称。
    :return: 一个字典，键为窗口名称，值为该窗口中滑动条的值组成的列表。
    """
    result = {}
    for info in window_info:
        window_name = info["window_name"]
        trackbars = info["trackbars"]
        values = []
        for trackbar_name in trackbars:
            value = cv.getTrackbarPos(trackbar_name, window_name)
            values.append(value)
        result[window_name] = values
    return result


def biggestContour(contours):
    max_area = 0
    biggest = np.array([])
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 5000:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def drawRectangle(img, biggest, thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def stackImages(scale, imgArray, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(labels[d][c])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv.FILLED) 
                cv.putText(ver, labels[d][c], (eachImgWidth*c+10, eachImgHeight*d+20), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

def spiltboxs(img):
    rows = np.vsplit(img, 7)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

def displayAnswers(img, myIndex, grading, ans, ques=7, choice=5):
    sectionWidth = int(img.shape[1]/choice)
    sectionHeight = int(img.shape[0]/ques)
    for x in range(0, ques):
        myAns = myIndex[x]
        cx, cy  = myAns * sectionWidth + sectionWidth//2, (x+1) * sectionHeight - sectionHeight//2
        if grading[x] == 1:
            myColor = (0, 255, 0)
            cv.circle(img, (cx, cy), 25, myColor, cv.FILLED)
        else:
            myColor = (0, 0, 255)
            rightAns = ans[x]
            cv.circle(img, (cx, cy), 25, myColor, cv.FILLED)
            cv.circle(img, ((rightAns * sectionWidth + sectionWidth//2), (x+1) * sectionHeight - sectionHeight//2), 10, (0, 255, 0), cv.FILLED)
            
    return img