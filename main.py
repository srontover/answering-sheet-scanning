import cv2 as cv
import numpy as np
import scanFunctions as sf

################################
path = "2.jpg"
Webcam = False
height = 640
width = 480
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10, 150)
#################################

sf.initTrackbars(1)
sf.initTrackbars(2)
count = 0

while True:
    
    img_blank = np.zeros((width, height, 3), np.uint8)
    
    if Webcam:
        success, img = cap.read()
    else:
        img = cv.imread(path)
    img = cv.resize(img, (width, height))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    thres = sf.valueTrackbars(1)
    img_thres = cv.Canny(img_blur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    img_dilate = cv.dilate(img_thres, kernel, iterations=2)
    img_erode = cv.erode(img_dilate, kernel, iterations=1)
    
    img_contours = img.copy()
    img_big_contour = img.copy()
    contours, hierarchy = cv.findContours(img_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = sf.biggestContour(contours)
    if biggest.size != 0:
        biggest = sf.reorder(biggest)
        cv.drawContours(img_big_contour, biggest, -1, (0, 255, 0), 20)
        img_big_contour = sf.drawRectangle(img_big_contour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        img_warpColored = cv.warpPerspective(img, matrix, (width, height))
        img_warpColored = img_warpColored[20:img_warpColored.shape[0] - 20, 20:img_warpColored.shape[1] - 20]
        img_warpColored = cv.resize(img_warpColored, (width, height))
        img_contours2 = img_warpColored.copy()
        
        

        img_warpGray = cv.cvtColor(img_warpColored, cv.COLOR_BGR2GRAY)
        img_warpthres = cv.adaptiveThreshold(img_warpGray, 255, 1, 1, 7, 2)
        img_warpthres = cv.bitwise_not(img_warpthres)
        img_warpthres = cv.medianBlur(img_warpthres, 5)

        imgs = ([img, img_gray, img_thres, img_contours],
            [img_big_contour, img_warpColored, img_warpGray, img_warpthres])
        
        img_warp_canny = cv.Canny(img_warpGray,50 , 150)
        img_warp_canny_rio = img_warp_canny[100:, 80:]
        imgWarpCanntRio = img_warp_canny_rio.copy()
        cv.imshow("warp canny rio", img_warp_canny_rio)
        contours_warp, _ = cv.findContours(img_warp_canny_rio, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        vaild_contours = []
        for cnt in contours_warp:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if approx.shape[0] > 7:
                vaild_contours.append(cnt)
        vaild_centers = []
        vaild_radius = []
        for cnt in vaild_contours:
            cv.drawContours(imgWarpCanntRio, cnt, -1, (0, 255, 0), 10)
            ((x, y), radius) = cv.minEnclosingCircle(cnt)
            vaild_centers.append([int(x), int(y)])
            vaild_radius.append(int(radius))
            
        cv.imshow("contours2", imgWarpCanntRio)
        print(vaild_centers)            
    else:
        imgs = ([img, img_gray, img_thres, img_contours],[img_blank, img_blank, img_blank, img_blank])
    labels = [["oringinal", "gray", "thres", "contours"], 
                   ["biggest", "warp colored", "warp gray", "warp thres"]]
    stackedImages = sf.stackImages(0.5, imgs, labels)
    
            
    cv.imshow("stacked images", stackedImages)
    
    cv.waitKey(1)