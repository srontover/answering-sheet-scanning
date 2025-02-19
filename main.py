import cv2 as cv
import numpy as np
import scanFunctions as sf

################################
path = "test.jpg"
Webcam = False
height = 640
width = 480
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10, 150)
window_info = [
    {
        "window_name": "threshold1",
        "trackbars": [("thres_min1", 0, 255), ("thres_max1", 255, 255)]
    },
    {
        "window_name": "threshold2",
        "trackbars": [("thres_min2", 0, 255), ("thres_max2", 255, 255)]
    }
]

window_info_result = [
    {
        "window_name": "threshold1",
        "trackbars": ["thres_min1", "thres_max1"]
    },
    {
        "window_name": "threshold2",
        "trackbars": ["thres_min2", "thres_max2"]
    }
]

ques = 7
choice = 5
ans = [1,1,1,1,1,1,1]
ans_limit = 2000
#################################


sf.initTrackbars(window_info)


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
    result = sf.valueTrackbars(window_info_result)
    thres = result["threshold1"]
    
    # thres = [100, 100]
    img_thres = cv.Canny(img_blur, 14, thres[1])
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
        
        
        
        
        img_warp_rio = img_warpColored[120:580, 120:442]
        img_warp_rio = cv.resize(img_warp_rio, (490,490))
        img_warpGray = cv.cvtColor(img_warpColored, cv.COLOR_BGR2GRAY)
        thres_warp = result["threshold2"]
        img_warpthres = cv.threshold(img_warpGray, thres_warp[0], 255, cv.THRESH_BINARY_INV)[1]
        
        
        
        
        # print(img_warp_canny.shape)
        img_warp_thres_rio = img_warpthres[120:580, 120:442]
        img_warp_thres_rio = cv.resize(img_warp_thres_rio, (490,490))
        cv.imshow("img_warp_thres_rio", img_warp_thres_rio)
        # print(img_warp_canny_rio.shape)
        boxs = sf.spiltboxs(img_warp_thres_rio)
        # cv.imshow("test",boxs[0])
        # print(cv.countNonZero(boxs[0]))
        # print(cv.countNonZero(boxs[1]))
        
        myPixelVal = np.zeros((ques, choice))
        
        rows = 0
        column = 0
        for box in boxs:
            
            totalPixels = cv.countNonZero(box)
            myPixelVal[rows][column] = totalPixels
            column += 1
            if column == choice:
                rows += 1
                column = 0
        
        myIndex = []
        for x in range(0, ques):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            if arr[myIndexVal[0][0]] > ans_limit:
                myIndex.append(myIndexVal[0][0])
            else:
                myIndex.append(-1)
        # print(myIndex)
        
        grade = []
        for x in range(0, ques):
            if myIndex[x] == ans[x]:
                grade.append(1)
            else:
                grade.append(0)
        # print(grade)
        score = (sum(grade)/ques) * 100
        # print(score)
        cv.putText(img_big_contour, "score: " + str(int(score)) + "%", (20, 80), cv.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)
        # cv.imshow("test", img_warp_thres_rio)
        # cv.imshow("test", boxs[0])
        
        img_result = img_warp_rio.copy()
        img_result = sf.displayAnswers(img_result, myIndex, grade, ans, ques, choice)
        cv.imshow("result", img_result)
        
        imgs = ([img, img_gray, img_thres, img_contours],
            [img_big_contour, img_warpColored, img_warpGray, img_warpthres],
            [img_result, img_blank, img_blank, img_blank])    
           
    else:
        imgs = ([img, img_gray, img_thres, img_contours],
            [img_blank, img_blank, img_blank, img_blank], 
            [img_blank, img_blank, img_blank, img_blank])
    labels = [["oringinal", "gray", "thres", "contours"], 
            ["biggest", "warp colored", "warp gray", "warp thres"],
            ["result", "blank", "blank", "blank"]]
    stackedImages = sf.stackImages(0.5, imgs, labels)
    
            
    cv.imshow("stacked images", stackedImages)
    
    cv.waitKey(1)