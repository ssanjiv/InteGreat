import cv2
import math
import numpy as np
DEBUG = True

def init():
    camFeed = cv2.VideoCapture(1)
    return camFeed

def getImg(camFeed):
    flag, image = camFeed.read()
    if DEBUG:
        cv2.imwrite("out.png", image)
    return image

def getROI(image):
    #perform low pass filter to start segmnting white board
    lowBound = np.array((100,100,100), dtype="uint8")
    highBound = np.array((255,255,255), dtype="uint8")
    mask = cv2.inRange(image, lowBound, highBound)
    boundedImage = cv2.bitwise_and(image, image, mask = mask)
    if DEBUG:
        cv2.imwrite("bound.png", boundedImage)

    #bianerize remaining components
    temp = cv2.cvtColor(boundedImage, cv2.COLOR_BGR2GRAY)
    bianImage = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    bianImage = cv2.bitwise_and(bianImage, bianImage, mask=mask)
    if DEBUG:

        cv2.imwrite("bian.png", bianImage)

    #mid pass filter, assuming whiteboard is largest remaining connected compnent
    ccMask = bianImage.copy()
    contours, heirarchy = cv2.findContours(bianImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    eroCount = 0
    eroKern= np.ones((10,10),np.uint8)

    while len(contours) > 5:
            eroCount+=1
            ccMask = cv2.erode(ccMask, eroKern)
            contCopy = ccMask.copy()
            contours, heirarchy = cv2.findContours(contCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG:
        cv2.imwrite("ccMaskEro.png", ccMask)

    #now that largest cc is found, redilate to create the second mask
    eroCountReductionFactor = int(math.ceil(eroCount * .33))
    for _ in range(0, eroCount - eroCountReductionFactor):
            eroCount-=1
            ccMask = cv2.dilate(ccMask, eroKern)

    if DEBUG:
        cv2.imwrite("ccMask.png", ccMask)

    #the largest white connected component can now be found by running another threshold
    #after this, we fill the area of that contour and return that as our ROI mask
    contours, heirarchy = cv2.findContours(ccMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    COI = contours[0]
    x, y, width, height = cv2.boundingRect(COI)
    for row in range(0, ccMask.shape[0]):
        for col in range(0, ccMask.shape[1]):
            if row > y and row < y + height and col > x and col < x+width:
                ccMask[row, col] = 255
            else:
                ccMask[row, col] = 0
    if DEBUG:
        cv2.imwrite("maskFinal.png", ccMask)

    return [ccMask, x, y, width, height]

def reduceToROI(image, ROI):
    boundedImage = cv2.bitwise_and(image, image, mask = ROI)

    if DEBUG:
        cv2.imwrite('reducedImage.png', boundedImage)
    return boundedImage

def reduceImgToCharList(boundedImg, ROIValList):
    #use adaptive gaussian to bianerize image
    grImg= cv2.cvtColor(boundedImg, cv2.COLOR_BGR2GRAY)
    bianImage = cv2.adaptiveThreshold(grImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    bianImage = (255 - bianImage)
    dilKern = np.ones((2,2),np.uint8)
    kern = np.ones((12,12),np.uint8)
    #erode image, then redilate, to create individual expression masks
    bianImage = cv2.erode(bianImage, dilKern)
    bianImage = cv2.dilate(bianImage, kern)
    bianImage = cv2.dilate(bianImage, kern)
    if DEBUG:
        cv2.imwrite('grimg.png', bianImage)

    #extract the contours in the image
    contours, heirarchy = cv2.findContours(bianImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #perform a filter on the contours and extract the 'letter like' ones
    letterList= []
    totalArea = boundedImg.shape[0] * boundedImg.shape[1]
    #low pass filter
    for contour in contours:
    #    if cv2.contourArea(contour) < .1 * bianImage.shape[0] * bianImage.shape[1]:
            letterList.append(contour)

    #mid pass filter
    finalList = []
    counter = 0
    for contour in letterList:
        mask = np.zeros(bianImage.shape, np.uint8)
        x, y, w, h = cv2.boundingRect(contour)
        #calculate the aspect ratio, and remove oblong chars
        #additionally, remove anything larger than 20% of the image
        aspect = float(w)/h
        area = w * h
        imgArea = bianImage.shape[0] * bianImage.shape[1]
        if aspect > .3 and aspect < 1.6 and area < float(imgArea) * .2:
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            name = str(counter) + 'letter.png'
            counter+=1
            if len(finalList) == 0:
                finalList.append([x, y, w, h])
            else:
                for letter in finalList:
                    if not (abs(x - letter[0]) < 10 and abs(y - letter[1]) < 10):
                        finalList.append([x, y, w, h])
                        cv2.imwrite(name, mask)

    returnList = []
    num = 0
    for letter in finalList:
        x = letter[0]
        y = letter[1]
        w = letter[2]
        h = letter[3]
        copyImg = boundedImg.copy()
        cropped = copyImg[y:y+h, x:x+w]
        if DEBUG:
            name = str(num) + 'midChar.png'
            cv2.imwrite(name, cropped)
            num +=1
        returnList.append(cropped)

    return returnList

def resolveCharListToBinary(myCharList):
    bianCharList = []
    num = 0
    for char in myCharList:
        temp = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
        #assuming the distribution of pizels is bimodal, splitting across
        #the pure mean will split the modes
        #mean
        tot = 0
        totNum= 0
        print temp.shape
        for row in range(0, temp.shape[0]):
            for col in range(0, temp.shape[1]):
                tot += temp[row][col]
                totNum += 1
        if not totNum == 0:
            print tot/totNum
            cv2.threshold(temp, tot/num, 255, cv2.THRESH_BINARY)
            bianCharList.append(temp)
            name = str(num) + 'finChar.png'
            if DEBUG:
                cv2.imwrite(name, temp)
            num+=1
    return bianCharList

def resolveBinListToLatex(myBinList):
    return

myCap = init()
rawImg = getImg(myCap)
#getROI returns a dict with the following (mask, x, y, w, h)
myROIValList = getROI(rawImg)
myROI = myROIValList[0]
boundedImg = reduceToROI(rawImg, myROI)
myCharList = reduceImgToCharList(boundedImg, myROIValList)
resolveCharListToBinary(myCharList)
#resolveBinListToLatex(myCharList)
