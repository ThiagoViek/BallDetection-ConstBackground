from __future__ import print_function
import os
import pdb
import glob
import argparse

import cv2 as cv
import numpy as np

class BallSegmentation():
    def __init__(self, imgPath, savePath):
        self.savePath = savePath
        self.imgPath = imgPath

        self.imgList = glob.glob(self.imgPath + '/*.jpg')
        self.ids = [file.split('\\')[-1][:-4] for file in self.imgList]
        
        assert self.imgList != [], "Empty Folder! No images were found!"

        self.imgAnnotations()

    def imgAnnotations(self):
        """Save the annotated mask in the given path.
        """
        for i in range(len(self.imgList) - 1):
            img = self.drawMask(i)
            cv.imwrite(self.savePath + self.ids[i+1] + f"_segm.png", img)

    def drawMask(self, idx):
        """Returns the mask with ball segmentation.
        idx : image index
        """
        img1 = cv.imread(self.imgList[idx], 0)
        img2 = cv.imread(self.imgList[idx+1], 0)
        imgBlack = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)

        imgSubtract = cv.subtract(img2, img1)
        imgSubtract = cv.bilateralFilter(imgSubtract, 5, 5, 5)
        imgCanny = cv.Canny(imgSubtract, 50, 100)
        cnt, hierarchy = cv.findContours(imgCanny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        cntX, cntY = list(), list()
        cnt = np.concatenate(cnt).ravel().tolist()
        for i in range(len(cnt)):
            if i % 2 == 0:
                cntX.append(cnt[i])
            else:
                cntY.append(cnt[i])  
        xMax, xMin = np.max(cntX), np.min(cntX)
        yMax, yMin = np.max(cntY), np.min(cntY)

        radiusX = int((xMax - xMin) / 2)
        radiusY = int((yMax - yMin) / 2)
        radius = np.max((radiusX, radiusY))

        centreCoordX = xMin + radius
        centreCoordY = yMin + radius
        centreCoord = (centreCoordX, centreCoordY)

        color = 255
        imgSegment = cv.circle(imgBlack, centreCoord, radius, color, -1)

        img = cv.addWeighted(img2, 0.6, imgSegment, 0.4, 0)

        cv.imshow("image", imgSubtract)
        keyboard = cv.waitKey(0)

        return imgSegment

class LiveExperiment():
    def __init__(self, algo, savePath):
        self.algo = algo
        self.savePath = savePath

    def drawMaskLive(self):
        capture = cv.VideoCapture(0)
        ret, frame_0 = capture.read()
        i = 0
        
        while True:
            ret, frame_1 = capture.read()
            if frame_1 is None:
                break

            imgMask = self.drawMask(frame_0, frame_1)
            cv.imshow('Mask', imgMask)

            # cv.imwrite(self.savePath + f"{i}.jpg", imgMask)

            frame_0 = frame_1 
            i += 1           

            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    def drawMask(self, img1, img2):
        """Returns the mask with ball segmentation.
        img1 : image at time t-1
        img2: image at time t (real-time)
        """ 
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        imgBlack = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)

        imgSubtract = cv.subtract(img2, img1)
        imgSubtract = cv.bilateralFilter(imgSubtract, 5, 5, 5)
        imgSubtract = cv.fastNlMeansDenoising(imgSubtract, None, 10, 5, 5)
        imgCanny = cv.Canny(imgSubtract, 50, 100)
        cnt, hierarchy = cv.findContours(imgCanny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        if cnt == []:
            return img2
        else:
            cntX, cntY = list(), list()
            cnt = np.concatenate(cnt).ravel().tolist()
            for i in range(len(cnt)):
                if i % 2 == 0:
                    cntX.append(cnt[i])
                else:
                    cntY.append(cnt[i])  
            xMax, xMin = np.max(cntX), np.min(cntX)
            yMax, yMin = np.max(cntY), np.min(cntY)

            radiusX = int((xMax - xMin) / 2)
            radiusY = int((yMax - yMin) / 2)
            radius = np.max((radiusX, radiusY))
            
            centreCoordX = xMin + radius
            centreCoordY = yMin + radius
            centreCoord = (centreCoordX, centreCoordY)

            color = 255
            imgSegment = cv.circle(imgBlack, centreCoord, radius, color, -1)

            img = cv.addWeighted(img2, 0.6, imgSegment, 0.4, 0)

            return img

    def videoBackgroundSubt(self):
        """Runs subtraction between sequential frames coming from webcam.
        """
        if self.algo == 'MOG2':
            backSub = cv.createBackgroundSubtractorMOG2(history=2, varThreshold=50, detectShadows=False)
        else:
            backSub = cv.createBackgroundSubtractorKNN(history=2, dist2Threshold=50, detectShadows=False)

        capture = cv.VideoCapture(0)

        if not capture.isOpened:
            print('Unable to open: ' + args.input)
            exit(0)

        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            
            fgMask = backSub.apply(frame)    
            
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)
            
            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    parser.add_argument('--imgPath', type=str, help='Path to folder of images', default='C:/Users/thiag/Desktop/InLevel/Football/AutoLabelling/img/')
    parser.add_argument('--savePath', type=str, help='Path to folder where images are saved', default='C:/Users/thiag/Desktop/InLevel/Football/AutoLabelling/masks/')
    args = parser.parse_args()

    s = BallSegmentation(args.imgPath, args.savePath)
    s.drawCircle(1)

    # s = LiveExperiment(args.algo, args.savePath)
    # s = s.drawMaskLive()