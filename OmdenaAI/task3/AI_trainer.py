import cv2
import numpy as np
import time
import utils as pm



cap = cv2.VideoCapture("test.mp4")
detector = pm.poseDetector()

while True:
    success,img = cap.read()
    #img = cv2.resize(1288,728)
    img = detector.findPose(img)
    lmList = detector.findPosition(img,False)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)