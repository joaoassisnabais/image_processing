import cv2 as cv


def SIFT(img): # SIFT feature extraction
    
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    
    return kp, des
