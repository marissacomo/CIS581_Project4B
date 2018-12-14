import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib as dl
from scipy.ndimage import geometric_transform
from scipy import ndimage

MAX_FEATURE_POINTS = 20
SAVE_VIDEO = False

def faceSwapping(rawVideo1, rawVideo2):

    # get video1 and video2 resolutions and FPS
    fps1 = rawVideo1.get(cv2.CAP_PROP_FPS)
    resW1 = int(rawVideo1.get(cv2.CAP_PROP_FRAME_WIDTH))
    resH1 = int(rawVideo1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps2 = rawVideo2.get(cv2.CAP_PROP_FPS)
    resW2 = int(rawVideo2.get(cv2.CAP_PROP_FRAME_WIDTH))
    resH2 = int(rawVideo2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if SAVE_VIDEO:
        out = cv2.VideoWriter('output.avi', fourcc, fps, (resW, resH))

    # get first frames
    ret1, prevFrame1 = rawVideo1.read()   
    ret2, prevFrame2 = rawVideo2.read()   

    prevGray1 = cv2.cvtColor(prevFrame1, cv2.COLOR_BGR2GRAY)
    prevGray2 = cv2.cvtColor(prevFrame2, cv2.COLOR_BGR2GRAY)

    # x_features, y_features = getFeatures(prevGray, bbox, maxFeaturePoints)

    prevGray1 = np.array(prevGray1)
    prevGray2 = np.array(prevGray2)

    frameCount = 1

    # get face and facial landmarks

    # feature extraction

    # swap faces w/ gradient domain blending

    # main loop for optical flow
    while(rawVideo1.isOpened() and rawVideo2.isOpened()):
        # load next frames
        ret1, nextFrame1 = rawVideo1.read()
        nextGray1 = cv2.cvtColor(nextFrame1, cv2.COLOR_BGR2GRAY)
        nextGray1 = np.array(nextGray1)

        ret2, nextFrame2 = rawVideo2.read()
        nextGray2 = cv2.cvtColor(nextFrame2, cv2.COLOR_BGR2GRAY)
        nextGray2 = np.array(nextGray2)

        # swap faces w/ gradient domain blending

    return