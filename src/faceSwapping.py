import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib as dl
from scipy.ndimage import geometric_transform
from scipy import ndimage
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib

MAX_FEATURE_POINTS = 20
SHOW_FRAME = True
SAVE_VIDEO = False

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # detect faces in the grayscale frame

    rects = detector(prevGray1, 0)
 
    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(prevFrame1, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)

    # p0 = cv2.goodFeaturesToTrack(prevGray1, mask = None, **feature_params)

    feature_pts = []
    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(prevFrame1, (bX, bY), (bX + bW, bY + bH),
            (0, 255, 0), 1)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(prevGray1, rect)
        shape = face_utils.shape_to_np(shape)
 
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(prevFrame1, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(prevFrame1, str(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            feature_pts.append(np.array([np.array([x,y])]))
    
    feature_pts = np.array(feature_pts).astype(np.float32)

    # while True:
    #     # show the frame
    #     cv2.imshow("prevFrame1", prevFrame1)

    #     if cv2.waitKey(33) & 0xFF == ord('q'):
    #         break
    # feature extraction

    # swap faces w/ gradient domain blending

    # main loop for optical flow
    while(ret1 and ret2):
        # load next frames
        ret1, nextFrame1 = rawVideo1.read()
        if ret1:
            nextGray1 = cv2.cvtColor(nextFrame1, cv2.COLOR_BGR2GRAY)
            nextGray1 = np.array(nextGray1)

        # calculate optical flow for video1
        feature_pts, st, err = cv2.calcOpticalFlowPyrLK(prevGray1, nextGray1, feature_pts,  None, **lk_params)

        hull = cv2.convexHull(feature_pts)
        cv2.drawContours(nextFrame1, [hull.astype(int)], 0, (255,0,0), 1, 8)

        for i in range(len(feature_pts)):
            point = feature_pts[i][0] # point x,y
            x = int(point[0])
            y = int(point[1])
            cv2.circle(nextFrame1, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(nextFrame1, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        prevFrame1 = nextFrame1.copy()
        prevGray1 = nextGray1.copy()

        #show the frame
        cv2.imshow("nextFrame1", nextFrame1)


        # ret2, nextFrame2 = rawVideo2.read()
        # if ret2:
        #     nextGray2 = cv2.cvtColor(nextFrame2, cv2.COLOR_BGR2GRAY)
        #     nextGray2 = np.array(nextGray2)

        # # swap faces w/ gradient domain blending

        # if SHOW_FRAME:
        #     plt.show()

        # frameCount += 1

        # if SAVE_VIDEO:
            # out.write(nextFrame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break


    if SAVE_VIDEO:
        out.release()

    rawVideo1.release()
    rawVideo2.release()
    cv2.destroyAllWindows()

    return