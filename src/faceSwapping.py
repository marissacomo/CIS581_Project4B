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

SHOW_FRAME = False
SAVE_VIDEO = True
DEBUG = False
MIN_FEATURIZE_DISTANCE = 2


# params for ShiTomasi corner detection
# open-source code: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
# open-source code: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
# open-source code: https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

#calculate delanauy triangle
# open-source code: https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri

def featurize(prevGray1, prevGray2, detector, predictor):
    # detections for video 1 and 2
    rects1 = detector(prevGray1, 0)
    rects2 = detector(prevGray2, 0)

     # feature extraction
    feature_pts1 = []
    feature_pts2 = []

    # loop over the face detections for video 1
    for rect in rects1:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(prevFrame1, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)

        shape = predictor(prevGray1, rect)
        shape = face_utils.shape_to_np(shape)
 
        # get feature points
        for (i, (x, y)) in enumerate(shape):
            feature_pts1.append(np.array([np.array([x,y])]))

    # loop over the face detections for video 2
    for rect in rects2:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(prevFrame2, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)

        # open-source code: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        shape = predictor(prevGray2, rect)
        shape = face_utils.shape_to_np(shape)
 
        # get feature points
        for (i, (x, y)) in enumerate(shape):
            feature_pts2.append(np.array([np.array([x,y])]))
    
    # conver to float32, needed for calcOpticalFlowPyrLK 
    feature_pts1 = np.array(feature_pts1).astype(np.float32)
    feature_pts2 = np.array(feature_pts2).astype(np.float32)

    return feature_pts1, feature_pts2

# Warps and alpha blends triangular regions from img1 and img2 to img
# open-source code: https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 

def correctFeaturePoints(trackedFeaturePoints, newFeaturePoints, prevFramePoints):
    result = []

    for idx1 in range(0, len(trackedFeaturePoints)):
        ofPt = trackedFeaturePoints[idx1]
        newPt = newFeaturePoints[idx1]
        prevPt = prevFramePoints[idx1]
        trackedPoint = np.array([ofPt[0][0], ofPt[0][1]])
        newPoint = np.array([newPt[0][0], newPt[0][1]])
        prevPoint = np.array([prevPt[0][0], prevPt[0][1]])

        newPointDelta = np.linalg.norm(newPoint - prevPoint)
        opticalFlowDelta = np.linalg.norm(trackedPoint - prevPoint)
        comparisonDelta = np.linalg.norm(trackedPoint - newPoint)

        # distSq = (trackedPoint[0] - newPoint[0]) * (trackedPoint[0] - newPoint[0]) + (trackedPoint[1] - newPoint[1]) * (trackedPoint[1] - newPoint[1])

        if comparisonDelta > MIN_FEATURIZE_DISTANCE:
            result.append(newPt)
        else:
            result.append(ofPt)
        # result.append(0.9 * newPt + 0.1 * ofPt)

    result = np.array(result).astype(np.float32)

    return result

def faceSwapping(rawVideo1, rawVideo2):

    # get video1 and video2 resolutions and FPS
    fps1 = rawVideo1.get(cv2.CAP_PROP_FPS)
    resW1 = int(rawVideo1.get(cv2.CAP_PROP_FRAME_WIDTH))
    resH1 = int(rawVideo1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps2 = rawVideo2.get(cv2.CAP_PROP_FPS)
    resW2 = int(rawVideo2.get(cv2.CAP_PROP_FRAME_WIDTH))
    resH2 = int(rawVideo2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if SAVE_VIDEO:
        out = cv2.VideoWriter('output.avi', fourcc, fps2, (resW2, resH2))

    # get first frames
    ret1, prevFrame1 = rawVideo1.read()   
    ret2, prevFrame2 = rawVideo2.read()  

    prevGray1 = cv2.cvtColor(prevFrame1, cv2.COLOR_BGR2GRAY)
    prevGray2 = cv2.cvtColor(prevFrame2, cv2.COLOR_BGR2GRAY)

    prevGray1 = np.array(prevGray1)
    prevGray2 = np.array(prevGray2)

    frameCount = 1

    # get facial landmarks
    # open-source code: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    of_tracked_feature_pts1, of_tracked_feature_pts2 = featurize(prevGray1, prevGray2, detector, predictor)

    # main loop for optical flow
    while(ret1 and ret2):
        # load next frames
        ret1, nextFrame1 = rawVideo1.read()
        ret2, nextFrame2 = rawVideo2.read()
        if (not ret1) or (not ret2):
            break

        # convert next frame for video 1 to gray and make it an np array
        if ret1:
            nextWarped1 = np.copy(nextFrame2)
            nextGray1 = cv2.cvtColor(nextFrame1, cv2.COLOR_BGR2GRAY)
            nextGray1 = np.array(nextGray1)

        # convert next frame for video 2 to gray and make it an np array
        if ret2:
            nextWarped2 = np.copy(nextFrame1)
            nextGray2 = cv2.cvtColor(nextFrame2, cv2.COLOR_BGR2GRAY)
            nextGray2 = np.array(nextGray2)

        new_feature_pts1, new_feature_pts2 = featurize(nextGray1, nextGray2, detector, predictor)

        corrected_feature_pts1 = []
        corrected_feature_pts2 = []

        prev_feature_pts1 = np.copy(of_tracked_feature_pts1)
        prev_feature_pts2 = np.copy(of_tracked_feature_pts2)

        # calculate optical flow for video 1 and 2
        of_tracked_feature_pts1, st, err = cv2.calcOpticalFlowPyrLK(prevGray1, nextGray1, of_tracked_feature_pts1,  None, **lk_params)
        of_tracked_feature_pts2, st, err = cv2.calcOpticalFlowPyrLK(prevGray2, nextGray2, of_tracked_feature_pts2,  None, **lk_params)

        # Filter the points
        corrected_feature_pts1 = correctFeaturePoints(of_tracked_feature_pts1, new_feature_pts1, prev_feature_pts1)
        corrected_feature_pts2 = correctFeaturePoints(of_tracked_feature_pts2, new_feature_pts2, prev_feature_pts2)

        of_tracked_feature_pts1 = corrected_feature_pts1
        of_tracked_feature_pts2 = corrected_feature_pts2

        # get convex hull for video 1 and 2
        hull1 = cv2.convexHull(corrected_feature_pts1)
        hull2 = cv2.convexHull(corrected_feature_pts2)
        hull1Proper = []
        hull2Proper = []

        for p in hull1:
            hull1Proper.append((p[0][0], p[0][1]))
        for p in hull2:
            hull2Proper.append((p[0][0], p[0][1]))

        # open-source code: https://github.com/spmallick/learnopencv/blob/master/FaceSwap/faceSwap.py
        # Find delanauy traingulation for convex hull points
        pts1 = []
        pts2 = []
        matches = []

        count = 0
        for p in corrected_feature_pts1:
            pts1.append(np.array([p[0][0], p[0][1]]))
            matches.append(cv2.DMatch(count, count, 0))
            count = count + 1

        for p in corrected_feature_pts2:
            pts2.append(np.array([p[0][0], p[0][1]]))

        sizeFrame1 = nextFrame1.shape
        rect1 = (0, 0, sizeFrame1[1], sizeFrame1[0])

        sizeFrame2 = nextFrame2.shape    
        rect2 = (0, 0, sizeFrame2[1], sizeFrame2[0])
        dt = calculateDelaunayTriangles(rect2, pts2)
        
        if len(dt) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []
            
            #get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(pts1[dt[i][j]])
                t2.append(pts2[dt[i][j]])
            
            warpTriangle(nextFrame1, nextWarped1, t1, t2)

        # tps = cv2.createThinPlatSplineShapeTransformer()
        # tps.estimateTransform(np.array(pts1).astype(np.int32), np.array(pts2).astype(np.int32), matches)
        # nextWarped1 = tps.warpImage(nextFrame1, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Calculate Mask
        hull8U = []
        for i in range(0, len(hull2Proper)):
            hull8U.append((hull2Proper[i][0], hull2Proper[i][1]))
        
        mask = np.zeros(nextFrame2.shape, dtype = nextFrame2.dtype)  
        
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
        
        r = cv2.boundingRect(np.float32([hull2Proper]))    
        
        center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(nextWarped1), nextFrame2, mask, center, cv2.NORMAL_CLONE)

        # draw contours for video 1 and 2
        if DEBUG:
            cv2.drawContours(output, [hull2.astype(int)], 0, (255,0,0), 1, 8)

            for i in range(len(corrected_feature_pts2)):
                point = corrected_feature_pts2[i][0] # point x,y
                x = int(point[0])
                y = int(point[1])
                cv2.circle(output, (x, y), 1, (0, 0, 255), -1)
           
        # set prev to next
        prevFrame1 = nextFrame1.copy() # video 1
        prevGray1 = nextGray1.copy()
        prevFrame2 = nextFrame2.copy() # video 2
        prevGray2 = nextGray2.copy()

        #show the frame
        if SHOW_FRAME:
            # cv2.imshow("nextFrame1", nextFrame1)
            # cv2.imshow("nextFrame2", nextFrame2)
            cv2.imshow("Face Swapped", output)

        if SAVE_VIDEO:
            print("Frame: ", frameCount)
            out.write(output)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        frameCount = frameCount + 1

    if SAVE_VIDEO:
        out.release()

    rawVideo1.release()
    rawVideo2.release()
    cv2.destroyAllWindows()

    return