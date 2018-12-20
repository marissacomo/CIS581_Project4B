import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib as dl
from scipy.ndimage import geometric_transform
from scipy import ndimage
from faceSwapping import faceSwapping

def main(video1Path, video2Path, prefix):
    # load videos
    rawVideo1 = cv2.VideoCapture(video1Path)
    rawVideo2 = cv2.VideoCapture(video2Path)

    if(not rawVideo1.isOpened() or not rawVideo2.isOpened()):
        print('Failed!!!')
        return

    faceSwapping(rawVideo1, rawVideo2, prefix)


main('../CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4', '../CIS581Project4PartCDatasets/Easy/MrRobot.mp4', "Frank_Robot")
# main('../CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4', '../CIS581Project4PartCDatasets/Easy/MrRobot.mp4', "Frank_Robot_Points")
# main('../CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4', '../CIS581Project4PartCDatasets/Easy/MrRobot.mp4', "Frank_Robot_Tri")
# main('../CIS581Project4PartCDatasets/Easy/Marissa.mp4', '../CIS581Project4PartCDatasets/Easy/Zach.mp4', "Marissa_Zach")
