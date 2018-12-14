import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import geometric_transform
from scipy import ndimage

def main():
  return


main()

# 1. Source and Replacement Face and Facial Landmark's Detection
# 2. Feature Extraction - features used to control the warp should be along the convex hull of the face
# 3.Face Swapping - for each frame, compute image transforms to warp replacemnt to source face and vice-versa
# 4. Video Refinement - incorporate Gradient Domain Blending and  Optical Flow
# 5. video to video replacement - account for the fact that the 2 videos may have a different number of frames
