# enhancement.py
import cv2
import numpy as np

def smooth_image(img, d=5, sigmaColor=50, sigmaSpace=50):
    

    # bilateralFilter preserves edges while reducing small noise.
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


def enhance_full_image(img, d=5, sigmaColor=50, sigmaSpace=50):
    
    final = smooth_image(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return {"final": final}
