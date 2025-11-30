import cv2
import numpy as np

def sobel_edges(img, ksize=3, normalize=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx*gx + gy*gy)
    if normalize:
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return mag.astype(np.uint8)
    return mag

def laplacian_edges(img, ksize=3, normalize=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    abs_lap = np.abs(lap)
    if normalize:
        abs_lap = cv2.normalize(abs_lap, None, 0, 255, cv2.NORM_MINMAX)
        return abs_lap.astype(np.uint8)
    return abs_lap

def canny_edges(img, threshold1=100, threshold2=200, apertureSize=3, L2gradient=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)
