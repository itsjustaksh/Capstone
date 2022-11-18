import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
from collections import deque

# Needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# Importing calibration images
filename = os.getcwd() + '/camera_cal/*.jpg'
cal_filenames = glob.glob(filename)
cal_images = np.array([np.array(plt.imread(img)) for img in cal_filenames])

# Importing test images
filename = os.getcwd() + '/test_images/*.jpg'
test_filenames = glob.glob(filename)
test_images = np.array([np.array(plt.imread(img)) for img in test_filenames])

# Chessboard edges
nx = 9
ny = 6

img = test_images[0]
img_y, img_x = (img.shape[0], img.shape[1])
offset = 50

# Lane masking and coordinates for perspective transform
source = np.float32([ # MASK
    [img_y-offset, offset], # bottom left
    [img_y-offset, img_x-offset], # bottom right
    [offset, offset], # top left
    [offset, img_x-offset]]) # top right

dest = np.float32([ # DESTINATION
    [300, 720], # bottom left
    [950, 720], # bottom right
    [300, 0], # top left
    [950, 0]]) # top right

class Camera():    
    def __init__(self):
        # Stores the source 
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        
        self.objpoints = [] # 3D points in real space
        self.imgpoints = [] # 2D points in img space
    
        
    def calibrate_camera(self, imgList):
        counter = 0
        for img in imgList:
            # Prepare object points (0,0,0), (1,0,0), etc.
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

            # Converting to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Finding chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(objp)
                counter+=1
        
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        return self.mtx, self.dist

def undistort(self, img):
        return cv2.undistort(img,self.mtx,self.dist,None,self.mtx)

def show_images():
    for i in range(1, 6):
        img = cv2.imread('test_images/test_online_'+ str(i) +'.jpg')
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(cv2.undistort(img,mtx,dist,None,mtx), cv2.COLOR_BGR2RGB))
        plt.title("Undistorted Image")

        plt.show()



def main():
    img = cv2.imread('camera_cal/calibration1.jpg')
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.undistort(img,mtx,dist,None,mtx))
    plt.title("Undistorted Image")
    
    plt.show()
    
    show_images()


if __name__ == '__main__':
    camera = Camera()
    mtx, dist = camera.calibrate_camera(cal_images)
    main()