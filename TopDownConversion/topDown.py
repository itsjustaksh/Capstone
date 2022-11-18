import cv2 as cv
import numpy as np

def locate_lane_points(image: cv.Mat):
    points = np.zeros((4,2), dtype=np.float32)

    
    # Find 4 points on image to use as landmarks

    return points

def transform_perspective(imgPath: str):

    # Var setup 
    im = cv.imread(imgPath)
    srcpoints = locate_lane_points(im)

    # Apply Transformation
    cv.getPerspectiveTransform(src=srcpoints, )


if __name__ == "__main__":

    print("Start")