import cv2
import numpy as np
import os
from pycocotools.coco import COCO
import numpy as np


def set_capture(is_test, filename=None):
    frame = None

    if (is_test):
        filename = '/test_data/' + filename
        frame = cv2.imread(os.getcwd() + filename)
        frame = cv2.resize(frame, (960, 540))

    else:
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)

    # Return frame being read from file or camera
    return frame
def labels_to_mask():
    # Path to COCO annotations file and images directory
    annFile = 'annotations/instances_val2017.json'
    imgDir = 'val2017'

    # Initialize COCO API
    coco = COCO(annFile)

    # Get category IDs for the classes you want to extract masks for
    catIds = coco.getCatIds(catNms=['person', 'car'])

    # Load annotations and iterate over images
    anns = coco.loadAnns(coco.getAnnIds(catIds=catIds))
    for ann in anns:
        # Load image
        img = cv2.imread(os.path.join(imgDir, coco.loadImgs(ann['image_id'])[0]['file_name']))
        
        # Create binary mask from segmentation polygons
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for seg in ann['segmentation']:
            seg = [int(x) for x in seg]
            poly = np.array(seg).reshape((-1, 2))
            cv2.fillPoly(mask, [poly], 1)
        
        # Save mask as image
        mask_path = os.path.join('masks', coco.loadImgs(ann['image_id'])[0]['file_name'][:-4] + '_' + coco.loadCats(ann['category_id'])[0]['name'] + '.png')
        cv2.imwrite(mask_path, mask * 255)

def white_and_yellow_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hsv, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_threshold, upper_threshold)

    # Combining white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    # Canny edge detection
    masked_image_canny = cv2.Canny(masked_image, 100, 200)

    return masked_image, masked_image_canny


def display_images(images):
    i = 1

    # Iterate through list of images and display them
    # with arbitruary names for now
    for image in images:
        cv2.imshow((str)(i), image)
        i += 1



def main(is_test: bool = False, filename=None):
    frame = set_capture(is_test, filename)

    while True:
        mask, mask_canny = white_and_yellow_mask(frame)
        display_images([frame, mask, mask_canny])

        # When q is pressed, program stops running
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    is_test = True
    filename = "lane_test.png"
    main(is_test, filename)
