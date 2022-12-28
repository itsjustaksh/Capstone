import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tensorflow import keras
import os

model = keras.models.load_model(os.getcwd() + '\\my_model.h5')


def test(image):
    return image


class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):

    small_img = cv2.resize(image, (180, 180))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]
    
    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)

    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = cv2.resize(lane_drawn, (1280, 720)).astype(np.uint8)

    print(lane_image.dtype)
    print(image.dtype)

    result = cv2.addWeighted(image, 0.9, lane_image, 0.1, 0)
    return result


# dir = os.getcwd() + r'\test_data\lanes_clip.mp4'
# vid_input = VideoFileClip(dir)
# vid_out = 'lanes_out_2.mp4'

# vid_clip = vid_input.fl_image(road_lines)
# vid_clip.write_videofile(vid_out)


lanes = Lanes()
clip1 = VideoFileClip("test_data\Pexels Videos 2048452.mp4")
final_clip = clip1.fl_image(road_lines)
final_clip.write_videofile('output.mp4', audio=False)
