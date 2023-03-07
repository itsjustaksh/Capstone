import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self):
        self.image_path = None
        self.original_image = None
        self.output_image = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.detect_lanes_with_labels()
            self.output_image = np.array(Image.open('test_2.jpg'))
            self.update_display()
            self.show_direction()

    def detect_lanes_with_labels(self):
        coco_file_path = os.path.dirname(self.image_path) + '/_annotations.coco.json'
        coco = COCO(coco_file_path)
        image_name = os.path.basename(self.image_path)
        image_ids = coco.getImgIds()
        for id in image_ids:
            image_coco = coco.loadImgs(ids=[id])
            if(image_name == image_coco[0]['file_name']):
                image_id = id
                break
        annids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(annids)
        image = np.array(Image.open(self.image_path))
        plt.imshow(image)
        plt.grid(False)
        plt.axis('off')
        anns = coco.loadAnns(annids)
        coco.showAnns(anns)
        fig = plt.gcf()
        fig.savefig('test_2.jpg', bbox_inches='tight',transparent=True, pad_inches=0) 
        fig.clf()
        #labeled_image = coco.annToMask(anns[0])
        #for ann in range(len(anns)):
            #labeled_image += coco.annToMask(anns[ann])
        #return labeled_image
        
    def detect_lanes(self):
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        # Define region of interest mask
        mask = np.zeros_like(edges)
        height, width = edges.shape
        polygon = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        # Apply Hough lines detection
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
        # Draw lane lines
        line_image = np.zeros_like(self.original_image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        # Merge lane lines with original image
        output = cv2.addWeighted(self.original_image, 0.8, line_image, 1, 0)
        # Calculate angle of lane lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        # Determine direction based on average angle
        if len(angles) > 0:
            average_angle = sum(angles) / len(angles)
            if average_angle < -10:
                self.direction = "Left"
            elif average_angle > 10:
                self.direction = "Right"
            else:
                self.direction = "Straight"
        else:
            self.direction = "No lane lines detected"
        return output

    def update_display(self):
        # Convert output image to PIL format for display in tkinter
        output_image = cv2.cvtColor(self.output_image, cv2.COLOR_BGR2RGB)
        output_image = self.output_image
        output_image = Image.fromarray(output_image)
        output_image = ImageTk.PhotoImage(output_image)
        canvas.itemconfig(image_on_canvas, image=output_image)
        canvas.image = output_image

    def show_direction(self):
        direction_label.config(text=f"Lane direction: {self.direction}")

# Create lane detector instance
detector = LaneDetector()

# Create UI
root = tk.Tk()
root.title("Lane Detector")

# Set window background color
root.configure(bg="#f2f2f2")

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(side="left", padx=20, pady=20)

# Create buttons
load_button = tk.Button(root, text="Load Image", command=detector.load_image, bg="#008CBA", fg="white", padx=10, pady=5, bd=0, font=("Arial", 14))
load_button.pack(pady=20)

# Create label to display direction
direction_label = tk.Label(root, text="Lane direction: ", bg="#f2f2f2", font=("Arial", 18))
direction_label.pack(side="right", padx=20, pady=20)

# Add image placeholder to canvas
output_image = Image.new("RGB", (800, 600), color="#f2f2f2")
output_image = ImageTk.PhotoImage(output_image)
image_on_canvas = canvas.create_image(0, 0, anchor="nw", image=output_image)

root.mainloop()