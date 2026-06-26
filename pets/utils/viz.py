import cv2
import matplotlib.pyplot as plt


def view_bbox(image, label, top_left, bottom_right):
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), thickness=2)
    cv2.imshow(label, image)
    plt.axis("off")
