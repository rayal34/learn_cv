import matplotlib.pyplot as plt
import numpy as np


def show_image(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
