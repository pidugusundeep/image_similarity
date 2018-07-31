import os

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "holiday-photos/image/jpg")


def draw_image(subplot, image, title):
    plt.subplot(subplot)
    plt.imshow(image)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def main():
    print("Start")

    ref_image = plt.imread(os.path.join(IMAGE_DIR, "100301.jpg"))
    sim_image = plt.imread(os.path.join(IMAGE_DIR, "100302.jpg"))
    dif_image = plt.imread(os.path.join(IMAGE_DIR, "127202.jpg"))

    draw_image(131, ref_image, "reference")
    draw_image(132, sim_image, "similar")
    draw_image(133, dif_image, "different")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
