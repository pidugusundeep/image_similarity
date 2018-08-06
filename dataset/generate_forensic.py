""" generate data for forensic similarity """
import os

import cv2
import numpy as np


def flip_filter(image):
    """ flip filter """
    return cv2.flip(image, 1)


def resize_filter(image):
    """ resize filter """
    factor = np.random.uniform(0.95, 1.05)
    # print(factor)
    shape = (int(image.shape[1]*factor), int(image.shape[0]*factor))
    return cv2.resize(image, shape)


def text_filter(image):
    """ put text in image """
    dst = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dst, 'OpenCV', (10, 500), font, 4,
                (255, 255, 255), 2, cv2.LINE_AA)

    return dst


def histogram_filter(image):
    """ hist image """
    return cv2.equalizeHist(image)


def translate_filter(image):
    """ translate image """

    # print(image.shape)
    rows, cols, _ = image.shape

    x_translate = np.random.uniform(-20, 20)
    y_translate = np.random.uniform(-20, 20)

    transformation = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
    return cv2.warpAffine(image, transformation, (cols, rows))


def filter_image(image):
    """ filter image """
    filters = [flip_filter, resize_filter, text_filter, translate_filter]

    # number_filters = np.random.randint(len(filters))+1
    number_filters = 2
    filter_index = np.arange(len(filters))
    np.random.shuffle(filter_index)
    filter_index = filter_index[:number_filters]

    similar = image.copy()
    for idx in filter_index:
        similar = filters[idx](similar)

    return similar


def main():
    """ main function """
    print("Start")

    image_dir = "/home/andrei/temp/validation"
    image_similar_dir = "/home/andrei/temp/validation_similar"

    images_name = os.listdir(image_dir)
    #print(images_name)
    images_name.sort()

    for idx, name in enumerate(images_name):
        print(idx)

        image = cv2.imread(os.path.join(image_dir, name))
        similar_image = filter_image(image)
        cv2.imwrite(os.path.join(image_similar_dir, name), similar_image)

    # cv2.imshow('image', image)
    # cv2.imshow('similar', similar_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
