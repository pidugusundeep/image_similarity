""" generate data for forensic similarity """
import cv2
import numpy as np


def flip_filter(image):
    """ flip filter """
    return cv2.flip(image, 1)


def resize_filter(image):
    """ resize filter """
    factor = np.random.uniform(0.5, 1.5)
    print(factor)
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

    print(image.shape)
    rows, cols, _ = image.shape

    x_translate = np.random.uniform(-50,50)
    y_translate = np.random.uniform(-50,50)

    transformation = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
    return cv2.warpAffine(image, transformation, (cols, rows))


def main():
    """ main function """
    print("Start")

    filters = [flip_filter, resize_filter, text_filter, translate_filter]

    image = cv2.imread("/home/andrei/temp/validation/ffb9838816c0021a.jpg")

    number_filters = np.random.randint(len(filters))+1
    # number_filters=2
    filter_index = np.arange(len(filters))
    np.random.shuffle(filter_index)
    filter_index = filter_index[:number_filters]

    similar = image.copy()
    for idx in filter_index:
        similar = filters[idx](similar)

    cv2.imshow('image', image)
    cv2.imshow('similar', similar)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(filter_index)


if __name__ == "__main__":
    main()
