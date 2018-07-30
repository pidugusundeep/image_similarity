
""" VGG19 features """
import sys

import cv2
import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Flatten, Input
from keras.models import Model

np.set_printoptions(threshold=np.nan)


def main():
    """ main function """

    if len(sys.argv) != 2:
        print("No input image.")
        exit(1)

    image_name = sys.argv[1]
    print("Image:{}".format(image_name))

    img_original = cv2.imread(image_name)
    img_resized = cv2.resize(img_original, (224, 224))
    img = np.resize(
        img_resized, (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))

    base_model = VGG19(weights='imagenet', include_top=False)
    input_layer = Input(shape=(224, 224, 3), name='image_input')
    net = base_model(input_layer)
    net = Flatten()(net)
    model = Model(inputs=input_layer, outputs=net)

    img = preprocess_input(img)

    features = model.predict(img)

    features=  features.squeeze()
    
    print(features)

    cv2.imshow('image', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
