
import sys

import cv2
import numpy as np
from keras.applications.resnet50 import (ResNet50, decode_predictions,
                                         preprocess_input)
from keras.layers import Flatten, Input
from keras.models import Model

np.set_printoptions(threshold=np.nan)


def main():

    if len(sys.argv) != 2:
        print("No input image.")
        exit(1)

    image_name = sys.argv[1]
    print("Image:{}".format(image_name))

    img = cv2.imread(image_name)
    img = cv2.resize(img, (224, 224))
    img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))

    print(img.shape)

    base_model = ResNet50(weights='imagenet', pooling=max, include_top=False)
    input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(input)
    x = Flatten()(x)
    model = Model(inputs=input, outputs=x)

    img = preprocess_input(img)

    features = model.predict(img)

    features=  features.squeeze()

    print(features)

    print(features.shape)


if __name__ == "__main__":
    main()
