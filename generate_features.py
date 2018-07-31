""" generate features """

import hashlib
import sys

import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

np.set_printoptions(threshold=np.nan)


def main():
    """ main function """

    if len(sys.argv) != 2:
        print("No input image.")
        exit(1)

    image_name = sys.argv[1]
    print("Image:{}".format(image_name))

    img = cv2.imread(image_name)
    img = cv2.resize(img, (224, 224))
    img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))

    base_model = ResNet50(weights='imagenet', include_top=True)
    model = Model(input=base_model.input,
                  output=base_model.get_layer("flatten_1").output)

    img = preprocess_input(img)

    features = model.predict(img)

    features = features.squeeze()

    print(features)

    print(features.shape)

    hash_data = hashlib.sha512(features).hexdigest()

    print(hash_data)

    with open("data/image_hash.txt", "w") as file:
        file.write(hash_data)


if __name__ == "__main__":
    main()
