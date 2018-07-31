""" Train """

import os
import numpy as np

import cv2
from keras.applications import inception_v3, resnet50, vgg19
from keras.models import Model

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "holiday-photos/image/jpg")

VECTOR_FILE = os.path.join(DATA_DIR, "vectors.tsv")

IMAGE_SIZE = 224


def get_vgg19():
    """ return vgg19 model and preprocessor """

    vgg19_model = vgg19.VGG19(weights="imagenet", include_top=True)

    model = Model(inputs=vgg19_model.input,
                  outputs=vgg19_model.get_layer("fc2").output)

    preprocessor = vgg19.preprocess_input

    return model, preprocessor


def get_resnet50():
    """ return resnet50 model and preprocessor """

    resnet_model = resnet50.ResNet50(weights="imagenet", include_top=True)

    model = Model(inputs=resnet_model.input,
                  outputs=resnet_model.get_layer("flatten_1").output)

    preprocessor = resnet50.preprocess_input

    return model, preprocessor


def get_inception3():
    """ return inception3 model and preprocessor """

    inception_model = inception_v3.InceptionV3(
        weights="imagenet", include_top=True)

    model = Model(inputs=inception_model.input,
                  outputs=inception_model.get_layer("flatten").output)

    preprocessor = inception_v3.preprocess_input

    return model, preprocessor


def image_batch_generator(image_names, batch_size):
    """ generator for vector """

    num_batches = len(image_names) // batch_size
    for i in range(num_batches):
        batch = image_names[i * batch_size: (i + 1) * batch_size]
        yield batch

    if len(image_names) % batch_size != 0:
        batch = image_names[num_batches * batch_size:]
        yield batch


def vectorize_features_images(image_dir, image_size, preprocessor, model, vector_file, batch_size=32):
    """ generate files for image features """

    image_names = os.listdir(image_dir)

    image_names.sort()

    num_vectors = 0

    with open(vector_file, "w") as file:
        for image_batch in image_batch_generator(image_names, batch_size):
            batch_images = []
            for image_name in image_batch:
                # print(image_name)
                image = cv2.imread(os.path.join(IMAGE_DIR, image_name))
                image = cv2.resize(image, (image_size, image_size))
                batch_images.append(image)

            x_data = preprocessor(np.array(batch_images, dtype="float32"))

            # print(x_data.shape)
            vectors = model.predict(x_data)
            # print(vectors.shape)

            for i in range(vectors.shape[0]):
                if num_vectors % 100 == 0:
                    print("{:d} vectors generated".format(num_vectors))

                image_vector = ",".join(["{:.5e}".format(v)
                                         for v in vectors[i].tolist()])
                # print(image_vector)
                file.write("{:s}\t{:s}\n".format(image_batch[i], image_vector))
                num_vectors += 1

        print("{:d} vectors generated".format(num_vectors))


def main():
    """ main """
    print("Start training")

    #model, preprocessor = get_vgg19()
    model, preprocessor = get_resnet50()

    if not os.path.isfile(os.path.join(DATA_DIR, "vectors.tsv")):
        vectorize_features_images(
            IMAGE_DIR, IMAGE_SIZE, preprocessor, model, VECTOR_FILE)

            


if __name__ == "__main__":
    main()
