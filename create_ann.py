import csv
import os
import sys

import cv2
import numpy as np
from annoy import AnnoyIndex
from keras.applications import inception_v3
from keras.models import Model

BATCH_SIZE = 32
BASE_DIR = "/home/andrei/temp/"
IMAGE_DIR_1 = "validation"
IMAGE_DIR_2 = "train_0"

IMAGE_SIZE = 299

FEATURES_COUNT = 2048

# generate model and preprocessor for inception


def get_inception3():
    """ return inception3 model and preprocessor """

    inception_model = inception_v3.InceptionV3(
        weights="imagenet", include_top=True)

    model = Model(inputs=inception_model.input,
                  outputs=inception_model.get_layer("avg_pool").output)

    preprocessor = inception_v3.preprocess_input

    return model, preprocessor


# image generator
def image_batch_generator(image_names, batch_size):
    """ generator for vector """

    num_batches = len(image_names) // batch_size
    for i in range(num_batches):
        batch = image_names[i * batch_size: (i + 1) * batch_size]
        yield batch

    if len(image_names) % batch_size != 0:
        batch = image_names[num_batches * batch_size:]
        yield batch


# main function
def main():
    print("Main")

    if len(sys.argv) != 4:
        print("No arguments!!!")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]

    index = AnnoyIndex(FEATURES_COUNT)

    image_names = os.listdir(input_dir)
    image_names.sort()

    model, preprocessor = get_inception3()

    images_add = []

    num_vectors = 0
    for image_batch in image_batch_generator(image_names, BATCH_SIZE):
        batch_images = []
        for image_name in image_batch:
            # print(image_name)
            image = cv2.imread(os.path.join(input_dir, image_name))
            if image is None:
                continue

            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            images_add.append(image_name)
            batch_images.append(image)

        x_data = preprocessor(np.array(batch_images, dtype="float32"))

        # print(x_data.shape)
        vectors = model.predict(x_data)
        # print(vectors.shape)

        # print(vectors.shape[0])
        for i in range(vectors.shape[0]):
            if num_vectors % 100 == 0:
                print("{:d} vectors generated".format(num_vectors))

            index.add_item(num_vectors, vectors[i])
            num_vectors += 1

    print("{:d} vectors generated".format(num_vectors))

    index.build(10)  # 10 trees
    index.save(os.path.join(output_dir, model_name+".ann"))

    with open(os.path.join(output_dir, model_name+".csv"), mode='w') as filep:
        writer = csv.writer(filep, delimiter=',')
        for image_name in images_add:
            writer.writerow([os.path.join(input_dir, image_name)])


if __name__ == "__main__":
    main()
