import os

import cv2
import numpy as np
from annoy import AnnoyIndex
from keras.applications import inception_v3
from keras.models import Model

BATCH_SIZE = 32
VIDEO_DIR = "/home/andrei/temp/video"
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

    index = AnnoyIndex(FEATURES_COUNT)

    videos = os.listdir(VIDEO_DIR)

    videos.sort()

    model, preprocessor = get_inception3()

    num_vectors = 0
    for video in videos:
        print("Process "+video)
        images = []
        vector = np.zeros(FEATURES_COUNT, dtype="float32")
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(frame)
                #print(len(images))
                if(len(images) == BATCH_SIZE):
                    x_data = preprocessor(
                        np.array(images, dtype="float32"))
                    vectors = model.predict(x_data)
                    for i in range(vectors.shape[0]):
                        vector = vector+vectors[i]
                    del images[:]

            else:
                break

        cap.release()

        index.add_item(num_vectors, vector)
        num_vectors += 1
        if num_vectors % 100 == 0:
            print("{:d} vectors generated".format(num_vectors))

    print("{:d} vectors generated".format(num_vectors))
    index.build(10)  # 10 trees
    index.save('data/model_video.ann')


if __name__ == "__main__":
    main()
