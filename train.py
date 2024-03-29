""" Train """

import itertools
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import inception_v3, resnet50, vgg19
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "holiday-photos/image/jpg")

VECTOR_FILE = os.path.join(DATA_DIR, "vectors.tsv")

IMAGE_SIZE = 299

VECTOR_SIZE = 2048

BATCH_SIZE = 32

NUM_EPOCHS = 10


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
                  outputs=inception_model.get_layer("avg_pool").output)

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
                image = cv2.imread(os.path.join(image_dir, image_name))
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
                image_file_path = os.path.join(image_dir, image_batch[i])
                file.write("{:s}\t{:s}\n".format(
                    image_file_path, image_vector))
                num_vectors += 1

        print("{:d} vectors generated".format(num_vectors))


def get_triples(image_dir, image_similar_dir):
    """ get trippler """

    images_name_list = sorted(os.listdir(image_dir))

    num_sims = 0
    image_triples = []

    for i, name in enumerate(images_name_list):
        

        image_file_path = os.path.join(image_dir, name)
        image_similar_file_path = os.path.join(image_similar_dir, name)

        image_triples.append((image_file_path, image_similar_file_path, 1))
        while True:
            j = np.random.randint(low=0, high=len(images_name_list), size=1)[0]
            if j != i:
                break

        image_not_similar_file_path = os.path.join(
            image_dir, images_name_list[j])
        image_triples.append((image_file_path, image_not_similar_file_path, 0))

    print("Generated {:d} pos + {:d} neg = {:d} total image triples"
          .format(num_sims, num_sims, 2*num_sims))

    return image_triples


def load_vectors(vector_file):
    """ load features file """

    vec_dict = {}
    with open(vector_file, "r") as file:
        for line in file:
            image_name, image_vec = line.strip().split("\t")
            vec = np.array([float(v) for v in image_vec.split(",")])
            vec_dict[image_name] = vec

    return vec_dict


def cosine_distance(vecs, normalize=False):
    """ cosine distance """
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)


def cosine_distance_output_shape(shapes):
    """ shape """
    return shapes[0]


def get_siamese_model():
    """ get siamese  model """

    input_1 = Input(shape=(VECTOR_SIZE,))
    input_2 = Input(shape=(VECTOR_SIZE,))
    #merged = Concatenate(axis=-1)([input_1, input_2])
    merged = Lambda(cosine_distance,
                    output_shape=cosine_distance_output_shape)([input_1, input_2])

    fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Activation("relu")(fc1)

    fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
    fc2 = Dropout(0.2)(fc2)
    fc2 = Activation("relu")(fc2)

    pred = Dense(2, )(fc2)
    pred = Activation("softmax")(pred)

    model = Model(inputs=[input_1, input_2], outputs=pred)

    return model


# batch to vectorimage_batch[i]
def batch_to_vectors(batch, vec_size, vec_dict):

    X1 = np.zeros((len(batch), vec_size))
    X2 = np.zeros((len(batch), vec_size))
    Y = np.zeros((len(batch), 2))
    for tid in range(len(batch)):
        X1[tid] = vec_dict[batch[tid][0]]
        X2[tid] = vec_dict[batch[tid][1]]
        Y[tid] = [1, 0] if batch[tid][2] == 0 else [0, 1]
    return ([X1, X2], Y)


def data_generator(triples, vec_size, vec_dict, batch_size=32):
    """ generator """

    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triples)))

        num_batches = len(triples) // batch_size

        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size: (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            yield batch_to_vectors(batch, vec_size, vec_dict)


def plot_history(history):
    """ plot history """
    plt.subplot(211)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="r", label="train")
    plt.plot(history.history["val_acc"], color="b", label="validation")
    plt.legend(loc="best")

    plt.show()


def main():
    """ main """
    print("Start training")

    #model, preprocessor = get_vgg19()
    model, preprocessor = get_inception3()

    image_dir = "/home/andrei/temp/validation"
    image_similar_dir = "/home/andrei/temp/validation_similar"

    image_vector_file = os.path.join("/home/andrei/temp/", "vectors.tsv")
    print(image_vector_file)
    if not os.path.isfile(image_vector_file):

        vectorize_features_images(
            image_dir, IMAGE_SIZE, preprocessor, model, image_vector_file)

    image_similar_vector_file = os.path.join(
        "/home/andrei/temp/", "vectors_similar.tsv")
    print(image_similar_vector_file)
    if not os.path.isfile(image_similar_vector_file):

        vectorize_features_images(
            image_similar_dir, IMAGE_SIZE, preprocessor, model, image_similar_vector_file)

    print("Load vec file for image")
    image_vec_dict = load_vectors(image_vector_file)
    print("Load vec file for image similar")
    image_similar_vec_dict = load_vectors(image_similar_vector_file)

    image_vec_dict.update(image_similar_vec_dict)

    print(len(image_vec_dict))
    print(len(image_similar_vec_dict))
    print("Done")
    #print(image_vec_dict.keys())
    

    triples = get_triples(image_dir, image_similar_dir)

    print(len(triples))

    #exit()

    # time.sleep(30)

    #triples_train, triples_test = train_test_split(triples, test_size=0.1)
    triples_train, triples_val = train_test_split(triples, test_size=0.3)

    print("Train :{:d}".format(len(triples_train)))
    print("Validation :{:d}".format(len(triples_val)))
    #print("Test :{:d}".format(len(triples_test)))

    train_gen = data_generator(
        triples_train, VECTOR_SIZE, image_vec_dict, BATCH_SIZE)
    val_gen = data_generator(triples_val, VECTOR_SIZE,
                             image_vec_dict, BATCH_SIZE)

    siamese_model = get_siamese_model()
    siamese_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint(os.path.join(
        DATA_DIR, "model.h5"), save_best_only=True)
    train_steps_per_epoch = len(triples_train) // BATCH_SIZE
    val_steps_per_epoch = len(triples_val) // BATCH_SIZE

    history = siamese_model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                                          epochs=NUM_EPOCHS,
                                          validation_data=val_gen, validation_steps=val_steps_per_epoch,
                                          callbacks=[checkpoint])

    plot_history(history)


if __name__ == "__main__":
    main()
