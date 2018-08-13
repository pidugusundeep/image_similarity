""" worker to process image """

import base64
import bisect
import json
import sys
from collections import deque

import cv2
import numpy as np
import redis
from keras.applications import inception_v3
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, load_model

IMAGE_QUEUE_LIST = "image_queue"


def base64_decode_image(data, dtype, shape):
    """ decode image """

    if sys.version_info.major == 3:
        data = bytes(data, encoding="utf-8")

    image = np.frombuffer(base64.decodebytes(data), dtype=dtype)
    image = image.reshape(shape)

    # return the decoded image
    return image


def extract_features(model, preprocessor, image_base64):
    """ process data """

    # print(data)

    if sys.version_info.major == 3:
        image_base64 = bytes(image_base64, encoding="utf-8")

    image_data = bytearray(base64.decodebytes(image_base64))
    file_bytes = np.asarray(image_data, dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)

    image = preprocessor(image)

    features = model.predict(image)

    features = features.squeeze()

    return features


def load_vectors(vector_file):
    """ load features file """

    vec_dict = {}
    with open(vector_file, "r") as file:
        for line in file:
            image_name, image_vec = line.strip().split("\t")
            vec = np.array([float(v) for v in image_vec.split(",")])
            vec_dict[image_name] = vec

    return vec_dict


def get_inception3():
    """ return inception3 model and preprocessor """

    inception_model = inception_v3.InceptionV3(
        weights="imagenet", include_top=True)

    # for i, layer in enumerate(inception_model.layers):

    #    print(i, layer.name)
    #    print(inception_model.get_layer(layer.name).output_shape)
    # exit(0)

    # inception_model.summary()

    model = Model(inputs=inception_model.input,
                  outputs=inception_model.get_layer("avg_pool").output)

    preprocessor = inception_v3.preprocess_input

    return model, preprocessor


def main():
    """ main function """

    print("Load vectors.")
    vectors = load_vectors("data/vectors.tsv")

    print("Load features model.")
    #base_model = ResNet50(weights='imagenet', include_top=True)
    # features_model = Model(inputs=base_model.input,
    #              outputs=base_model.get_layer("flatten_1").output)
    features_model, preprocessor = get_inception3()

    print("Load model.")
    model = load_model('data/model.h5')

    db_redis = redis.StrictRedis(host="localhost", port=6379, db=0)
    while True:
        data = db_redis.brpop(IMAGE_QUEUE_LIST, timeout=5)

        # print(data)
        if data:
            data = json.loads(data[1].decode("utf-8"))

            result = {}
            result["id"] = data["id"]

            print("Extract features "+data["id"])
            features = extract_features(
                features_model, preprocessor, data["image"])

            features = np.expand_dims(features, axis=0)
            print(features)

            v = 0

            similar_images = []

            image_count = 0
            for key, value in vectors.items():

                image_count += 1

                value = np.expand_dims(value, axis=0)
                res = model.predict([features, value])

                res = res[0]
                # print(res)

                v = res[1]
                file_name = key
                if not similar_images:
                    similar_images.append(
                        {"name": file_name, "similarity": float(v)})
                else:
                    if v > similar_images[-1]["similarity"]:
                        for idx, val in enumerate(similar_images):
                            if v > val["similarity"]:
                                if len(similar_images) == 9:
                                    similar_images.pop()
                                similar_images.insert(idx,
                                                      {"name": file_name, "similarity": float(v)})
                                break

                if image_count > 50:
                    result["images"] = list(similar_images)
                    db_redis.setex(
                        "result:"+result["id"], 60, json.dumps(result))

            result["images"] = list(similar_images)
            print(result)
            db_redis.setex("result:"+result["id"], 60, json.dumps(result))


if __name__ == "__main__":
    main()
