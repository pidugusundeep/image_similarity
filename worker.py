""" worker to process image """

import json
import sys

import cv2
import numpy as np
import redis
from keras.applications import inception_v3
from keras.models import Model

IMAGE_QUEUE_LIST = "image_queue"


def extract_features(model, preprocessor, image_path):
    """ process data """

    print(image_path)

    image = cv2.imread(image_path)

    image = cv2.resize(image, (299, 299))
    image = np.expand_dims(image, axis=0)

    image = preprocessor(image)

    features = model.predict(image)

    features = features.squeeze()

    return features





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

    print("Load features model.")
    features_model, preprocessor = get_inception3()

    db = redis.StrictRedis(host="localhost", port=6379, db=0)
    while True:
        data = db.brpop(IMAGE_QUEUE_LIST, timeout=5)

        # print(data)
        if data:
            data = json.loads(data[1].decode("utf-8"))

            result = {}
            result["id"] = data["id"]

            print("Extract features "+data["id"])
            features = extract_features(
                features_model, preprocessor, data["image"])

            features = np.expand_dims(features, axis=0)
            features = features[0]
            # print(features[0])

            features = features.tolist()
            # print(features[0])

            result["features"] = features
            result_json = json.dumps(result)
            db.lpush("image_0_index_queue", result_json)
            db.lpush("image_1_index_queue", result_json)


if __name__ == "__main__":
    main()
