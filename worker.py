""" worker to process image """

import base64
import json
import sys

import cv2
import numpy as np
import redis
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

IMAGE_QUEUE_LIST = "image_queue"


def base64_decode_image(data, dtype, shape):
    """ decode image """

    if sys.version_info.major == 3:
        data = bytes(data, encoding="utf-8")

    image = np.frombuffer(base64.decodebytes(data), dtype=dtype)
    image = image.reshape(shape)

    # return the decoded image
    return image


def process_data(model, data):
    """ process data """

    # print(data)
    print("Work "+data["id"])
    result = {}
    result["id"] = data["id"]

    image_base64 = data["image"]
    if sys.version_info.major == 3:
        image_base64 = bytes(image_base64, encoding="utf-8")

    image_data = bytearray(base64.decodebytes(image_base64))
    file_bytes = np.asarray(image_data, dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    print(image.shape)

    image = preprocess_input(image)

    features = model.predict(image)

    features = features.squeeze()

    print(features)

    return result


def main():
    """ main function """

    base_model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer("flatten_1").output)

    db_redis = redis.StrictRedis(host="localhost", port=6379, db=0)
    while True:
        data = db_redis.brpop(IMAGE_QUEUE_LIST, timeout=5)

        # print(data)
        if data:
            data = json.loads(data[1].decode("utf-8"))

            result = process_data(model, data)

            db_redis.setex("result:"+result["id"], 300, json.dumps(result))


if __name__ == "__main__":
    main()
