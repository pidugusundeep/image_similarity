# import the necessary packages
import base64
import io
import json
import sys
import time
import uuid

import flask
import numpy as np
import redis
from PIL import Image

app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)


IMAGE_SIZE = 224
IMAGE_QUEUE = "image"


def base64_encode_image(image):
    """ base64 encode the input NumPy array """
    return base64.b64encode(image).decode("utf-8")


def base64_decode_image(data, dtype, shape):
    """ decode image """

    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object

    if sys.version_info.major == 3:
        data = bytes(data, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    image = np.frombuffer(base64.decodebytes(data), dtype=dtype)
    image = image.reshape(shape)

    # return the decoded image
    return image


@app.route("/search", methods=["POST"])
def search():

    data = {"queue": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            image = flask.request.files["image"].read()

            id_image = str(uuid.uuid4())
            d = {"id": id_image, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))
            data["queue"] = True
            data["id"] = id_image

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def main():
    """ main """

    app.run()


if __name__ == "__main__":
    main()
