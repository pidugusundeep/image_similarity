# import the necessary packages
import base64
import io
import json
import os
import sys
import time
import uuid

import cv2
import flask
import numpy as np
import redis
from PIL import Image

app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)


IMAGE_SIZE = 224
IMAGE_QUEUE_LIST = "image_queue"


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


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file('index.html')


@app.route("/search", methods=["POST"])
def search():

    data = {"queue": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image_data = flask.request.files["image"].read()

            img_stream = io.BytesIO(image_data)

            file_bytes = np.asarray(
                bytearray(img_stream.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            image_height = image.shape[0]
            image_width = image.shape[1]

            #print(image_height, image_width)
            print(image.dtype)

            id_image = str(uuid.uuid4())
            d = {"id": id_image, "image": base64_encode_image(
                image_data)}

            db.lpush(IMAGE_QUEUE_LIST, json.dumps(d))
            data["queue"] = True
            data["id"] = id_image

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/result/<string:image_id>", methods=["GET"])
def status(image_id):

    data = db.get("result:"+image_id)

    if not data:
        return flask.jsonify({})

    data = json.loads(data.decode("utf-8"))

    return flask.jsonify(data)


@app.route("/result/images/<path:file_name>", methods=["GET"])
def image(file_name):

    path_folder = os.path.join(app.root_path, "/home/andrei/temp/validation/")
    return flask.send_from_directory(directory=path_folder, filename=file_name)


def main():
    """ main """

    app.run(debug=True, host="0.0.0.0")


if __name__ == "__main__":
    main()
