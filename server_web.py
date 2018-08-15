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
            file = flask.request.files["image"]

            id_image = str(uuid.uuid4())
            ext=file.filename.rsplit('.',1)[1]
            temp_path=os.path.join("temp",id_image+"."+ext)
            file.save(temp_path)

            

            d = {"id": id_image, "image": temp_path}

            db.lpush(IMAGE_QUEUE_LIST, json.dumps(d))
            data["queue"] = True
            data["id"] = id_image

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/result/<string:image_id>", methods=["GET"])
def status(image_id):

    result_0 = db.get("result:"+image_id+":"+"image_0")
    result_1 = db.get("result:"+image_id+":"+"image_1")

    if not result_0 and not result_1:
        return flask.jsonify({})

    result_list = []

    if result_0:
        result_0 = json.loads(result_0.decode("utf-8"))
        # print(result_0)
        result_list.extend(result_0["images"])

    if result_1:
        result_1 = json.loads(result_1.decode("utf-8"))
        result_list.extend(result_1["images"])

    #print(result_list)

    result_list = sorted(result_list, key=lambda k: k["distance"])

    result_list=result_list[:15]

    data = {}
    data["id"] = image_id
    data["images"] = result_list

    return flask.jsonify(data)


@app.route("/result/images/<path:file_name>", methods=["GET"])
def image(file_name):

    path_folder = os.path.join(app.root_path, "/home/andrei/temp/")
    return flask.send_from_directory(directory=path_folder, filename=file_name)


def main():
    """ main """

    app.run(debug=True, host="0.0.0.0")


if __name__ == "__main__":
    main()
