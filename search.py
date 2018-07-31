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


db = redis.StrictRedis(host="localhost", port=6379, db=0)


def main():
    """ main """

    app.run()


if __name__ == "__main__":
    main()
