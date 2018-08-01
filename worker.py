""" worker to process image """

import base64
import json
import sys

import cv2
import numpy as np
import redis
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


def extract_features(model, image_base64):
    """ process data """

    # print(data)
    
    
    if sys.version_info.major == 3:
        image_base64 = bytes(image_base64, encoding="utf-8")

    image_data = bytearray(base64.decodebytes(image_base64))
    file_bytes = np.asarray(image_data, dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

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


def main():
    """ main function """

    print("Load vectors.")
    vectors=load_vectors("data/vectors.tsv")

    print("Load features model.")
    base_model = ResNet50(weights='imagenet', include_top=True)
    features_model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer("flatten_1").output)

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
            features = extract_features(features_model, data["image"])
            
            features=np.expand_dims(features,axis=0)
            print(features)

            max_1=0
            max_2=0
            max_1_name=""
            max_2_name=""
            for key, value in vectors.items():

                print(key,value)

                value=np.expand_dims(value,axis=0)
                r=model.predict([features,value])
                

                print(r)
                r=r[0]

                if(r[0]>max_1):
                    max_1=r[0]
                    max_1_name=key
                if(r[1]>max_2):
                    max_2=r[1]
                    max_2_name=key

                print(r)


            print(max_1,max_1_name)
            print(max_2,max_2_name)


            d=vectors["100200.jpg"]
            print("---"+str(model.predict([features,value])))


           

            db_redis.setex("result:"+result["id"], 300, json.dumps(result))


if __name__ == "__main__":
    main()
