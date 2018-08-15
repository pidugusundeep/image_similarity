""" worker to process image """

import csv
import json
import os
import sys

import redis
from annoy import AnnoyIndex


def main():
    """ main function """

    if len(sys.argv) != 3:
        print("No arguments!!")
        exit()

    dir_name = sys.argv[1]
    index_name = sys.argv[2]

    ann_path = os.path.join(dir_name, index_name+".ann")
    metadata_path = os.path.join(dir_name, index_name+".csv")

    print("Load ann model {}. ".format(ann_path), end="")
    index = AnnoyIndex(2048)
    index.load(ann_path)
    print("Ok.")

    print("Load metadata {}. ".format(metadata_path), end="")
    images = []
    with open(os.path.join(dir_name, index_name+".csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # print(row)
            images.append(row[0])
    print("Ok.")

    db_redis = redis.StrictRedis(host="localhost", port=6379, db=0)
    while True:
        data = db_redis.brpop(index_name+"_index_queue", timeout=5)

        # print(data)
        if data:
            data = json.loads(data[1].decode("utf-8"))

            # print(data)
            result = {}
            result["id"] = data["id"]

            features = data["features"]
            # print(features[0])

            ann_results = index.get_nns_by_vector(
                features, 10, include_distances=True)
            index_results = ann_results[0]
            disntances_results = ann_results[1]

            similar_images = []
            for idx, distance in zip(index_results, disntances_results):
                file_name = images[idx]
                names = file_name.split("/")
                file_name = names[-2]+"/"+names[-1]
                #print(file_name)
                similar_images.append(
                    {"name": file_name, "distance": distance})

            result["images"] = similar_images
            # print(result)
            db_redis.setex(
                "result:"+result["id"]+":"+index_name, 60, json.dumps(result))


if __name__ == "__main__":
    main()
