""" worker to process image """

import json
import os

import cv2
import ffmpeg
import numpy as np
import redis
from keras.applications import inception_v3
from keras.models import Model
from scipy.io import wavfile
from vggish import vggish
from vggish.preprocess_sound import preprocess_sound

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


def extract_features_from_video(model, preprocessor, video_path):
    """ process data """

    print(video_path)

    images = []
    vector = np.zeros(2048, dtype="float32")
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (299, 299))
            images.append(frame)
            # print(len(images))
            if(len(images) == 32):
                x_data = preprocessor(
                    np.array(images, dtype="float32"))
                vectors = model.predict(x_data)
                for i in range(vectors.shape[0]):
                    vector = vector+vectors[i]
                del images[:]

        else:
            break

    cap.release()

    return vector


def extract_sound_features_from_video(model, preprocessor, video_path):
    """ process data """

    print(video_path)
    stream = ffmpeg.input(video_path)
    wav_path = video_path+".wav"
    stream = ffmpeg.output(stream, wav_path, ac=1)
    stream = stream.overwrite_output()
    ffmpeg.run(stream)

    sr, wav_data = wavfile.read(wav_path)
    os.remove(wav_path)

    length = sr * 120

    cur_wav = wav_data[0:length]
    cur_spectro = preprocessor(cur_wav, sr)
    cur_wav = cur_wav / 32768.0
    # print(cur_spectro.shape)
    cur_spectro = np.expand_dims(cur_spectro, 3)
    # print(cur_spectro.shape)

    result = model.predict(cur_spectro)
    result = np.sum(result, axis=0)

    return result


def get_vggish():
    """ return inception3 model and preprocessor """

    vggish_model = vggish.VGGish(
        weights="audioset", include_top=True)

    model = Model(inputs=vggish_model.input,
                  outputs=vggish_model.get_layer("vggish_fc2").output)

    return model, preprocess_sound


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
    sound_features_model, sound_preprocessor = get_vggish()

    db = redis.StrictRedis(host="localhost", port=6379, db=0)
    while True:
        data = db.brpop(IMAGE_QUEUE_LIST, timeout=5)

        # print(data)
        if data:
            data = json.loads(data[1].decode("utf-8"))

            result = {}
            result["id"] = data["id"]

            print("Extract features "+data["id"])
            media_path = data["image"]
            media_type = data["media_type"]
            if media_type == "movie":
                features = extract_features_from_video(
                    features_model, preprocessor, media_path)
                features_audio = extract_sound_features_from_video(
                    sound_features_model, sound_preprocessor, media_path)

            else:
                features = extract_features(
                    features_model, preprocessor, media_path)

            features = np.expand_dims(features, axis=0)
            features = features[0]
            # print(features[0])

            features = features.tolist()
            # print(features[0])

            result["features"] = features
            result_json = json.dumps(result)

            index_image = ["image_0", "image_1", "image_2",
                           "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "validation", "video_1"]
            index_video = ["video_1"]
            index_audio = ["audio_0"]

            if media_type == "image":
                for index in index_image:
                    db.lpush(index+"_index_queue", result_json)
            else:
                #for index in index_video:
                #    db.lpush(index+"_index_queue", result_json)
                
                result["id"] = data["id"]
                features_audio = np.expand_dims(features_audio, axis=0)
                features_audio = features_audio[0]
                # print(features[0])

                features_audio = features_audio.tolist()
                result["features"] = features_audio
                result_json = json.dumps(result)

                for index in index_audio:
                    db.lpush(index+"_index_queue", result_json)

            print("forward to search in index "+data["id"])

            os.remove(media_path)


if __name__ == "__main__":
    main()
