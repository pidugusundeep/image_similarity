import csv
import os
import sys

import cv2
import ffmpeg
import numpy as np
from annoy import AnnoyIndex
from keras.models import Model
from scipy.io import wavfile
from vggish import vggish
from vggish.preprocess_sound import preprocess_sound

BATCH_SIZE = 32
VIDEO_DIR = "/home/andrei/temp/video"

FEATURES_COUNT = 128

# generate model and preprocessor for inception


def get_vggish():
    """ return inception3 model and preprocessor """

    vggish_model = vggish.VGGish(
        weights="audioset", include_top=True)

    model = Model(inputs=vggish_model.input,
                  outputs=vggish_model.get_layer("vggish_fc2").output)

    return model, preprocess_sound


# main function
def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]

    index = AnnoyIndex(FEATURES_COUNT)

    videos = os.listdir(input_dir)

    videos.sort()

    model, preprocessor = get_vggish()

    num_vectors = 0

    videos_added = []

    for video in videos:
        print("Process "+video)

        stream = ffmpeg.input(os.path.join(input_dir, video))
        wav_name = video+".wav"
        wav_path = os.path.join("temp", wav_name)
        stream = ffmpeg.output(stream, wav_path, ac=1)
        stream = stream.overwrite_output()
        ffmpeg.run(stream)

        sr, wav_data = wavfile.read(wav_path)
        os.remove(wav_path)

        length = sr * 60

        # print(length)

        cur_wav = wav_data[0:length]
        cur_spectro = preprocessor(cur_wav, sr)
        cur_wav = cur_wav / 32768.0
        # print(cur_spectro.shape)
        cur_spectro = np.expand_dims(cur_spectro, 3)
        # print(cur_spectro.shape)

        result = model.predict(cur_spectro)
        result = np.sum(result, axis=0)
        print(result.shape)

        index.add_item(num_vectors, result)
        videos_added.append(video)
        num_vectors += 1
        if num_vectors % 100 == 0:
            print("{:d} vectors generated".format(num_vectors))

    

    print("{:d} vectors generated".format(num_vectors))
    index.build(10)  # 10 trees
    index.save(os.path.join(output_dir, model_name+".ann"))

    with open(os.path.join(output_dir, model_name+".csv"), mode='w') as filep:
        writer = csv.writer(filep, delimiter=',')
        for video in videos_added:
            writer.writerow([os.path.join(input_dir, video)])


if __name__ == "__main__":
    main()
