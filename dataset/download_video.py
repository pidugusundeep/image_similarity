# Download video dataset
import csv

from pytube import YouTube

# main function


def main():
    print("Download ")

    videos = []
    with open('video.csv') as csv_file:
        count = 0
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # print(row)
            videos.append(row[0])

    videos = set(videos)
    print(len(videos))
    count = 0
    for video in videos:
        try:
            YouTube('http://youtube.com/watch?v=' +
                    video).streams.first().download(output_path="/home/andrei/temp/video")
        except:
            print("Error "+'http://youtube.com/watch?v=' + video)

        count += 1
        print(count)
        if count > 50:
            break


if __name__ == "__main__":
    main()
