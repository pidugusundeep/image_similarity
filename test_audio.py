from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import ffmpeg


stream = ffmpeg.input('/home/andrei/Downloads/Texas_1.mp4')
stream = ffmpeg.output(stream, '/home/andrei/Downloads/output.wav', ac=1)
stream = stream.overwrite_output()
ffmpeg.run(stream)

[Fs, x] = audioBasicIO.readAudioFile("/home/andrei/Downloads/output.wav")
print(Fs)

F, f_names = audioFeatureExtraction.stFeatureExtraction(
    x, Fs, 0.050*Fs, 0.025*Fs)

print(F.shape)
plt.subplot(2, 1, 1)
plt.plot(F[0, :])
plt.xlabel('Frame no')
plt.ylabel(f_names[0])
plt.subplot(2, 1, 2)
plt.plot(F[1, :])
plt.xlabel('Frame no')
plt.ylabel(f_names[1])
plt.show()
