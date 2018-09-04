import librosa
import librosa.display
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np

stream = ffmpeg.input('/home/andrei/Downloads/parker.mp4')
stream = ffmpeg.output(stream, '/home/andrei/Downloads/output.wav', ac=1)
stream = stream.overwrite_output()
ffmpeg.run(stream)

output = "/home/andrei/Downloads/output.wav"

y, sr = librosa.load(output)

#mel = librosa.feature.melspectrogram(y=y, sr=sr)




D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
