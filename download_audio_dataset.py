import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


os.makedirs("audio_dataset/real", exist_ok=True)
os.makedirs("audio_dataset/fake", exist_ok=True)


def generate_audio(freq=220, duration=2, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio, sr


def save_spectrogram(audio, sr, path):
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db, sr=sr)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


for i in range(50):
   
    audio, sr = generate_audio(freq=220 + i)
    save_spectrogram(audio, sr, f"audio_dataset/real/{i}.png")

   
    noise = audio + 0.3 * np.random.randn(len(audio))
    save_spectrogram(noise, sr, f"audio_dataset/fake/{i}.png")

    print(f"Generated {i}")

print("\nDataset ready!")