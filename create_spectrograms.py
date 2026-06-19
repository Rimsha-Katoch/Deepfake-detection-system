import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy import VideoFileClip


VIDEO_DATASET = "dataset"   # your existing dataset
OUTPUT_DATASET = "audio_dataset"


def extract_audio(video_path, output_path):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_path, logger=None)
    except:
        print("Error processing:", video_path)


def create_spectrogram(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        plt.figure(figsize=(3, 3))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(S_db, sr=sr)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except:
        print("Error spectrogram:", audio_path)


def process_folder(folder_name):
    input_folder = os.path.join(VIDEO_DATASET, folder_name)
    output_folder = os.path.join(OUTPUT_DATASET, folder_name)

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(input_folder, file)

            audio_path = "temp.wav"
            extract_audio(video_path, audio_path)

            save_path = os.path.join(output_folder, file.replace(".mp4", ".png"))
            create_spectrogram(audio_path, save_path)

            print("Processed:", file)


if __name__ == "__main__":
    process_folder("real")
    process_folder("fake")