import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os


def extract_mfccs(audio_path, sr=None):
    print(audio_path)
    audio, sr = librosa.load(audio_path, sr=sr)
    # Modification ici
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs.T)
    kmeans = KMeans(n_clusters=2)  # Ajuster en fonction du nombre de locuteurs attendus
    speaker_labels = kmeans.fit_predict(mfccs_scaled)

    hop_length = 512  # Par défaut dans librosa, vous pouvez ajuster cela si besoin

    for i, label in enumerate(speaker_labels):
        timestamp_ms = (i * hop_length / sr) * 1000
        print(f"Time Segment {i} (at {timestamp_ms:.2f} ms): Speaker {label}")

    return speaker_labels


if __name__ == "__main__":
    extract_mfccs(os.path.join(os.getcwd(), "temp", "audio.wav"))
