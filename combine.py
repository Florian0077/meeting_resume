import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings

warnings.filterwarnings("ignore")


def extract_mfccs(audio_path, sr=None):
    audio, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs.T)
    kmeans = KMeans(n_clusters=2)  # Ajuster en fonction du nombre de locuteurs attendus
    speaker_labels = kmeans.fit_predict(mfccs_scaled)

    hop_length = 512  # Par défaut dans librosa
    speaker_segments = []

    for i, label in enumerate(speaker_labels):
        timestamp_ms = (i * hop_length / sr) * 1000
        speaker_segments.append((timestamp_ms, label))

    return speaker_segments


def transcribe_whisper(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    audio, sr = librosa.load(audio_path, sr=16000)

    result = pipe(
        audio,
        generate_kwargs={
            "language": "fr",
            # "task": "transcribe",
            "num_beams": 5,
            "temperature": 0,
            "suppress_tokens": None,
        },
    )

    return result["chunks"]


def combine_diarization_and_transcription(audio_path):
    speaker_segments = extract_mfccs(audio_path)
    transcription_chunks = transcribe_whisper(audio_path)

    combined_result = []

    for chunk in transcription_chunks:
        start_time = chunk["timestamp"][0] * 1000  # Convert to ms
        end_time = (
            chunk["timestamp"][1] * 1000
            if chunk["timestamp"][1] is not None
            else float("inf")
        )

        # Find the most frequent speaker in this time range
        speakers_in_range = [
            seg[1] for seg in speaker_segments if start_time <= seg[0] < end_time
        ]
        if speakers_in_range:
            most_common_speaker = max(
                set(speakers_in_range), key=speakers_in_range.count
            )
        else:
            most_common_speaker = -1  # Unknown speaker

        combined_result.append(f"Speaker_{most_common_speaker} || {chunk['text']}")

    return combined_result


if __name__ == "__main__":
    audio_path = os.path.join(os.getcwd(), "temp", "audio.wav")
    result = combine_diarization_and_transcription(audio_path)
    for line in result:
        print(line)
