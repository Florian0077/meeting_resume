import warnings

warnings.filterwarnings("ignore")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from IPython.display import Audio as display_Audio, display
import torchaudio
import librosa


# utility functions
def load_recorded_audio(path_audio, input_sample_rate=48000, output_sample_rate=16000):
    # Dataset: convert recorded audio to vector
    waveform, sample_rate = torchaudio.load(path_audio)
    waveform_resampled = torchaudio.functional.resample(
        waveform, orig_freq=input_sample_rate, new_freq=output_sample_rate
    )  # change sample rate to 16000 to match training.
    sample = waveform_resampled.numpy()[0]
    return sample


def run_inference(path_audio, output_lang, pipe):
    sample = load_recorded_audio(path_audio)
    result = pipe(
        sample, generate_kwargs={"language": output_lang, "task": "transcribe"}
    )
    print(result["text"])


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

    # Chargez l'audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Effectuez l'inférence
    result = pipe(
        audio,
        generate_kwargs={
            "language": "fr",
            "task": "transcribe",
            "num_beams": 5,
            "temperature": 0,  # Désactive l'échantillonnage aléatoire
            "suppress_tokens": None,  # Désactive la suppression de tokens
        },
    )

    transcription_text = result["text"]
    return transcription_text
