import audio_process
import diarization
import whisper_process
import os
import filelock

print(filelock.__file__)

if __name__ == "__main__":
    audio_path = audio_process.audio_prepare(
        os.path.join(os.getcwd(), "temp", "video.mp4"), 16000
    )

    audio_diarize = diarization.extract_mfccs(audio_path, 16000)
    audio_transcribe = whisper_process.transcribe_whisper(audio_path)

    print(audio_transcribe)
