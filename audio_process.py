from pydub import AudioSegment
import os


def open_audio_file(file_name):
    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, file_name)
    audio = AudioSegment.from_file(file_path)
    return audio


def change_frequency(audio, new_freq):
    audio = audio.set_frame_rate(new_freq)
    return audio


def audio_prepare(audio, new_freq):
    audio = open_audio_file("temp/video.mp4")
    audio = change_frequency(audio, new_freq)
    # Sauvegarde du fichier audio
    audio_path = os.path.join(os.getcwd(), "temp", "audio.wav")
    saved = audio.export(audio_path, format="wav")
    return audio_path
