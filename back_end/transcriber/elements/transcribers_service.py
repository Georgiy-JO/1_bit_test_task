
import librosa

def load_audio_array(audio_file):
    wav_np, sr = librosa.load(audio_file, sr=None)
    return wav_np, sr

def extract_segment_from_array(wav_np, sr, seg):
    start = int(seg["start"] * sr)
    end   = int(seg["end"] * sr)
    return wav_np[start:end], sr
