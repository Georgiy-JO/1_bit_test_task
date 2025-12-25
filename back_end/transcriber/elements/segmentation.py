import torch
import torchaudio
import librosa
from models import silero_model, get_speech_timestamps#, speaker_model, device, encode_segment

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import json

import settings    #for windows
settings

# force torchaudio to use the sox_io backend (classic)
# torchaudio.set_audio_backend("sox_io")  

def run_vad(audio_file):
    wav_np, sr = librosa.load(audio_file, sr=None)  # supports wav/mp3
    wav = torch.from_numpy(wav_np).unsqueeze(0)     # shape [1, n_samples]
    wac = wav.square()
    timestamps = get_speech_timestamps(wac, silero_model, sampling_rate=sr)
    return wav, sr, timestamps
"""
def run_vad(audio_file):
    wav, sr = torchaudio.load(audio_file)
    wac=wav.square()
    timestamps = get_speech_timestamps(wac, silero_model, sampling_rate=sr)
    return wav, sr, timestamps
"""

"""
# i'm not working
def diarize_chunks(chunks: list, wav, sr):
    embeddings = []
    segments = []

    for ts in chunks:
        st, en = ts['start'], ts['end']
        seg = wav[int(st*sr): int(en*sr)]
        emb_np = encode_segment(seg) 
        embeddings.append(emb_np)
        segments.append((st, en))

    if not embeddings:
        return []

    labels = AgglomerativeClustering(
        distance_threshold=1.0, n_clusters=None
    ).fit_predict(embeddings)

    results = []
    for (st, en), lab in zip(segments, labels):
        results.append({"start": st, "end": en, "speaker": f"SPK_{lab}"})

    return results
"""
def segmentation_wrapper (audio_file, output_json=None):
    wav, sr, timestamps = run_vad(audio_file)

    chunks = [
        {"start": t["start"] / sr, "end": t["end"] / sr}
        for t in timestamps
    ]

    # diar = diarize_chunks(chunks, wav, sr)
    diar=chunks

    if output_json:
        with open(output_json, "w") as f:
            json.dump(diar, f, indent=2)

    return diar

# print (segmentation_pipeline("tmp/call_conv_0.mp3"))

