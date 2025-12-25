import torch
import torchaudio
from back_end.transcriber.elements.models import silero_model, get_speech_timestamps, speaker_model, device

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import json

# import service    #for windows

# service

def run_vad(audio_file):
    wav, sr = torchaudio.load(audio_file)
    wac=wav.square()
    timestamps = get_speech_timestamps(wac, silero_model, sampling_rate=sr)
    return wav, sr, timestamps      #??


def diarize_chunks(chunks: list, wav, sr):
    embeddings = []
    segments = []

    for ts in chunks:
        st, en = ts['start'], ts['end']
        seg = wav[int(st*sr): int(en*sr)].to(device)
        emb = speaker_model.encode_batch(seg.unsqueeze(0))
        emb_np = emb.squeeze().to("cpu").numpy()
        # emb_np = encode_segment(seg) 
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

def segmentation_pipeline (audio_file, output_json=None):
    wav, sr, timestamps = run_vad(audio_file)

    chunks = [
        {"start": t["start"] / sr, "end": t["end"] / sr}
        for t in timestamps
    ]

    diar = diarize_chunks(chunks, wav, sr)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(diar, f, indent=2)

    return diar

segmentation_pipeline







from pyannote.audio import Pipeline

from silero_vad import VADIterator, utils

def diarize_speakers(audio_file): # Returns list of speaker segments: (start_time, end_time, speaker_id)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_HF_TOKEN")
    diarization = pipeline(audio_file)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments



def run_vad(audio_file):
    # Load audio (16kHz mono)
    wav = utils.load_audio(audio_file)
    vad_model, utils_func = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                           model='silero_vad', force_reload=False, verbose=False)
    get_speech_timestamps = utils_func['get_speech_timestamps']

    speech_segments = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    return speech_segments

# Example
vad_segments = run_vad("cleaned_A.wav")
print(vad_segments)

def extract_segments(audio_file, segments, prefix="B_segment"):
    """
    Split audio into segments according to diarization/VAD segments
    """
    segment_files = []
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        out_file = f"{prefix}_{i}.wav"
        cmd = [
            "ffmpeg",
            "-i", audio_file,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            "-y",
            out_file
        ]
        subprocess.run(cmd, check=True)
        segment_files.append(out_file)
    return segment_files

# Example
B_segment_files = extract_segments("cleaned_B.wav", speaker_segments, prefix="B_segment")

