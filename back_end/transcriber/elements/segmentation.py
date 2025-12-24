
from pyannote.audio import Pipeline
import torch
from silero_vad import VADIterator, utils

def diarize_speakers(audio_file):
    """
    Returns list of speaker segments: (start_time, end_time, speaker_id)
    """
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

