
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8_float16")

def transcribe_whisper(segment_file):
    segments, info = model.transcribe(
        segment_file,
        beam_size=5,
        temperature=[0.0, 0.2, 0.4]
    )
    text = " ".join([seg.text for seg in segments])
    return text

# Transcribe all B segments
whisper_transcripts = []
for i, seg_file in enumerate(B_segment_files):
    text = transcribe_whisper(seg_file)
    whisper_transcripts.append({
        "speaker": speaker_segments[i]['speaker'],
        "start": speaker_segments[i]['start'],
        "end": speaker_segments[i]['end'],
        "text": text
    })
