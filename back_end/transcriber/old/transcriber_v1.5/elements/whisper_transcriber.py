from transcribers_service import  load_audio_array, extract_segment_from_array
from models import whisper_model


def transcribe_whisper_segments(audio_file, segments):
    wav_np, sr = load_audio_array(audio_file)
    results = []

    for seg in segments:
        arr, sr = extract_segment_from_array(wav_np, sr, seg)
        if len(arr) < 300:   # avoid useless <0.3s silence
            results.append({**seg, "text": ""})
            continue

        segments_list, info = whisper_model.transcribe(
            arr,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4]
        )
        text = " ".join([s.text for s in segments_list])
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })
    return results
