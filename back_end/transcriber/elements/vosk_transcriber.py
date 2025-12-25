
from models import vosk_model, KaldiRecognizer
from transcribers_service import  load_audio_array, extract_segment_from_array
import io
import json
import soundfile as sf


def transcribe_vosk_segments(audio_file, segments):

    wav_np, sr = load_audio_array(audio_file)

    results = []
    for seg in segments:
        arr, sr = extract_segment_from_array(wav_np, sr, seg)
        if len(arr) < 300:
            results.append({**seg, "text": ""})
            continue

        # convert numpy → PCM bytes
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        buf.seek(0)

        rec = KaldiRecognizer(vosk_model, sr)
        while True:
            data = buf.read(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        text = json.loads(rec.FinalResult()).get("text", "")
        results.append({**seg, "text": text})

    return results