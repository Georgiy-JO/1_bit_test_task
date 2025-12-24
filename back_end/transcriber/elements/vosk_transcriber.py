
import wave
import json
from vosk import Model, KaldiRecognizer

def transcribe_vosk(audio_file, model_path="vosk-model-small-ru"):
    wf = wave.open(audio_file, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    # Concatenate text
    text = " ".join([r.get("text", "") for r in results])
    return text

# Example for all A segments
vosk_transcripts = []
for i, seg in enumerate(vad_segments):
    filename = f"A_segment_{i}.wav"  # Assuming you split A like B
    text = transcribe_vosk(filename)
    vosk_transcripts.append({"speaker": seg['speaker'], "start": seg['start'], "end": seg['end'], "text": text})

