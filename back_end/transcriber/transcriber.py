
import elements.cleanup
import elements.segmentation
import elements.vosk_transcriber
import elements.fast_whisper_transcriber
import elements.finalizers

from datetime import datetime
from typing import Dict
from enum import Enum
import threading
import os
import wave
import json

from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer




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








class Transcriber():
    class Status_code(Enum):
        READY=0
        ERROR=1
        BUSY=2
    class Status_message(Enum):
        READY="Ready"
        ERROR="Error accured during transcribing process"
        BUSY="Transcribing in progress"
    def __init__(self):
        self.status=Transcriber.Status_code.READY
        self.text="Nothing is transcribed."
        self.path=""
        self.lock = threading.Lock() 
        
    def _transcribe_worker(self):
        try:
            result = model.transcribe(self.path)
            with self.lock:
                self.text = result["text"]
                self.status = Transcriber.Status.READY
        except Exception as e:
            with self.lock:
                self.status = Transcriber.Status.READY
                self.text = str(e)

    def transcribe(self,audio_path):
        with self.lock:
            if self.status==Transcriber.Status_code.BUSY:
                return
            self.status=Transcriber.Status_code.BUSY
            self.text=""
            self.path=audio_path
            try:
                self.path=elements.cleanup.cleanup_bisic_wrapper(self.path)
                if os.path.exists(self.path):
                    #here
                    thred=threading.Thread(target=self._transcribe_worker,daemon=True)
                    thred.start()
                else:
                    self.status=Transcriber.Status_code.ERROR
                    self.text = "Transcribing file can't be found."
            except Exception as e:
                self.status=Transcriber.Status_code.ERROR
                self.text = ""









