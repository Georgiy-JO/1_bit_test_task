
import elements.cleanup
import elements.segmentation
import elements.vosk_transcriber
import elements.fast_whisper_transcriber
import elements.finalizers

from datetime import datetime
from typing import Dict
from enum import Enum

class Transcriber():
    class Status_codes(Enum):
        READY=0
        ERROR=1
        BUSY=2
    class Status_message(Enum):
        READY="Ready"
        ERROR="Error accured during transcribing process"
        BUSY="Transcribing in progress"
    def __init__(self):
        self.status=Transcriber.Status.READY
        self.text=""
        self.path="Nothing is transcribed."
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
            if self.status==Transcriber.Status.BUSY:
                return
            self.status=Transcriber.Status.BUSY
            self.text=""
            self.path=audio_path
            if os.path.exists(self.path):
                thred=threading.Thread(target=self._transcribe_worker,daemon=True)
                thred.start()
            else:
                self.status=Transcriber.Status.READY








