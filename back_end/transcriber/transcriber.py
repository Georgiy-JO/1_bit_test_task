
import os
import threading
from datetime import datetime
from typing import Dict
from enum import Enum

import whisper 
from fastapi import FastAPI, File, UploadFile, HTTPException

model = whisper.load_model("base")
app=FastAPI()

class Transcriber():
    class Status(Enum):
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


users: Dict[int, Transcriber] = {}

@app.get("/initialize/{user_id}")
def initialize(user_id: int):
    users[user_id] = Transcriber()
    return {"id":user_id, "status": users[user_id].status, "text": users[user_id].text}

@app.post("/transcribe/{user_id}")
async def transcribe_audio(user_id: int,file: UploadFile = File(...)):
    if user_id not in users:
        initialize(user_id)

    transcriber = users[user_id]

    file_ext = os.path.splitext(file.filename or "unknown")[1]
    if not file_ext:        #add checking file types here: .wav .mp3 .m4a .ogg .flac .mp4 .webm
        return {"id":user_id, "status": "Error: File extension id not allowed", "text": transcriber.text}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"files/audio_{user_id}_{timestamp}{file_ext}"

    contents = await file.read()
    with open(filename, "wb") as f:
        f.write(contents)

    transcriber.transcribe(filename)

    return {"id":user_id, "status": transcriber.status, "text": transcriber.text}

@app.get("/status/{user_id}")
def get_status(user_id: int):
    if user_id not in users:
        initialize(user_id)
    return {"id":user_id, "status": users[user_id].status, "text": users[user_id].text}

@app.get("/result/{user_id}")
def get_result(user_id: int):
    if user_id not in users:
        initialize(user_id)

    transcriber = users[user_id]

    if transcriber.status != Transcriber.Status.BUSY:
        if transcriber.path and os.path.exists(transcriber.path):
            os.remove(transcriber.path)

    return {"id":user_id, "status": transcriber.status, "text": transcriber.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("transcriber:app", host="127.0.0.1", port=8000, reload=True)

