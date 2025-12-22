import whisper 
from fastapi import FastAPI, file, UploadFile
from fastapi.responses import JSONResponse
import os
import threading


def transcribe_audio(path, text):
    model=whisper.load_model("base")
    result=model.transcribe(path)
    text = result["text"]

class Transcriber():
    def __init__(self,new_id):
        id=new_id
        status="Ready"
        text=""
        path=""
        
    def transcribe(audio_path):
        status="Buisy"
        text=""
        path=audio_path
        if os.path.exists(path):
            thred=threading.Thread(target=transcribe_audio,args=(path,text))
            thred.start()
            thred.join
            status="Ready"
        else:
            status="ERROR"
    def __eq__(self,other):
        if isinstance(other, int):
            return self.id==other
        else:
            return False

user_array=[]

app=FastAPI()

@app.post("/transcribe/{id}")
async def transcribe(file: UploadFile=File(...)):                #bring file here somehow!
    contents=await file.read()
    with open("audio.mp3","wb") as f: f.write(file)
    if user_array.index(id).status!="Buisy":
        user_array.index(id).transcribe(audio_path)
    return {"message": user_array.index(id).status}

@app.get("/initialize/{id}")
def initialize():
    if(user_array.index(id)):
        if(user_array.index(id).status=="Buisy"):
            thred=threading.Thread(target=lambda elem : elem.status!="Buisy", args= user_array.index(id))
            thred.start()
            thred.join
        user_array.remove(id)
    user_array.append(id)
    return {"message": user_array.index(id).status}

@app.get("/status/{id}")
def get_status():
    return {"message": user_array.index(id).status}

@app.get("/get_transcribtion/{id}")
def get_transcribtion():
    if user_array.index(id).status!="Buisy":
        os.remove(user_array.index(id).path)
        return {"message": user_array.index(id).text}
    return {"message": user_array.index(id).status}
    

if __name__=="__main__":
    app.run(debug=True)
