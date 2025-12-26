# Final stable version: transcriber.py

import os
import json
import torch
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import subprocess
from datetime import datetime
from typing import List, Dict
from faster_whisper import WhisperModel
from vosk import Model as VoskModel, KaldiRecognizer  # comment if don't want Vosk 
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
from transformers import pipeline
import soundfile as sf
import io
import gc

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
from enum import Enum
import uvicorn


# Paths (adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "venv"))
ffmpeg_path = os.path.abspath(os.path.join(ENV_DIR, "ffmpeg", "bin", "ffmpeg.exe"))

# Device & compute type – safe defaults
device = "cpu" #="cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8" #= "float16" if device == "cuda" else "int8" # Optimize for device

# ==================== MODEL LOADING (once at startup) ====================
try:
    print("[INIT] Loading Silero VAD...")
    silero_model, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                trust_repo=True)
    get_speech_timestamps = silero_utils[0]
    print("Loaded Silero VAD.")
except Exception as e:
    print(f"Error loading Silero: {e}")
    raise

try:
    print("[INIT] Loading SpeechBrain speaker encoder...")
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models",
        local_strategy=LocalStrategy.COPY_SKIP_CACHE
    )
    print("Loaded SpeechBrain.")
except Exception as e:
    print(f"Error loading SpeechBrain: {e}")
    raise

try:
    print("[INIT] Loading Faster-Whisper model...")
    whisper_model = WhisperModel("small", device=device, compute_type=compute_type) # try medium time will probably double
    print("Loaded Faster-Whisper.")
except Exception as e:
    print(f"Error loading Whisper: {e}")
    raise

# Vosk loading – commented out (easy toggle)---------------------------------------------------------------------------------
try:
    print("[INIT] Loading Vosk model...")
    vosk_model_path = "models/vosk"  # Path to unzipped vosk-model-ru-0.42
    if not os.path.exists(vosk_model_path):
        raise RuntimeError("Vosk model path not found.")
    vosk_model = VoskModel(vosk_model_path)
    print("Loaded Vosk.")
except Exception as e:
    print(f"Error loading Vosk: {e}")
    raise
#-----------------------------------------------------------------------------------------------------------------------------

try:
    print("[INIT] Loading T5 for polishing (optional)...")
    polish_pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    print("Loaded T5.")
except Exception as e:
    print(f"Error loading T5: {e}. Polishing disabled.")
    polish_pipe = None

# ==================== AUDIO CLEANUP ====================

def cleanup_audio(input_file: str, output_file: str = "cleaned_audio.wav") -> str:
    # print(f"[CLEANUP] Input file: {os.path.abspath(input_file)}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")
    if not os.path.exists(ffmpeg_path):
        raise RuntimeError(f"ffmpeg.exe not found at {ffmpeg_path}")

    cmd = [
        ffmpeg_path,
        "-i", input_file,
        "-ac", "1",      # Mono
        "-ar", "16000",  # 16kHz
        "-y", output_file
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # print("[CLEANUP] Success")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed (code {e.returncode})")
        # print("STDOUT:", e.stdout)
        # print("STDERR:", e.stderr)
        raise

    return output_file

# ==================== VAD + DIARIZATION ====================

def vad_and_diarize(audio_file: str) -> List[Dict]:
    wav_np, sr = librosa.load(audio_file, sr=16000)
    wav_torch = torch.from_numpy(wav_np).unsqueeze(0)

    timestamps = get_speech_timestamps(wav_torch, silero_model, sampling_rate=sr)
    segments = [{"start": t['start'] / sr, "end": t['end'] / sr} for t in timestamps]

    embeddings = []
    valid_segments = []  # Keep track of which segments we kept for clustering
    for seg in segments:
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        if end_sample - start_sample < 8000:  # Skip <0.5s
            continue
        seg_audio = wav_torch[:, start_sample:end_sample]
        with torch.no_grad():
            emb = speaker_model.encode_batch(seg_audio)
        embeddings.append(emb.squeeze().cpu().numpy())
        valid_segments.append(seg)

    # If too few reliable segments → fallback: all Speaker_1 (safe)
    if len(embeddings) < 2:
        return [{"start": s["start"], "end": s["end"], "speaker": "Speaker_1"} for s in segments]
    
    # FIXED: Force exactly 2 clusters
    from sklearn.cluster import KMeans  # Better than Agglomerative for fixed n=2
    clustering = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels = clustering.fit_predict(np.array(embeddings))

    # Assign speakers to valid segments
    speaker_map = {}
    for i, seg in enumerate(valid_segments):
        speaker_map[id(seg)] = f"Speaker_{labels[i] + 1}"

    # Now assign to ALL segments (including short ones) by nearest neighbor in time
    # Simple but effective: assign short segment to the closest long segment's speaker
    diarized_segments = []
    for seg in segments:
        if id(seg) in speaker_map:
            speaker = speaker_map[id(seg)]
        else:
            # Find closest valid segment by start time
            closest = min(valid_segments, key=lambda x: abs(x["start"] - seg["start"]))
            speaker = speaker_map[id(closest)]
        diarized_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker
        })

    return sorted(diarized_segments, key=lambda x: x["start"])

# ==================== TRANSCRIPTION (SEQUENTIAL) ====================
def remove_hallucinations(text: str) -> str:
    common_hallucinations = [
        "Редактор субтитров",
        "Субтитры сделал",
        "Спасибо за субтитры",
        "Субтитры подогнал",
        "Корректор",
        # etc??
    ]
    for phrase in common_hallucinations:
        text = text.replace(phrase, "").strip()
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transcribe_segment(args):
    audio_file, seg = args
    wav_np, sr = librosa.load(audio_file, sr=16000, offset=seg['start'], duration=seg['end'] - seg['start'])

    if len(wav_np) < 500:
        return {**seg, "text": ""}

    # Primary: Faster-Whisper (Russian)
    whisper_segments, _ = whisper_model.transcribe(wav_np, beam_size=10, temperature=0.0, language="ru")
    text = " ".join([s.text for s in whisper_segments]).strip()

    # === Vosk fallback  =======================================
    if not text:  # Only if Whisper returned empty
        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="WAV")
        buf.seek(0)
        rec = KaldiRecognizer(vosk_model, sr)
        while True:
            data = buf.read(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
        vosk_text = json.loads(rec.FinalResult()).get("text", "").strip()
        text = vosk_text
    # === Vosk fallback  =======================================

    text=remove_hallucinations(text)
    return {**seg, "text": text}

def transcribe_segments(audio_file: str, segments: List[Dict]) -> List[Dict]:
    transcribed = []
    total = len(segments)
    for i, seg in enumerate(segments):
        if (i+1 == total/4 or i+1 == total/2 or i+1 == total/3 or  i+1 == (total-total/4)):
            print(f"[TRANSCRIBE] Processing segment {i+1}/{total} ({seg['start']:.1f}s → {seg['end']:.1f}s)")
        result = transcribe_segment((audio_file, seg))
        transcribed.append(result)
    gc.collect()
    return transcribed

# ==================== POLISHING & MERGING ====================

def polish_text(text: str) -> str:
    if not text or polish_pipe is None:
        return text
    prompt = f"Исправьте грамматику и пунктуацию в этом медицинском русском тексте, проверь, что слова согласуются в смысловом и грамматическом смысле и замени на похожие но подходящие: {text}"
    return polish_pipe(prompt, max_new_tokens=512)[0]['generated_text'].strip()

def merge_to_dialogue(transcribed_segments: List[Dict], use_polish: bool = True) -> Dict:
    if not transcribed_segments:
        return {}

    merged = {}
    current_speaker = transcribed_segments[0]['speaker']
    current_text = transcribed_segments[0]['text']

    for seg in transcribed_segments[1:]:
        if seg['speaker'] == current_speaker:
            current_text += " " + seg['text']
        else:
            if use_polish:
                current_text = polish_text(current_text)
            merged[current_speaker] = current_text.strip()
            current_speaker = seg['speaker']
            current_text = seg['text']

    if use_polish:
        current_text = polish_text(current_text)
    merged[current_speaker] = current_text.strip()

    return merged

# ==================== MAIN PIPELINE ====================

def transcribe_audio(input_audio: str, output_json: str = "transcript.json", use_polish: bool = False) -> None:
    start_time = datetime.now()

    cleaned_audio = cleanup_audio(input_audio)
    segments = vad_and_diarize(cleaned_audio)
    print(f"[INFO] Detected {len(segments)} speech segments")

    transcribed_segments = transcribe_segments(cleaned_audio, segments)

    dialogue = merge_to_dialogue(transcribed_segments, use_polish=use_polish)

    # with open(output_json, "w", encoding="utf-8") as f:
    #     json.dump(dialogue, f, ensure_ascii=False, indent=2)

    os.remove(cleaned_audio)
    print(f"✅ Transcription complete in {(datetime.now() - start_time).total_seconds():.1f}s")
    # print(f"   Saved to {output_json}")
    return dialogue

# if __name__ == "__main__":
#     transcribe_audio("clien_example.m4a", use_polish=True)


# ==================== FASTAPI PART ====================

# Allowed audio files
allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}

app=FastAPI()

class Transcriber:
    class StatusCode(Enum):
        READY=0
        ERROR=1
        BUSY=2
    StatusMessage = ["Ready", "Error occurred during transcribing process", "Transcribing in progress"]

    def __init__(self, audio_file:str):
        self.status=self.StatusCode.READY
        self.result: Dict ={} # Final dialogue
        self.path=audio_file
        # self.error: str = ""

# Global task storage     
tasks: Dict[str, Transcriber] = {}

def background_process(task_id: str):
    task = tasks.get(task_id)
    if not task or task.status==Transcriber.StatusCode.BUSY:
        return #"File is already processing."
    task.status = Transcriber.StatusCode.BUSY
    try:
        result =  transcribe_audio(task.path, use_polish=True)
        task.result=result
        task.status = Transcriber.StatusCode.READY
    except Exception as e:
        task.status = Transcriber.StatusCode.ERROR
    finally:
        if os.path.exists(task.path):
            try:
                os.remove(task.path) 
            except:
                pass   # Ignore cleanup errors


# ==========================================
@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...), background_tasks: BackgroundTasks = None  ):
    file_ext = os.path.splitext(audio_file.filename or "unknown")[1]
    if file_ext not in allowed_extensions:      
        return JSONResponse({
            "task_id": task_id,
            "status_code":Transcriber.StatusCode.ERROR.value, 
            "status_message": "Error: File extension is not allowed or file was not recived", 
            "text": {}})
    task_id = str(uuid4())
    os.makedirs("files", exist_ok=True)
    # filename = f"files/audio_{task_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}{file_ext}"
    filename = f"files/audio_{task_id}{file_ext}"
    with open(filename, "wb") as f:
        f.write(await audio_file.read())
    
    task = Transcriber(filename)
    tasks[task_id] = task
    background_tasks.add_task(background_process, task_id)

    return JSONResponse({
        "task_id": task_id,
        "status_code":task.status.value,
        "status_message":task.StatusMessage[task.status.value],
        "text":{}
    })
    

@app.get("/result/{task_id}")
def get_result(task_id: str):
    task = tasks.get(task_id)
    if not task:
       #raise HTTPException(status_code=404, detail="Task not found")
       return JSONResponse({
            "task_id": task_id,
            "status_code":Transcriber.StatusCode.ERROR.value, 
            "status_message": "Task not found", 
            "text": {}})

    return JSONResponse({
        "task_id": task_id,
        "status_code":task.status.value,
        "status_message":task.StatusMessage[task.status.value],
        "text":task.result
    })

if __name__ == "__main__":
    uvicorn.run("transcriber:app", host="127.0.0.1", port=8000, reload=False)

