# Global
## BackEnd
### Transcribing
#### Architecture improvements

|Current	|Professional|
|Threading	|BackgroundTasks or task queue (Celery, RQ)|
|In-memory storage	|Redis / database|
|Single process	|Multi-worker (gunicorn / uvicorn workers)|
|Manual state	|State machine / enums|
|Blocking Whisper	|Dedicated worker service|

#### Better async approach (FastAPI-native)

Instead of threads:
```py
from fastapi import BackgroundTasks

@app.post("/transcribe/{user_id}")
async def transcribe(..., background_tasks: BackgroundTasks):
    background_tasks.add_task(transcriber.transcribe, filename)
```

#### Production-level enhancements

- Add timeouts
- Add file size limits
- Add authentication
- Use UUIDs instead of int IDs
- Store results in database
- Use logging instead of prints
- Add rate limiting
- Dockerize the app
- Add unit tests

#### Project: accurency apdate

##### 1
```Bash
pip install ctranslate2
pip install faster-whisper
```
```py
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8_float16")
segments, info = model.transcribe("audio.wav")
```
##### 2

Example of preprocessing (Python + ffmpeg)
```py
import subprocess
cmd = [
    "ffmpeg",
    "-i", "input.wav",
    "-ac", "1",  # Mono
    "-ar", "16000",  # 16kHz
    "-af", "loudnorm",  # Normalize volume
    "processed.wav"
]
subprocess.run(cmd)
```
You can also add noise reduction steps with specialized libs (rnnoise, noisered etc.).

##### 3
```py
import torch
from silero_vad import VADIterator

audio = load_audio("processed.wav")  # 16k, mono
vad = VADIterator(torch.tensor(audio), model, threshold=0.7)

segments = []
for speech_frame in vad:
    if speech_frame: segments.append(speech_frame)
```

##### 4
```json
[
  {"speaker": "A", "text": "Hello! How are you?"},
  {"speaker": "B", "text": "I’m good, thank you."}
]
```
##### 5 
```py
segments, info = model.transcribe(
    "audio.wav",
    beam_size=5,
    temperature=[0.0, 0.2, 0.4],
    vad_filter=True,   # if using faster-whisper built-in VAD
)
```
##### 6 
- Option A (lightweight, acceptable)
    - Use pause lengths + VAD gaps
    - LLM guesses speaker turns conservatively
    - Works OK for:
        - interviews
        - meetings with clear turn-taking

- Option B (better, still feasible)
    - Use a diarization model before ASR:
        - pyannote.audio (CPU works, slowish)
        - Output: (start, end, speaker_id)
    - Then:
        - Map ASR segments → speaker_id
        - LLM just formats, not guesses

### Text analusis

#### Road map
- Phase 1 — MVP
    - Choose LLaMA-3-8B
    - Build strict JSON schema
    - One-stage extraction
    - Manual ICD verification
- Phase 2 — Production
    - Split extraction & ICD mapping
    - Add validation layer
    - Add logging & traceability
    - Test with real transcripts
- Phase 3 — Medical grade
    - Fine-tune on real anonymized data
    - Add confidence flags
    - Add audit trails
    - Regulatory alignment
- Global
    - design exact prompt templates
    - design ICD specificity checker
    - write FastAPI endpoints
    - or build fine-tuning dataset structure
  
#### Extra notes
prompt = TEMPLATE.replace("{TRANSCRIPT_HERE}", transcript)

#### How to enforce exact fields?
- Method 1 — Prompt schema (basic)  -- Provide exact JSON structure in prompt.
- Method 2 — Pydantic (BEST):
    ```py
    from pydantic import BaseModel

    class Diagnosis(BaseModel):
        text: str
        icd_code: str
    ```
    - If LLM deviates → validation error.

#### SQL table
```py
import sqlite3

conn = sqlite3.connect("icd.db")
cursor = conn.cursor()

with open("ICD_schema.sql") as f:
    cursor.executescript(f.read())

with open("ICD_data.sql") as f:
    cursor.executescript(f.read())

conn.commit()
```


#### validation module #1
```py
class ICDValidator:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def find(self, text):
        self.cursor.execute("""
            SELECT code, name, node_count
            FROM class_mkb
            WHERE name LIKE ?
        """, (f"%{text}%",))
        return self.cursor.fetchall()

    def validate(self, diagnosis, proposed):
        candidates = []
        for word in diagnosis.split():
            candidates += self.find(word)

        if not candidates:
            return proposed

        best = max(candidates, key=lambda x: x[2])
        return best[0]

```


#### validation module #2

```py
import sqlite3

conn = sqlite3.connect("icd.db")
cursor = conn.cursor()

def find_icd_candidates(keyword):
    cursor.execute("""
        SELECT code, name, node_count
        FROM class_mkb
        WHERE name LIKE ?
    """, (f"%{keyword}%",))
    return cursor.fetchall()

def choose_most_specific(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[2])  # node_count

def validate_icd(diagnosis_text, proposed_code):
    keywords = diagnosis_text.split()

    candidates = []
    for kw in keywords:
        candidates.extend(find_icd_candidates(kw))

    best = choose_most_specific(candidates)

    if not best:
        return proposed_code  # fallback

    best_code, best_name, _ = best

    if best_code != proposed_code:
        return best_code

    return proposed_code

