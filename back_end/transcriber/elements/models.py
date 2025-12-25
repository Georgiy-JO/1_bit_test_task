import torch
from speechbrain.pretrained import SpeakerRecognition

# ---- Load Silero VAD ----
print("[INIT] Loading Silero VAD model...")
silero_model, silero_utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', trust_repo=True, source='github'
)
get_speech_timestamps = silero_utils['get_speech_timestamps']
collect_chunks = silero_utils['collect_chunks']

# ---- Load SpeechBrain Speaker Encoder ----
print("[INIT] Loading SpeechBrain speaker encoder...")
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    speaker_model.embedding_model.to("cuda")
    speaker_model.model.to("cuda")  # speaker_model.mods.to("cuda") 

# def encode_segment(seg):
#     seg = seg.to(device)
#     emb = speaker_model.encode_batch(seg.unsqueeze(0))
#     return emb.squeeze().to("cpu").numpy()

# ---- EXPORT MODELS ----
__all__ = [
    "silero_model",
    "get_speech_timestamps",
    "collect_chunks",
    "speaker_model",
    "device",
]


