import torch
from settings import SPKREC_DIR
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer


# ---- Load Silero VAD ----
print("[INIT] Loading Silero VAD model...")
silero_model, silero_utils = torch.hub.load(
     repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, source='github'
)
get_speech_timestamps=silero_utils[0]
# collect_chunks = silero_utils["collect_chunks"]


"""""
from torchaudio.pipelines import SPEAKER_RECOGNITION_XVEC_TDNN

# ---- Load SpeechBrain Speaker Encoder ----
print("[INIT] Loading SpeechBrain speaker encoder...")
SPKREC_DIR_new = os.path.abspath(os.path.join(SPKREC_DIR, "new"))
speaker_model = SpeakerRecognition.from_hparams(
    source=SPKREC_DIR,
    savedir=SPKREC_DIR_new,
    skip_download=True  # for Windows to not try to create symlinks and download anything
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    speaker_model.embedding_model.to("cuda")
    speaker_model.model.to("cuda")  # speaker_model.mods.to("cuda") 
"""""
"""
from speechbrain.pretrained import SpeakerRecognition

# ---- Load TorchAudio X-Vector Speaker Encoder ----
print("[INIT] Loading TorchAudio X-Vector speaker encoder...")
bundle = SPEAKER_RECOGNITION_XVEC_TDNN
speaker_model = bundle.get_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = speaker_model.to(device)

def encode_segment(seg):
    seg = seg.to(device)
    if len(seg.shape) == 1:  # ensure batch dimension
        seg = seg.unsqueeze(0)
    with torch.inference_mode():
        emb = speaker_model(seg)
    return emb.squeeze(0).cpu().numpy()
"""


device = "cuda" if torch.cuda.is_available() else "cpu"
ctp="float16" if device =='cuda' else "int8"#"int8_float16"
    
whisper_model = WhisperModel("medium", device=device, compute_type=ctp)
# vosk_model = Model("vosk-model-small-ru")

# ---- EXPORT MODELS ----
__all__ = [
    "silero_model",
    "get_speech_timestamps",
    "collect_chunks",
    "speaker_model",
    "device",
    "whisper_model",
    "vosk_model",
]


