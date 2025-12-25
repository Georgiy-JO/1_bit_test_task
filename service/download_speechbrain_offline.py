import os
import shutil
from speechbrain.pretrained import SpeakerRecognition
from back_end.transcriber.elements.settings import SPKREC_DIR

os.makedirs(SPKREC_DIR, exist_ok=True)

print("[INFO] Downloading full SpeechBrain ECAPA-TDNN speaker model...")
# This downloads everything needed from HuggingFace to a local cache
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=SPKREC_DIR,
    skip_symlinks=True  # Windows: avoid symlink errors
)

print("[INFO] Model downloaded and ready offline.")

# Now check files in SPKREC_DIR
print("[INFO] Files in offline folder:")
for root, dirs, files in os.walk(SPKREC_DIR):
    for f in files:
        print(os.path.join(root, f))

# Optional: adjust hyperparams.yaml to point to local files
yaml_path = os.path.join(SPKREC_DIR, "hyperparams.yaml")
if os.path.exists(yaml_path):
    with open(yaml_path, "r") as f:
        content = f.read()

    content = content.replace("!ref pretrained_path/embedding_model.ckpt", "embedding_model.ckpt")
    content = content.replace("!ref pretrained_path/classifier.ckpt", "classifier.ckpt")
    content = content.replace("!ref pretrained_path/label_encoder.txt", "label_encoder.txt")

    with open(yaml_path, "w") as f:
        f.write(content)

print("[INFO] hyperparams.yaml updated for fully offline use.")
