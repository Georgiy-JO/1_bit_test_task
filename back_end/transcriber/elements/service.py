import os
#for win -- Point to local ffmpeg.exe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "myenv", "ffmpeg", "bin", "ffmpeg.exe"))
if not os.path.exists(ffmpeg_path):
    raise RuntimeError(f"ffmpeg not found at {ffmpeg_path}")
os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")