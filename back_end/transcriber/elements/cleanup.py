import subprocess
import service

def cleanup_denoise(input_file, output_file):
    """
    Noise cancel + normalization
    """
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-af", "arnndn=m=rnnoise_model.pth",  # RNNoise noise suppressor
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz
        "-y",
        output_file
    ]
    subprocess.run(cmd, check=True)
    print(f"Noise-cleaned file saved to {output_file}")


def cleanup_basic(input_file, output_file):
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        "-y",
        output_file
    ]
    subprocess.run(cmd, check=True)
    print(f"Basic-cleaned file saved to {output_file}")

cleanup_denoise("D:/od_ga/test_data/call_conv_0.mp3", "1.mp3")
cleanup_basic("D:/od_ga/test_data/call_conv_0.mp3", "2.mp3")
