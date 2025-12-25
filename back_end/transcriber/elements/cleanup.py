import subprocess
import os
# import service    #for windows

def cleanup_denoise(input_file, output_file): # Noise cancel + normalization
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
    # print(f"Noise-cleaned file saved to {output_file}")


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
    # print(f"Basic-cleaned file saved to {output_file}")

def cleanup_bisic_wrapper(input_file):
    new_path=f"{os.path.splitext(input_file)[0]}_new_{os.path.splitext(input_file)[1]}"
    cleanup_basic(input_file,new_path)
    os.remove(input_file)
    return new_path
# cleanup_basic("/home/jack_oneill/Programming/1bit/test_data/call_conv_0.mp3", "2.mp3")