from whisper_transcriber import transcribe_whisper_segments
# from vosk_transcriber import transcribe_vosk_segments

from cleanup import cleanup_bisic_wrapper
from segmentation import segmentation_wrapper

print("FUCK ME")

file="clien_example.m4a"
file=cleanup_bisic_wrapper(file)
sg=segmentation_wrapper(file)

# res1=transcribe_vosk_segments(file,sg)
res2=transcribe_whisper_segments(file,sg)

# print(res1)
print("\n\n\n\n")
print(res2)