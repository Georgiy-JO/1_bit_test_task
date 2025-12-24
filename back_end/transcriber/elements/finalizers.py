from transformers import pipeline


def merge_transcripts(vosk_segments, whisper_segments):
    merged = []
    for a, b in zip(vosk_segments, whisper_segments):
        # Example: prefer Whisper if text non-empty
        text = b['text'] if len(b['text'].strip()) > 0 else a['text']
        merged.append({
            "speaker": a['speaker'],
            "start": a['start'],
            "end": a['end'],
            "text": text
        })
    return merged

merged_segments = merge_transcripts(vosk_transcripts, whisper_transcripts)




# Small local LLM for text editing
llm = pipeline("text2text-generation", model="tiiuae/falcon-7b-instruct", device_map="cpu")  # Example

def polish_segments_with_llm(segments):
    polished = []
    for seg in segments:
        prompt = f"""
        Fix grammar, punctuation, capitalization for this text.
        Do not add content. Keep meaning.
        Speaker: {seg['speaker']}
        Text: {seg['text']}
        """
        result = llm(prompt, max_length=512)[0]['generated_text']
        polished.append({
            "speaker": seg['speaker'],
            "start": seg['start'],
            "end": seg['end'],
            "text": result.strip()
        })
    return polished

polished_segments = polish_segments_with_llm(merged_segments)

# Save to JSON
import json
with open("final_transcript.json", "w", encoding="utf-8") as f:
    json.dump(polished_segments, f, ensure_ascii=False, indent=2)

print("✅ Final JSON saved as final_transcript.json")
