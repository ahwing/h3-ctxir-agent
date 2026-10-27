# Speech-to-text for the agent's media probe and for the judge's dialogue check.
# Usage: <python with faster-whisper> src/asr.py file.(wav|mp3|mp4) [--model small]
# Prints ONE JSON line: {"language": "zh", "language_prob": 0.98, "segments": [{"start":0.0,"end":1.2,"text":"..","no_speech":0.01}]}
# Decodes with ffmpeg to 16 kHz mono PCM itself (avoids PyAV issues). CPU int8.
import json
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel


def load(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main():
    args = sys.argv[1:]
    model = "small"
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
    audio = load(args[0])
    if audio.size < 1600:
        print(json.dumps({"language": None, "language_prob": 0, "segments": []}))
        return
    m = WhisperModel(model, device="cpu", compute_type="int8")
    try:
        segs, info = m.transcribe(audio, beam_size=5, vad_filter=True)
        segs = list(segs)
    except ValueError:  # VAD kept nothing (music / ambience only): language detection has no input
        print(json.dumps({"language": None, "language_prob": 0, "segments": []}))
        return
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "no_speech": round(s.no_speech_prob, 3)} for s in segs]
    print(json.dumps({"language": info.language, "language_prob": round(info.language_probability, 3), "segments": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
