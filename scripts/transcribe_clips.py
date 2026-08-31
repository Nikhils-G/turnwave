"""Transcribe smart-turn clips with Whisper, so fusion can train on them.

The conversational corpus carries no transcripts (`spoken_text` is null on every
row), which is why Phase 5 was audio-only. This script closes that gap: it streams
the corpus, runs faster-whisper over each clip, and writes `{id, text}` lines to a
JSONL the cache builder can consume via --transcripts.

Two properties matter more than speed:

* **Resumable.** Output is append-only, and on restart every id already present is
  skipped. Free-tier GPU sessions die; a transcription run must never be lost to
  one. (The same lesson training already learned via --resume.)
* **Measured before committed.** --measure transcribes a small sample and reports
  clips/minute, so the scale is chosen from evidence instead of a guessed timeline.

    python scripts/transcribe_clips.py --config eng --measure 200
    python scripts/transcribe_clips.py --config eng --max-clips 40000 \
        --out transcripts_train.jsonl
    python scripts/transcribe_clips.py \
        --dataset pipecat-ai/smart-turn-data-v3.2-test --languages eng \
        --max-clips 9000 --out transcripts_eval.jsonl
"""

import argparse
import io
import json
import time
from pathlib import Path

DEFAULT_DATASET = "giangndm/smart-turn-data-v3.1-en-vi"
DEFAULT_CONFIG = "eng"


def load_done_ids(path: Path) -> set[str]:
    """Ids already transcribed — the resume set. Tolerates a torn final line from
    a killed session rather than refusing to start."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue  # torn tail line from a dead session; it will be redone
    return done


def build_model(model_name: str, device: str):
    from faster_whisper import WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_name, device=device, compute_type=compute)


def transcribe_array(model, audio, sample_rate: int) -> str:
    """One clip -> plain text. Clips are short (~8 s), so VAD chunking is off —
    trailing silence is part of the signal here, not noise to strip."""
    import numpy as np

    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != 16000:
        return ""  # corpus is 16 kHz throughout; anything else is a bad row
    segments, _ = model.transcribe(wav, language="en", beam_size=1,
                                   vad_filter=False, without_timestamps=True)
    return " ".join(s.text.strip() for s in segments).strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("transcripts.jsonl"))
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--config", default=None,
                    help=f"HF config, e.g. {DEFAULT_CONFIG!r} for the English "
                         "derivative; omit for repos with a single default config")
    ap.add_argument("--languages", nargs="*", default=None,
                    help="ISO-639-3 filter, e.g. eng — skip other rows without "
                         "spending GPU on them")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-clips", type=int, default=40000)
    ap.add_argument("--measure", type=int, default=0,
                    help="transcribe only N clips and report the rate, writing nothing")
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args(argv)

    import soundfile as sf
    from datasets import Audio, load_dataset

    done = set() if args.measure else load_done_ids(args.out)
    if done:
        print(f"resuming: {len(done)} clips already transcribed in {args.out}")

    model = build_model(args.model, args.device)
    ds = (load_dataset(args.dataset, args.config, split=args.split, streaming=True)
          if args.config else
          load_dataset(args.dataset, split=args.split, streaming=True))
    ds = ds.cast_column("audio", Audio(decode=False))

    target = args.measure or args.max_clips
    written = empty = seen = 0
    start = time.monotonic()
    sink = open(args.out, "a") if not args.measure else None
    try:
        for row in ds:
            if written + len(done) >= target and not args.measure:
                break
            if args.measure and seen >= target:
                break
            if args.languages and (row.get("language") or "").lower() not in args.languages:
                continue
            clip_id = row.get("id") or ""
            if clip_id in done:
                continue
            seen += 1
            try:
                wav, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            except Exception:
                continue
            text = transcribe_array(model, wav, sample_rate)
            if not text:
                empty += 1
            if sink:
                sink.write(json.dumps({"id": clip_id, "text": text}) + "\n")
                written += 1
                if written % args.flush_every == 0:
                    sink.flush()
            if seen % 200 == 0:
                rate = seen / max(time.monotonic() - start, 1e-9) * 60
                print(f"  {seen} clips  ({rate:.0f}/min, {empty} empty)")
    finally:
        if sink:
            sink.close()

    elapsed = time.monotonic() - start
    rate = seen / max(elapsed, 1e-9) * 60
    print(f"done: {seen} transcribed in {elapsed/60:.1f} min = {rate:.0f} clips/min "
          f"({empty} empty)")
    if args.measure:
        for n in (20000, 40000, 65000):
            print(f"  projected {n:>6,} clips: {n / max(rate, 1e-9):.0f} min")
    else:
        print(f"total in {args.out}: {len(done) + written}")


if __name__ == "__main__":
    main()
