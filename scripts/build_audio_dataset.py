"""Stream an EOT corpus and precompute the log-mel feature cache.

    python scripts/build_audio_dataset.py --out data/audio --max-examples 60000

Features are computed once and stored as a float16 memmap; training then reads
tensors and never decodes audio again. Without this a Colab session spends its
whole life re-decoding the same FLAC files.

Audio is decoded with soundfile from the raw bytes rather than through the
`datasets` Audio decoder, which now requires torchcodec — an awkward dependency
to pin against a CPU-only torch build, and unnecessary when every clip here is
already 16 kHz mono.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from turnwave.data.eot_audio import iter_cuts, slice_tail
from turnwave.data.features import LogMel, MelConfig

DEFAULT_DATASET = "Scicom-intl/semantic-vad-eot"


def build_split(dataset: str, config: str, split: str, out_dir: Path, max_examples: int,
                mel: LogMel, quiet: bool = False) -> dict:
    cfg = mel.cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"{split}.f16.npy"
    meta_path = out_dir / f"{split}.jsonl"

    # Preallocated because a streamed corpus has no length until it is consumed;
    # the manifest records how much of the file is real.
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16,
        shape=(max_examples, cfg.n_mels, cfg.n_frames),
    )

    ds = load_dataset(dataset, config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    written = 0
    rows_used = 0
    positives = 0
    progress = tqdm(total=max_examples, desc=f"{split:10s}", disable=quiet, unit="ex")
    with open(meta_path, "w") as meta_file:
        for row in ds:
            if written >= max_examples:
                break
            cuts = list(iter_cuts(row))
            if not cuts:
                continue
            try:
                wav, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            except Exception:
                continue  # a handful of clips fail to decode; skip rather than abort
            if sample_rate != cfg.sample_rate:
                continue
            waveform = torch.from_numpy(wav)
            rows_used += 1
            for cut in cuts:
                if written >= max_examples:
                    break
                tail = slice_tail(waveform, cut.cut_seconds, cfg.sample_rate, cfg.n_samples)
                features[written] = mel(tail).numpy().astype(np.float16)
                meta_file.write(json.dumps({
                    "i": written, "text": cut.text, "label": cut.label,
                    "id": row.get("id"), "cut": round(cut.cut_seconds, 3),
                }) + "\n")
                positives += cut.label
                written += 1
                progress.update(1)
    progress.close()
    features.flush()

    return {
        "split": split, "examples": written, "rows_used": rows_used,
        "positives": positives, "negatives": written - positives,
        "positive_rate": round(positives / written, 4) if written else 0.0,
        "features": feature_path.name, "meta": meta_path.name,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/audio"))
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--config", default="en")
    ap.add_argument("--max-examples", type=int, default=60000, help="cap for the train split")
    ap.add_argument("--max-eval-examples", type=int, default=6000)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    mel = LogMel(MelConfig())
    summaries = []
    for split in args.splits:
        cap = args.max_examples if split == "train" else args.max_eval_examples
        summaries.append(build_split(args.dataset, args.config, split, args.out, cap,
                                     mel, quiet=args.quiet))

    manifest = {
        "dataset": args.dataset, "config": args.config,
        "mel": {k: getattr(mel.cfg, k) for k in
                ("sample_rate", "n_fft", "hop_length", "n_mels", "f_min", "f_max",
                 "window_seconds", "log_floor")},
        "shape": [mel.cfg.n_mels, mel.cfg.n_frames],
        "splits": summaries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for s in summaries:
        print(f"{s['split']:12s} {s['examples']:7,d} examples from {s['rows_used']:6,d} rows "
              f"({s['positives']:,} complete / {s['negatives']:,} mid-turn, "
              f"{s['positive_rate']:.1%} positive)")
    print(f"manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
