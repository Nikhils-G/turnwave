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

from turnwave.data.eot_audio import DEFAULT_CUT_OFFSET, iter_cuts, slice_tail
from turnwave.data.features import fit_window
from turnwave.data.smart_turn import DATASET as SMART_TURN_DATASET
from turnwave.data.smart_turn import TEST_DATASET as SMART_TURN_TEST
from turnwave.data.smart_turn import iter_clips, load_transcripts
from turnwave.data.features import LogMel, MelConfig

DEFAULT_DATASET = "Scicom-intl/semantic-vad-eot"

SOURCES = ("semantic-vad", "smart-turn")


def build_split(dataset: str, config: str, split: str, out_dir: Path, max_examples: int,
                mel: LogMel, cut_offset: float = DEFAULT_CUT_OFFSET,
                source: str = "semantic-vad", languages: set[str] | None = None,
                real_only: bool = False, quiet: bool = False,
                split_name: str | None = None, skip: int = 0,
                transcripts: dict[str, str] | None = None) -> dict:
    cfg = mel.cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    # The output name is ours; `split` is whatever the upstream repo calls it.
    name = split_name or split
    feature_path = out_dir / f"{name}.f16.npy"
    meta_path = out_dir / f"{name}.jsonl"

    # Preallocated because a streamed corpus has no length until it is consumed;
    # the manifest records how much of the file is real.
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16,
        shape=(max_examples, cfg.n_mels, cfg.n_frames),
    )

    ds = (load_dataset(dataset, split=split, streaming=True) if config is None
          else load_dataset(dataset, config, split=split, streaming=True))
    ds = ds.cast_column("audio", Audio(decode=False))

    written = 0
    rows_used = 0
    positives = 0
    progress = tqdm(total=max_examples, desc=f"{name:10s}", disable=quiet, unit="ex")
    skipped = 0
    with open(meta_path, "w") as meta_file:
        for row in ds:
            if written >= max_examples:
                break
            if skipped < skip:
                # smart-turn ships one eval repo, so validation and test are carved
                # from it by offset. Without this they would be the same rows.
                skipped += 1
                continue
            items = (list(iter_cuts(row, cut_offset_seconds=cut_offset))
                     if source == "semantic-vad"
                     else list(iter_clips(row, languages=languages, real_only=real_only,
                                          transcripts=transcripts)))
            if not items:
                continue
            try:
                wav, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            except Exception:
                continue  # a handful of clips fail to decode; skip rather than abort
            if sample_rate != cfg.sample_rate:
                continue
            waveform = torch.from_numpy(wav)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            rows_used += 1
            for item in items:
                if written >= max_examples:
                    break
                if source == "semantic-vad":
                    window = slice_tail(waveform, item.cut_seconds, cfg.sample_rate,
                                        cfg.n_samples)
                    extra = {"cut": round(item.cut_seconds, 3)}
                else:
                    # smart-turn clips already end at the decision point, so the
                    # window is just the tail -- no cut logic, no offset.
                    window = fit_window(waveform, cfg.n_samples)
                    extra = {"language": item.language, "synthetic": item.synthetic,
                             "source": item.source}
                features[written] = mel(window).numpy().astype(np.float16)
                meta_file.write(json.dumps({
                    "i": written, "text": item.text, "label": item.label,
                    "id": getattr(item, "clip_id", None) or row.get("id"), **extra,
                }) + "\n")
                positives += item.label
                written += 1
                progress.update(1)
    progress.close()
    features.flush()

    return {
        "split": name, "examples": written, "rows_used": rows_used,
        "positives": positives, "negatives": written - positives,
        "positive_rate": round(positives / written, 4) if written else 0.0,
        "features": feature_path.name, "meta": meta_path.name,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/audio"))
    ap.add_argument("--source", choices=SOURCES, default="semantic-vad",
                    help="smart-turn: conversational clips with endpoint_bool labels "
                         "and no transcripts (audio branch only)")
    ap.add_argument("--dataset", default=None, help="overrides the source default")
    ap.add_argument("--config", default=None)
    ap.add_argument("--languages", nargs="*", default=None,
                    help="smart-turn: ISO-639-3 filter, e.g. eng. Default keeps all.")
    ap.add_argument("--real-only", action="store_true",
                    help="smart-turn: drop synthetic (TTS) clips")
    ap.add_argument("--hf-split", default=None,
                    help="smart-turn: upstream split name when it is not 'train' — "
                         "e.g. 'eng' for the English derivative, whose languages "
                         "are splits rather than configs")
    ap.add_argument("--transcripts", type=Path, default=None,
                    help="smart-turn: JSONL from scripts/transcribe_clips.py; "
                         "fills each clip's text so fusion can train")
    ap.add_argument("--max-examples", type=int, default=60000, help="cap for the train split")
    ap.add_argument("--max-eval-examples", type=int, default=6000)
    ap.add_argument("--cut-offset", type=float, default=DEFAULT_CUT_OFFSET,
                    help="seconds into each pause to place the decision point; "
                         "0.2 matches eot-bench and real endpointing")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    mel = LogMel(MelConfig())
    languages = set(args.languages) if args.languages else None
    transcripts = load_transcripts(args.transcripts) if args.transcripts else None
    if transcripts is not None:
        print(f"transcripts: {len(transcripts):,} clips covered")
    summaries = []
    for split in args.splits:
        cap = args.max_examples if split == "train" else args.max_eval_examples
        skip = 0
        if args.source == "smart-turn":
            # The upstream ships train and test as separate repos, each with a
            # split literally named "train". Validation and test are both carved
            # from the eval repo, test offset past validation so they never overlap.
            dataset = args.dataset or (SMART_TURN_DATASET if split == "train"
                                       else SMART_TURN_TEST)
            config, hf_split = args.config, (args.hf_split or "train")
            if split == "test":
                skip = args.max_eval_examples
        else:
            dataset = args.dataset or DEFAULT_DATASET
            config, hf_split = args.config or "en", split
        summaries.append(build_split(dataset, config, hf_split, args.out, cap, mel,
                                     cut_offset=args.cut_offset, source=args.source,
                                     languages=languages, real_only=args.real_only,
                                     quiet=args.quiet, split_name=split, skip=skip,
                                     transcripts=transcripts))

    manifest = {
        "source": args.source, "dataset": args.dataset, "config": args.config,
        "languages": sorted(languages) if languages else None,
        "real_only": args.real_only,
        "transcripts": str(args.transcripts) if args.transcripts else None,
        "cut_offset_seconds": args.cut_offset if args.source == "semantic-vad" else None,
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
