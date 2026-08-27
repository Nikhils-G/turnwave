"""Quick single-utterance prediction, for demos and sanity checks.

    python -m turnwave.predict --ckpt checkpoints/text_eot/best.pt \
        --tokenizer checkpoints/tokenizer/spm.model \
        --context "what would you like to order" "i want a large pepperoni and"
"""

import argparse
from pathlib import Path

import torch

from .data.text_pairs import normalize_asr
from .evaluate import load_model
from .tokenizer import Tokenizer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="the (partial) utterance heard so far")
    ap.add_argument("--context", default="", help="previous turn in the conversation")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    args = ap.parse_args(argv)

    tok = Tokenizer(args.tokenizer)
    model = load_model(args.ckpt, torch.device("cpu"))
    ids = tok.encode_example(normalize_asr(args.context), normalize_asr(args.text),
                             model.cfg.max_seq_len)
    with torch.no_grad():
        logit = model(torch.tensor([ids]), torch.tensor([len(ids)]))
    p = torch.sigmoid(logit).item()
    verdict = "complete — respond" if p >= 0.5 else "incomplete — keep listening"
    print(f"P(turn complete) = {p:.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
