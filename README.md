# TurnWave

**End-of-turn detection for voice agents, trained from scratch.**

Voice agents today decide "has the caller finished speaking?" with a fixed silence
timeout (300–700 ms). Too short and the agent interrupts mid-thought; too long and
every exchange feels laggy. Production stacks are replacing that timeout with
learned turn-taking models — but the open references each cover only half the
signal: [LiveKit's turn-detector](https://huggingface.co/livekit/turn-detector)
is text-only (semantic completeness), Pipecat's smart-turn is audio-only
(prosody). TurnWave trains **both branches from scratch and fuses them**.

```
audio (last ~2s) ──▶ log-mel spectrogram ──▶ CNN encoder ─────┐
                                                              ├─▶ fusion ─▶ P(turn complete)
transcript tail ──▶ BPE ──▶ causal transformer (from scratch) ┘      <50 ms CPU inference
```

Every layer is hand-written in PyTorch — RoPE, RMSNorm, SwiGLU, causal attention,
the training loop, the metrics. No HF `Trainer`, no pretrained weights.

## Status

- [x] **Phase 0 — scaffold**: uv project, package layout, data builders
- [x] **Phase 1 — text branch**: BPE tokenizer + ~7M-param causal transformer over
      the transcript tail (with previous-turn context), trained on DailyDialog
      complete-vs-truncated pairs
- [ ] **Phase 2 — acoustic branch**: CNN on log-mel spectrograms (AMI corpus +
      synthetic prosody data via Sarvam TTS)
- [ ] **Phase 3 — fusion + deployment**: ablation (text / audio / fused), ONNX +
      INT8 export, streaming inference under 50 ms on CPU
- [ ] **Phase 4 — production benchmark**: interruption-rate vs response-latency
      curves against fixed timeouts, LiveKit turn-detector, and smart-turn;
      live LiveKit agent demo
- [ ] **Phase 5 — Indic multilingual**: Hindi + more via Sarvam-generated data

## Quickstart

```bash
uv sync
uv run python scripts/build_text_dataset.py --out data/text
uv run python -m turnwave.tokenizer data/text/corpus.txt checkpoints/tokenizer
uv run python -m turnwave.train \
    --train data/text/train.jsonl --val data/text/validation.jsonl \
    --tokenizer checkpoints/tokenizer/spm.model --out checkpoints/text_eot
uv run python -m turnwave.evaluate \
    --ckpt checkpoints/text_eot/best.pt \
    --tokenizer checkpoints/tokenizer/spm.model --data data/text/test.jsonl
```

CPU-only machines: add `--limit 20000 --steps 300` to `turnwave.train` for a smoke
run; real training is a few GPU-hours on a free Colab T4
(`notebooks/colab_train.ipynb`).

Try it:

```bash
uv run python -m turnwave.predict "i want a large pepperoni and" \
    --context "what would you like to order" \
    --ckpt checkpoints/text_eot/best.pt --tokenizer checkpoints/tokenizer/spm.model
```

## Design decisions

- **ASR-normalized text.** Training text is lowercased and stripped of
  punctuation because that is what streaming ASR emits. Leaving punctuation in
  lets the model key on terminal periods it will never see in production.
- **Truncation negatives.** Negatives are real utterances cut at random word
  boundaries. Some cuts are accidentally complete phrases — irreducible label
  noise for a text-only model, and precisely the ambiguity the acoustic branch
  (falling pitch, trailing energy) resolves in Phase 3.
- **Classification at the last token.** Right padding + causal mask means pads
  can never influence real tokens, so the last real token's hidden state is a
  clean sequence summary (verified by `tests/test_model.py`).
- **Dialogue-level splits.** Train/val/test split by dialogue, following
  DailyDialog's own splits — no utterance appears in two splits.

## Tests

```bash
uv run pytest
```

Covers: dataset determinism and truncation invariants, model shapes, a causality
check (future tokens cannot affect past hidden states), padding leak check, and
a CPU training smoke test asserting the loss actually falls.
