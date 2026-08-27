"""BPE tokenizer for the text branch, trained on our own dialogue corpus.

We train sentencepiece BPE ourselves (rather than borrowing a pretrained vocab)
so the whole text stack — vocab included — is reproducible from the raw corpora.
"""

import argparse
from pathlib import Path

import sentencepiece as spm

SEP_PIECE = "<sep>"  # separates the previous turn (context) from the current partial utterance


def train_tokenizer(corpus_path: Path, out_dir: Path, vocab_size: int = 8192) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(out_dir / "spm"),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=[SEP_PIECE],
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )
    return out_dir / "spm.model"


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id: int = self.sp.pad_id()
        self.sep_id: int = self.sp.piece_to_id(SEP_PIECE)
        self.vocab_size: int = self.sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def encode_example(self, context: str, text: str, max_len: int = 128) -> list[int]:
        """`context <sep> text`, keeping the most recent max_len tokens.

        Truncating from the left preserves the tail of the utterance, which is
        where the end-of-turn signal lives.
        """
        ids: list[int] = []
        if context:
            ids.extend(self.sp.encode(context))
        ids.append(self.sep_id)
        ids.extend(self.sp.encode(text))
        return ids[-max_len:]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the BPE tokenizer on a corpus file (one utterance per line).")
    ap.add_argument("corpus", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--vocab-size", type=int, default=8192)
    args = ap.parse_args(argv)
    model_path = train_tokenizer(args.corpus, args.out_dir, args.vocab_size)
    tok = Tokenizer(model_path)
    print(f"trained {model_path} vocab={tok.vocab_size} pad_id={tok.pad_id} sep_id={tok.sep_id}")


if __name__ == "__main__":
    main()
