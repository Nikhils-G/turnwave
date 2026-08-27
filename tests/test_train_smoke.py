"""One CPU training loop on a tiny separable synthetic task: loss must fall."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from turnwave.data.loader import EOTTextDataset, make_collate
from turnwave.models.text_transformer import TextEOTConfig, TextEOTModel
from turnwave.train import configure_optimizer

VOCAB = 32


class FakeTokenizer:
    """Maps 't<N>' words to token id N, so tests need no sentencepiece training."""

    pad_id = 0
    sep_id = 1

    def encode_example(self, context, text, max_len=16):
        ids = [self.sep_id] + [int(w[1:]) for w in text.split()]
        return ids[-max_len:]


def _write_dataset(path, n_rows: int, seed: int):
    # label = 1 iff the last token id is in the top half of the vocab
    gen = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for _ in range(n_rows):
            length = int(torch.randint(2, 8, (1,), generator=gen))
            ids = torch.randint(2, VOCAB, (length,), generator=gen).tolist()
            label = 1 if ids[-1] >= VOCAB // 2 else 0
            text = " ".join(f"t{i}" for i in ids)
            f.write('{"context": "", "text": "%s", "label": %d}\n' % (text, label))


def test_loss_decreases(tmp_path):
    torch.manual_seed(0)
    data_path = tmp_path / "train.jsonl"
    _write_dataset(data_path, 512, seed=0)

    ds = EOTTextDataset(data_path, FakeTokenizer(), max_len=16)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True,
                        collate_fn=make_collate(FakeTokenizer.pad_id))
    cfg = TextEOTConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4,
                        max_seq_len=16, dropout=0.0)
    model = TextEOTModel(cfg)
    optimizer = configure_optimizer(model, lr=1e-3, weight_decay=0.0)

    losses = []
    for _ in range(15):
        for idx, lengths, y in loader:
            loss = F.binary_cross_entropy_with_logits(model(idx, lengths), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    first, last = sum(losses[:8]) / 8, sum(losses[-8:]) / 8
    assert last < first * 0.6, f"loss did not fall: {first:.3f} -> {last:.3f}"
