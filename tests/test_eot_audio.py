import pytest

from turnwave.data.eot_audio import Cut, iter_cuts, words_before

WORDS = [
    {"word": "lots", "start": 0.11, "end": 0.70},
    {"word": "of", "start": 0.70, "end": 0.91},
    {"word": "good", "start": 0.91, "end": 1.09},
    {"word": "ideas", "start": 1.09, "end": 1.60},
    {"word": "here", "start": 1.60, "end": 2.00},
    {"word": "and", "start": 2.40, "end": 2.70},
    {"word": "they", "start": 2.70, "end": 3.00},
    {"word": "work", "start": 3.00, "end": 5.00},
    {"word": "really", "start": 5.80, "end": 6.40},
    {"word": "well", "start": 6.70, "end": 7.10},
    {"word": "thanks", "start": 7.50, "end": 9.20},
]

ROW = {
    "duration": 9.81,
    "silence_spans": [
        {"start": 2.02, "end": 2.38},
        {"start": 5.04, "end": 5.73},
        {"start": 6.54, "end": 6.65},
        {"start": 7.18, "end": 7.43},
        {"start": 9.31, "end": 9.81},
    ],
    "words": WORDS,
}


def test_words_before_excludes_word_still_being_spoken():
    # "here" ends at 2.00 and is included; "and" starts at 2.40 and is not
    assert words_before(WORDS, 2.02) == "lots of good ideas here"
    # cutting mid-"work" (3.00-5.00) must not reveal it
    assert words_before(WORDS, 4.0) == "lots of good ideas here and they"


def test_words_before_is_asr_normalized():
    words = [{"word": "I'm", "start": 0.0, "end": 0.5}, {"word": "Fine.", "start": 0.5, "end": 1.0}]
    assert words_before(words, 2.0) == "i'm fine"


def test_one_row_yields_one_positive_and_the_rest_negative():
    cuts = list(iter_cuts(ROW))
    assert len(cuts) == 5
    assert [c.label for c in cuts] == [0, 0, 0, 0, 1]
    assert sum(c.is_final for c in cuts) == 1


def test_cuts_are_time_ordered_and_text_grows():
    cuts = list(iter_cuts(ROW))
    assert [c.cut_seconds for c in cuts] == sorted(c.cut_seconds for c in cuts)
    for earlier, later in zip(cuts, cuts[1:], strict=False):
        assert later.text.startswith(earlier.text)


def test_spans_are_sorted_before_labelling():
    """The final span by time is the positive even if the input order is scrambled."""
    shuffled = dict(ROW, silence_spans=list(reversed(ROW["silence_spans"])))
    cuts = list(iter_cuts(shuffled))
    assert cuts[-1].label == 1
    assert cuts[-1].cut_seconds == pytest.approx(9.31)


def test_row_with_a_single_pause_yields_only_a_positive():
    row = {"silence_spans": [{"start": 2.02, "end": 2.60}], "words": WORDS}
    cuts = list(iter_cuts(row))
    assert [c.label for c in cuts] == [1]


def test_short_final_silence_is_rejected():
    """A 0.05 s trailing gap is not evidence the turn ended."""
    row = dict(ROW, silence_spans=[{"start": 2.02, "end": 2.38}, {"start": 9.31, "end": 9.36}])
    assert list(iter_cuts(row)) == []


def test_overlong_pause_rejects_the_row():
    row = dict(ROW, silence_spans=[{"start": 2.02, "end": 8.5}, {"start": 9.31, "end": 9.81}])
    assert list(iter_cuts(row)) == []


def test_cut_before_any_word_is_skipped():
    """No transcript yet means nothing for the text branch to consume."""
    row = dict(ROW, silence_spans=[{"start": 0.05, "end": 0.30}, {"start": 9.31, "end": 9.81}])
    labels = [c.label for c in iter_cuts(row)]
    assert labels == [1]


def test_missing_or_empty_spans_yield_nothing():
    assert list(iter_cuts({"words": WORDS})) == []
    assert list(iter_cuts({"silence_spans": [], "words": WORDS})) == []


def test_cut_is_hashable_and_frozen():
    cut = Cut(1.0, "hi", 1, True)
    with pytest.raises(Exception):
        cut.label = 0  # type: ignore[misc]


def test_slice_tail_ends_at_the_cut():
    import torch

    from turnwave.data.eot_audio import slice_tail

    audio = torch.arange(16000, dtype=torch.float32)  # 1 s ramp at 16 kHz
    tail = slice_tail(audio, cut_seconds=0.5, sample_rate=16000, n_samples=1600)
    assert tail.shape == (1600,)
    assert tail[-1].item() == pytest.approx(7999.0)  # sample just before 0.5 s


def test_slice_tail_left_pads_when_cut_is_early():
    import torch

    from turnwave.data.eot_audio import slice_tail

    audio = torch.ones(16000)
    tail = slice_tail(audio, cut_seconds=0.05, sample_rate=16000, n_samples=1600)
    assert tail.shape == (1600,)
    assert tail[0].item() == 0.0 and tail[-1].item() == 1.0
