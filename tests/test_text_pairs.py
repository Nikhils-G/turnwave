from turnwave.data.text_pairs import iter_examples, normalize_asr

DIALOGUES = [
    ["Hi , how are you ?", "I ’ m fine , thank you very much !"],
    ["Do you want to order a pizza tonight ?", "Yes , a large pepperoni please ."],
]


def test_normalize_asr():
    assert normalize_asr("I ’ m Fine , thank YOU .") == "i'm fine thank you"
    assert normalize_asr("Well ... it ' s  complicated!") == "well it's complicated"


def test_examples_deterministic():
    a = list(iter_examples(DIALOGUES, seed=42))
    b = list(iter_examples(DIALOGUES, seed=42))
    assert a == b
    c = list(iter_examples(DIALOGUES, seed=43))
    assert a != c  # different seed -> different truncation points


def test_negatives_are_proper_prefixes():
    examples = list(iter_examples(DIALOGUES, seed=0, negatives_per_positive=2))
    positives = {(e["context"], e["text"]) for e in examples if e["label"] == 1}
    negatives = [e for e in examples if e["label"] == 0]
    assert positives and negatives
    for neg in negatives:
        # each negative is a strict prefix of the positive sharing its context
        full = next(text for ctx, text in positives if ctx == neg["context"]
                    and text.startswith(neg["text"] + " "))
        assert len(neg["text"].split()) < len(full.split())


def test_context_is_previous_turn():
    examples = list(iter_examples(DIALOGUES, seed=0))
    first_dialogue = [e for e in examples if e["label"] == 1][:2]
    assert first_dialogue[0]["context"] == ""
    assert first_dialogue[1]["context"] == first_dialogue[0]["text"]
