"""smart-turn reader tests.

The corpus this phase moves to has a human-authored label, no transcripts, and a
`synthetic` flag covering 82% of it. After Phase 4 -- where an unexamined training
corpus turned out to be read speech and cost the whole result -- the flag being
carried through rather than dropped is itself a requirement worth testing.
"""

from turnwave.data.smart_turn import Clip, iter_clips


def row(**overrides) -> dict:
    base = {"endpoint_bool": True, "language": "eng", "synthetic": False,
            "dataset": "liva_1", "id": "abc", "spoken_text": None}
    base.update(overrides)
    return base


def only(r: dict, **kwargs) -> Clip | None:
    clips = list(iter_clips(r, **kwargs))
    return clips[0] if clips else None


def test_endpoint_bool_becomes_the_label():
    assert only(row(endpoint_bool=True)).label == 1
    assert only(row(endpoint_bool=False)).label == 0


def test_string_booleans_are_parsed():
    """Parquet columns arrive as real bools or as 'True'/'False' strings."""
    assert only(row(endpoint_bool="True")).label == 1
    assert only(row(endpoint_bool="false")).label == 0
    assert only(row(synthetic="True")).synthetic is True
    assert only(row(synthetic="False")).synthetic is False


def test_unlabelled_rows_are_dropped():
    """A missing label must not silently become a negative."""
    assert only(row(endpoint_bool=None)) is None
    assert only(row(endpoint_bool="maybe")) is None


def test_no_transcripts():
    """spoken_text is null across the corpus, so text is always empty and the
    fused model cannot train here."""
    assert only(row()).text == ""
    assert only(row(spoken_text="ignored if present")).text == ""


def test_synthetic_flag_is_preserved():
    assert only(row(synthetic=True)).synthetic is True
    assert only(row(synthetic=False)).synthetic is False


def test_real_only_filter():
    assert only(row(synthetic=True), real_only=True) is None
    assert only(row(synthetic=False), real_only=True) is not None


def test_language_filter_is_case_insensitive():
    assert only(row(language="ENG"), languages={"eng"}) is not None
    assert only(row(language="zho"), languages={"eng"}) is None
    assert only(row(language="zho")) is not None  # no filter keeps everything


def test_provenance_is_carried_through():
    clip = only(row(dataset="human_5", id="xyz", language="hin"))
    assert (clip.source, clip.clip_id, clip.language) == ("human_5", "xyz", "hin")


def test_missing_optional_fields_do_not_crash():
    clip = only({"endpoint_bool": True})
    assert clip.label == 1 and clip.language == "" and clip.source == ""


def test_clip_is_frozen():
    import pytest

    with pytest.raises(Exception):
        only(row()).label = 0  # type: ignore[misc]
