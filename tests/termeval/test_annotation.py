from scripts.termeval.annotate_sample import round_robin_sample, spans_for
import random


def test_candidate_spans_are_normalized_token_indices():
    tokens, spans = spans_for("Python का code और फाइल")
    assert tokens == ["python", "का", "code", "और", "फाइल"]
    assert {tuple(s[k] for k in ("start", "end", "label")) for s in spans} >= {
        (0, 1, "TERM"), (2, 3, "TERM"), (4, 5, "EN_DEV")
    }


def test_speaker_round_robin():
    rows = [{"speaker": speaker, "utt_id": f"{speaker}{i}"} for speaker in "ABC" for i in range(4)]
    picked = round_robin_sample(rows, 6, random.Random(1))
    assert {speaker: sum(r["speaker"] == speaker for r in picked) for speaker in "ABC"} == {
        "A": 2, "B": 2, "C": 2
    }
