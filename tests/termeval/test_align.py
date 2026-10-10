import random

import jiwer

from ast_asr.termeval.align import align_words, edit_counts


def test_raw_alignment_matches_jiwer_on_20_fixed_random_pairs():
    rng = random.Random(20261011)
    vocabulary = ["हम", "python", "का", "code", "run", "फाइल", "save", "123"]
    for _ in range(20):
        ref = rng.choices(vocabulary, k=rng.randint(1, 12))
        hyp = rng.choices(vocabulary, k=rng.randint(0, 12))
        observed = edit_counts(align_words(ref, hyp))
        expected = jiwer.process_words(" ".join(ref), " ".join(hyp))
        assert (observed["SUB"], observed["DEL"], observed["INS"]) == (
            expected.substitutions, expected.deletions, expected.insertions
        )


def test_script_equivalence_realigns():
    assert edit_counts(align_words(["फाइल", "खोलें"], ["file", "खोलें"]))["SUB"] == 1
    assert [e.op for e in align_words(["फाइल", "खोलें"], ["file", "खोलें"], {("file", "फाइल")})] == [
        "EQ", "EQ"
    ]
