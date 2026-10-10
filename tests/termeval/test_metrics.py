from ast_asr.termeval.metrics import cluster_bootstrap, pooled_metrics, score_utterance


def test_reference_anchored_terms_and_classes():
    row = score_utterance(
        "u1", "s1", "फाइल को save करो", "file को save अब करो",
        spans=[{"start": 0, "end": 1, "label": "TERM"}, {"start": 2, "end": 3, "label": "TERM"}],
    )
    counts = pooled_metrics([row])
    assert counts["WER"] == (0.5, 2, 4)
    assert counts["TokER[DEV]"] == (1 / 3, 1, 3)
    assert counts["TermER"] == (0.5, 1, 2)
    assert {r["status"] for r in row.term_rows} == {"CORRECT", "SUBSTITUTED"}
    assert cluster_bootstrap([row], n_resamples=10)["WER"] == (0.5, 0.5)


def test_condition_c_recomputes_word_and_character_error():
    raw = score_utterance("u", "s", "फाइल", "file")
    equiv = score_utterance("u", "s", "फाइल", "file", equivalences={("file", "फाइल")})
    assert pooled_metrics([raw])["WER"][0] == 1.0
    assert pooled_metrics([equiv])["WER"][0] == 0.0
    assert pooled_metrics([equiv])["CER"][0] == 0.0


def test_internal_insertion_breaks_multiword_term_but_boundary_does_not():
    span = [{"start": 0, "end": 2, "label": "TERM"}]
    inside = score_utterance("u", "s", "python code", "python extra code", spans=span)
    boundary = score_utterance("u", "s", "python code", "extra python code", spans=span)
    assert inside.term_rows[0]["status"] == "PARTIAL"
    assert inside.term_rows[0]["internal_ins"] == 1
    assert boundary.term_rows[0]["status"] == "CORRECT"
    assert boundary.term_rows[0]["boundary_ins"] == 1
