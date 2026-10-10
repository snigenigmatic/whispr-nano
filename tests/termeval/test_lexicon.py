from ast_asr.termeval.lexicon import approved_pairs, build_lexicon, phonetic_fold


def test_fold_and_candidate():
    assert phonetic_fold("PHAAIL") == "fail"
    rows = build_lexicon([{"utt_id": "u1", "text": "फाइल"}], {"u1": "fail"})
    assert rows[0]["n_occurrences"] == 1
    assert rows[0]["latin"] == "fail"
    assert rows[0]["devanagari"] == "फाइल"
    assert rows[0]["status"] in {"AUTO_ACCEPT", "REVIEW"}
    assert approved_pairs([{"latin": "x", "devanagari": "य", "status": "REVIEW"}]) == set()


def test_short_false_friend_is_not_auto_accepted():
    rows = build_lexicon([{"utt_id": "u1", "text": "इन"}, {"utt_id": "u2", "text": "इन"}],
                         {"u1": "in", "u2": "in"})
    assert rows[0]["status"] == "COLLISION"
