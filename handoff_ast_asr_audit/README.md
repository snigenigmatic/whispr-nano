# Handoff: Whisper-family Svarah audit → belongs in `ast-asr`, not here

These files were produced running the script in this same folder against
the `ast-asr` repo's own code (`ast_asr.taxonomy`, `ast_asr.metrics`) on
Modal. They're staged here only because this is the one repo this session
has write access to — **this whole folder should end up in `ast-asr`'s
`audit/` directory, not in whispr-nano.**

## To install into your `ast-asr` clone

```bash
cp handoff_ast_asr_audit/modal_whisper_family_audit.py  <ast-asr>/audit/
cp handoff_ast_asr_audit/results_*.csv                  <ast-asr>/audit/
cp handoff_ast_asr_audit/summary_*.json                  <ast-asr>/audit/
cd <ast-asr>
git add audit/
git commit -m "Add Whisper-family Svarah audit results (tiny/small/large-v3/distil-large-v3)"
git push
```

## What's in each summary JSON

| System | Overall WER | ΔDP | Poisson p | Verdict | Cost |
|---|---|---|---|---|---|
| whisper-tiny | 20.60% | 14.19pp | 3.3e-24 | structural | $0.034 |
| whisper-small | 10.88% | 3.18pp | 0.0045 | structural | $0.095 |
| whisper-large-v3 | 7.04% | 2.38pp | 0.0152 | structural | $0.411 |
| distil-large-v3 | 10.03% | 4.02pp | 0.00025 | structural | $0.285 |

Every system shows a statistically significant per-family gap on Svarah, and
Sino-Tibetan is the worst-performing family in all four. Each `results_*.csv`
has 2,500 per-utterance rows (utt_id, family, gender, reference/hypothesis
text raw + normalized via `ast_asr.metrics.normalize_for_wer`, WER, edit
counts). Total cost for this whole track: $0.836.

Script (`modal_whisper_family_audit.py`) is the exact one already validated
and described in `docs/TEAM_AGENT_TASKS_20260828.md` in this repo — running
it again for any of these four models should reproduce the same order of
magnitude.
