# Weekend Execution Plan — PW25_BJD_05 Pivot

Companion to [TEAM_BRIEF.md](TEAM_BRIEF.md). Every step is tagged **[HUMAN]** (judgment or access required) or **[AGENT]** (paste the brief into a Cursor agent and review its output).

**Goal by Sunday 22:00 IST:** the three memo decisions made with evidence, a first-cut audit table with ≥3 systems, Blockers A & B resolved, and a panel-ready deck.

---

## Phase 0 — Friday night, everyone, 45 minutes

> The memo already says it: *"Agree the CSV schema and naming before anyone runs anything, or we lose the last day to reconciling columns."*

**0.1 [HUMAN, 15 min] Freeze the schema.** Adopt (or amend tonight, then never again):

```
results_<system>_<condition>.csv     # one row per utterance
  utt_id, speaker_id, accent, family, condition{clean,noise10db},
  system, system_version, decode_config,
  ref_text_raw, hyp_text_raw, ref_text_norm, hyp_text_norm,
  wer, cer, n_sub, n_del, n_ins, ref_len_words

systems.yaml    # registry: system -> checkpoint/API version, decode params, run date
```

**0.2 [HUMAN, 10 min] Freeze decode + scoring rules** (this is what makes an audit an audit):

- Greedy / beam 1 everywhere. Temperature 0. No vendor "smart formatting."
- **Whisper `EnglishTextNormalizer` on both ref and hyp, for every system.** (Supersedes Phase 2's lowercase+strip; vendors return punctuation/numerals differently and raw edit distance would be unfair. Note the change once in the report.)
- Noise condition: generate the corrupted eval set **once** (white noise, 10 dB SNR, fixed seed), share the same 1,998 noisy wavs with everyone via Modal Volume or drive. Nobody generates their own noise.

**0.3 [HUMAN, 10 min] Access checks, live on the call:** Sarvam API key works (Adithya, one curl); Modal workspace access (Aditya, added by Kau); everyone can read `data/svarah_split/eval_uids.txt` from the ast-asr repo.

**0.4 [HUMAN, 10 min]** Confirm assignments below; book Sunday 20:00 IST sync.

---

## Kau — blockers, stats, deck (critical path)

### K1. Blocker A — GLM coefficient autopsy [AGENT tonight, YOU judge]

Agent brief:

> Using the existing per-utterance eval CSVs (`outputs/results_whisper-small_clean.csv`, `outputs/results_whisper-rl-fair_clean.csv`), refit the Poisson GLM from `compute_fairness_summary` (errors = subs+dels+ins, offset = log utterance length, family fixed effect) for both models. Produce per model: per-family coefficient, standard error, rate ratio vs. Dravidian baseline, 95% CI, n_utterances and n_speakers per family, plus pairwise contrasts (IA-Drav, ST-Drav, ST-IA). Write `outputs/blocker_a_coefficients.csv` and a readable printed summary. CPU only, statsmodels.

**Judgment call:** if the RL model's family effect now rests on ST/Bodo alone (large coefficient, large SE) while IA ≈ Dravidian, then Table 6.3's "gap became non-structural" is power loss + bias concentration → §6.3, §6.5, §7.1, abstract get rewritten. Either outcome supports the pivot; you just need to know which story the deck tells.

### K2. Blocker B — train/eval text overlap [AGENT tonight]

Agent brief:

> In the ast-asr repo: load Svarah, apply the committed split (`data/svarah_split/`). Normalize transcripts (lowercase, strip punctuation). Report: (1) eval utterances whose normalized transcript appears in train under a different speaker — count and % of 1,998; (2) near-duplicates (normalized edit distance ≤ 0.1); (3) overall and per-family WER for whisper-small zero-shot and whisper-small-rl recomputed on the non-overlapping eval subset only, from existing results CSVs. Write `outputs/blocker_b_overlap.csv`, print the WER deltas.

**Decision rule:** if the 2-point RL improvement shrinks materially on the non-overlap subset, drop the WER claim from the deck; the audit and GRL findings are untouched.

### K3. Mixed-effects upgrade [AGENT Sat, YOU validate Sat night]

Agent brief:

> Implement a mixed-effects Poisson model replacing our fixed-effects GLM: per-utterance error count, log-length offset, family fixed effect, speaker random intercept — the ASR-FAIRBENCH specification (Rai et al., Interspeech 2025, arXiv:2505.11572). Use statsmodels `PoissonBayesMixedGLM`, or rpy2 + lme4 `glmer` if unstable (pick one, justify). Validate on synthetic data with a known family effect (recover within CI). Deliver `fit_family_effect(csv_path) -> {coef table, LRT p}` and run it on all Phase 2 result CSVs. Include a pytest.

**You personally check the synthetic-recovery test** — this function becomes the load-bearing statistic of the audit.

### K4. §6.3 rewrite + merge Aditi's fixes [HUMAN, Sun]

### K5. The deck [AGENT drafts skeleton Sat, YOU own Sun] — 10 slides:

1. What Phase 2 proved (protocol worked; Table 6.1, one line)
2–3. What happened in August: Saaras V3 / ASR-FAIRBENCH / clinical audit, receipts quoted (from Aditi's pack)
4. The coverage table (everyone has columns, nobody has the row)
5. The proposal + **the first audit rows already running** (Aditya/Adithya's output — the single most persuasive slide)
6. What Phase 2 becomes: motivation with receipts, Blocker A/B outcomes stated honestly
7. Upgraded stats: mixed-effects, citing ASR-FAIRBENCH
8. Timeline, split, cost (≈ zero)
9. Risks + watch list (someone publishes per-accent Svarah first; mitigation: narrow scope, speed)
10. The ask: approve the pivot; ACL SRW paper unaffected; OPD story parked with a date (Interspeech 2027), already PoC-validated for $0.16

---

## Aditya — Whisper family on Modal, clean + noisy

**A1. [HUMAN, Fri, 30 min]** Get Modal workspace access; skim whispr-nano's `modal_app.py` — you're reusing its skeleton (cached image, HF-cache Volume, cost guardrail, T4).

**A2. [AGENT, Sat] Port the audit decode to Modal.** Agent brief:

> Create `modal_audit.py` in the ast-asr repo, modeled on whispr-nano's `modal_app.py` (same image/Volume/guardrail pattern). It must: load the Svarah eval split (1,998 committed uids), decode with a given HF checkpoint (greedy, temp 0), run clean and the shared 10 dB SNR noisy set from the Volume, apply Whisper `EnglishTextNormalizer` to ref+hyp, and emit `results_<system>_<condition>.csv` in the agreed schema exactly. Systems: openai/whisper-tiny, openai/whisper-small, openai/whisper-large-v3, distil-whisper/distil-large-v3. One T4, batch 8 (small models) / 4 (large-v3), pre-flight cost estimate before any GPU spend, hard timeout per system.

Budget (from whispr-nano's measured T4 throughput): large-v3 over ~2,000 utts × 2 conditions ≈ 1.5–3 T4-hours ≈ **$1–2**; smaller models are noise. Fits the Modal free tier.

**A3. [HUMAN, Sat night]** Eyeball 10 transcripts per system before full launch (garbage detection is a human job). Launch, verify the guardrail projections, walk away.

**A4. [HUMAN, Sun]** Run Kau's `fit_family_effect` on each CSV; post rows (system × condition × per-family WER × p) to the shared sheet by 18:00.

---

## Adithya — Saaras V3 API + Qwen3-ASR (+ designated fallback)

**AD1. [HUMAN, Fri, 30 min] The Saaras gate** — our biggest unknown, flagged in the memo. Send 5 Svarah clips through the API by hand. Inspect: casing, punctuation, numeral formatting ("twenty five" vs "25"), Devanagari/code-mix leakage, disfluency markers, response structure. Post the 5 raw outputs to the group. Whisper-normalizer fixes it → green light; structurally weird → write the extraction rule now.

**AD2. [AGENT, Sat] Saaras client.** Agent brief:

> Write `audit/saaras_client.py`: batch-transcribe wav paths via the Sarvam Saaras V3 API. Requirements: disk response-cache keyed by (utt_id, condition, api_version) so re-runs are free; exponential backoff on 429/5xx; strict transcript-text extraction; `--limit N` flag; per-call cost logging. Plus a runner emitting `results_saarasv3_<condition>.csv` in the agreed schema, using the shared noisy wavs and Whisper EnglishTextNormalizer.

**[HUMAN]** Check Sarvam's pricing and post projected cost for 1,998 × 2 calls to the group **before** the full run — the only real money risk of the weekend. Clean first, sanity-check 20 rows, then noisy.

**AD3. [HUMAN 20 min, then AGENT] Qwen3-ASR.** Confirm how it's served (HF weights vs. API) and that plain English transcription works on 3 clips. Weights → reuse Aditya's A2 brief with the Qwen checkpoint. API → clone the AD2 brief. Unusable within an evening → **invoke fallback**: take IndicWhisper / additional Whisper variants on Aditya's Modal skeleton instead. Rows beat completeness.

**AD4. [STRETCH, only if AD2–AD3 land by Sun noon]** The memo's 20-minute DiarBench check: prompt Gemma-4-E4B-it for speaker-labelled JSON on a few clips, temp 0. Park whatever comes out; it's a future-work bullet, nothing more.

---

## Aditi — assembly, report integrity, receipts

**AT1. [AGENT, Sat] Results assembler.** Agent brief:

> Write `audit/assemble.py`: glob `results_*_*.csv`, validate each against the agreed schema (hard-fail with a named-column error so drift is caught on landing, not Sunday night), emit (a) the master audit table — rows = system × condition; columns = overall WER, per-family WER, ΔDP, mixed-effects p — and (b) a markdown render for the deck. Include a `--strict` pytest with a fixture CSV.

Run it on every CSV the moment one lands. You are the schema police.

**AT2. [HUMAN + AGENT, Sat] The three confirmed report bugs:** missing HuBERT row in Table 6.1 (numbers exist in run logs; if not, write "run missing" honestly), §7.2's dangling half-sentence, §2.2's citation ("Radhakrishnan et al. [17]" → [17] is Unni et al.). Agent produces the diff; you approve.

**AT3. [HUMAN, Sun] Receipts pack:** screenshots/quotes with page refs — Sarvam blog (Svarah eval, underrepresentation rationale, five aggregate metrics, no subgroup table), ASR-FAIRBENCH's mixed-effects paragraph, the clinical audit's system list. One PDF to Kau for slides 2–3.

**AT4. [HUMAN, Sun] Cold-read the deck.** Every number must trace to a CSV in the repo.

---

## Sunday 20:00 IST — the three decisions

1. **Drop GRPO paper, run audit?** Decided by Blocker A+B outcomes + the audit rows already on screen.
2. **System split** — reconcile against what actually ran; formalize any fallback reassignment.
3. **Mixed-effects adoption** — Kau shows synthetic-recovery test + re-run Phase 2 numbers; adopt and cite.

Then: deck freezes Sunday night; Adithya books the BJD pre-meeting for early week. **BJD sees it before the panel** — he asked for this pivot; his edits predict the panel's objections.

## Contingencies

| If... | Then... |
|---|---|
| Blocker A ⇒ bias concentration | Slide 6 says so plainly — it becomes *evidence for* the audit |
| Blocker B kills the WER claim | Drop the claim; GRL result and audit unaffected |
| Saaras output unscoreable | Report it as a finding ("vendor output resists standardized audit") — that is itself audit material |
| Qwen3-ASR refuses/breaks | Fallback: distil-whisper + IndicWhisper on the Modal skeleton |
| Modal free tier exhausted | Won't happen (<$5 projected); caching means partial runs are never lost |

## The closing argument for the panel

This isn't another pivot. It's the half of Phase 2 that *worked*, pointed at the gap the funded labs left open, executed for approximately zero rupees, with the training work recycled as motivation and the ambitious story (OPD — already PoC-validated for $0.16, see this repo) deliberately parked with a date on it. That's not a pivot; that's triage done properly.
