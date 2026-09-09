# Days 1–3 brief for Dr. Bhaskarjyoti Das

**PW25_BJD_05 — Whose Indian accent?** One page. 9 September 2026.  
Team: Aditya Sharma, Adithya V Holla, C Kaustubh, Aditi Mangala Udaya.

## What is already a negative result (exploratory, not a paper claim)

Six systems on a frozen 2,500-clip Svarah sample show the **same structural pattern**: a significant per-family WER gap, and **Sino-Tibetan (Bodo) worst in every row**, including Saaras V3 (best overall 5.81%, ΔDP 3.45 pp). GRL fairness training failed because SPIRE-SIES has **zero** ST speakers. On-policy distillation of whisper-tiny from large-v3 did not beat off-policy pseudo-labels on LibriSpeech (7.89% vs 7.57%), and a 16-step Modal smoke left WER flat. We do not have a training method that closes the Indian-English subgroup gap.

## Proposed question (hypothesis, not novelty)

**When do corrections learned from a teacher’s mistakes transfer to Indian-English speakers, and when do they fail?**

The mechanism: a strong teacher (whisper-large-v3) makes *systematic* errors on some Indian L1s. Students trained on those transcripts can inherit the errors. Filtering by agreement with a reference, or by confidence, may drop noise without correcting the structured bias. We will not invent a new loss until an existing method leaves a demonstrated failure.

## Closest competing method

[**Pseudo2Real**](https://aclanthology.org/2026.findings-acl.59/) (Lin et al., ACL Findings 2026). In a *source* domain with both gold and teacher transcripts, fine-tune two copies of the same init; the weight difference \(\tau = \theta_s^{\text{real}} - \theta_s^{\text{pseudo}}\) is added to a pseudo-labeled *target* model. Up to 35% relative WER drop on AfriSpeech-200 with Whisper-tiny. They already state that confidence filtering “suppress[es] noise without correcting structured teacher biases.” Their own limitations are the ones we would have to beat:

1. **Stationarity of \(\tau\).** Source and target are assumed to share the same systematic pseudo-label errors. Indian-English L1s — especially Bodo, n small — may not share the error geometry of AfriSpeech folds (or of Hindi-heavy source data).
2. **Source supervision.** \(\tau\) needs a labeled source. If that source is Western or Hindi-dominant English, the vector may encode the *wrong* correction for Dravidian or ST speakers.
3. **One vector vs many L1s.** Pseudo2Real-SC clusters speakers acoustically; it does not guarantee that cluster = native-language group, and it still averages \(\tau_c\) into one shift.

uDistil-Whisper (NAACL 2025) is the strongest *filtering* baseline, not a correction method. We will include it in Days 4–10 only if the gate below fires.

## What we are doing this week, before any new method

1. **Dual transcription of 120 clips** (six L1s × accepted/rejected × 10), two reviewers, blind to both transcripts. If the Svarah reference is itself the disagreement, we repair data before training.
2. **The missing label-source experiment:** 2 recording pools (filtered vs matched-random) × 2 targets (reference vs teacher) × 3 paired seeds = **12 runs, one implementation**. Eval at 600 and **2,400 updates (primary, designated before launch)**. Same init, order, decode, optimizer within each pair.
3. **Stop rule (screening, not significance):** continue only if teacher targets **worsen matched-random macro WER by ≥ 0.5 pp on average**, in **at least two seeds**, with **no reversal under the Whisper normalizer**. Otherwise we close this mechanism under the tested conditions and keep a technical report.

SPIRE-SIES and L2-ARCTIC stay frozen. Test numbers will not trigger retuning.

## Question for you

If the gate fires, the remaining scientific work is: *does Pseudo2Real’s correction transfer across Indian L1s, and where does it fail?* If it already closes the gap, that is a result — not a paper about a new loss. **Would resolving Pseudo2Real’s transfer failures on Indian English (especially ST/Bodo, and source–target mismatch of \(\tau\)) matter as a contribution, or should we change direction even if the 0.5 pp screen passes?**
