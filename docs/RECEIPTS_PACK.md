# Receipts Pack — for Kau's deck slides 2–3

**AT3, per `docs/WEEKEND_PLAN.md`:** *"screenshots/quotes with page refs — Sarvam blog
(Svarah eval, underrepresentation rationale, five aggregate metrics, no
subgroup table), ASR-FAIRBENCH's mixed-effects paragraph, the clinical
audit's system list."* All three below, with exact quotes, verified against
the live source on 2026-08-29 (not from memory), plus one finding that's
sharper than what the brief asked for.

---

## 1. Sarvam Saaras V3 blog — underrepresentation rationale, Svarah "validation," no subgroup table

**Source:** [sarvam.ai/blogs/asr](https://www.sarvam.ai/blogs/asr), "Introducing
Saaras V3," published February 10, 2026. Section: **"English Benchmark: Svarah."**

> "Indian-accented English is underrepresented in widely used ASR benchmarks
> such as LibriSpeech and Switchboard, despite India having an estimated 130
> million English speakers. As a result, performance on these standard
> datasets does not fully reflect how systems handle English as spoken in
> India.
>
> To address this gap, Saaras V3 was evaluated on Svarah, a 9.6-hour
> benchmark comprising 117 speakers across 65 districts in 19 Indian states.
> The dataset captures substantial accent variation and includes both read
> and spontaneous conversational speech across real-world domains, providing
> a more representative assessment of Indian English ASR performance.
>
> This performance reflects the model's strong ability to handle accent
> variation and phonetic shifts more reliably than systems primarily
> optimized on Western English benchmarks."

**Three things to point at on the slide:**

1. **The underrepresentation argument is verbatim ours.** "Indian-accented
   English is underrepresented in widely used ASR benchmarks... despite India
   having an estimated 130 million English speakers" is exactly Chapter 1's
   framing — shipped by a funded lab, on our exact benchmark, before we could
   publish it.
2. **No WER number is given for Svarah at all.** Read that Svarah section
   again: it validates on Svarah, describes the dataset, and asserts the
   model handles "accent variation... more reliably" — but never states a
   number. Not an aggregate WER, let alone a subgroup one. This is a
   *stronger* gap than "aggregate-only" — it's "claimed, unquantified."
3. **Where they do quantify, it's a single number, no subgroups.** Their
   IndicVoices claim two paragraphs earlier — "Saaras V3 achieves a word
   error rate of 19.31 percent" — is one aggregate number across 10
   languages, with a second sentence noting the gap "widens" on 12 more
   languages, but still no per-language or per-accent breakdown, and no
   significance test anywhere in the post.

**Companion source (aggregate-metrics framing):** [sarvam.ai/speech-to-text](https://www.sarvam.ai/speech-to-text), retrieved 2026-08-29:

> "Sarvam also uses a 5-metric evaluation framework (WER, LLM-WER, COMET,
> Intent Score, Entity Preservation) instead of relying on WER alone,
> because for Indian languages, two transcriptions with identical WER can
> have very different semantic accuracy."

Five metrics, zero subgroups, zero significance tests — even where they
explicitly reject "WER alone" as insufficient, the axis they add is
*metric diversity*, not *demographic-subgroup* diversity. That's precisely
the gap our audit fills.

---

## 2. ASR-FAIRBENCH — the mixed-effects Poisson paragraph

**Source:** A. K. Rai, S. Rahangdale, U. Anand, A. Mukherjee, "ASR-FAIRBENCH:
Measuring and Benchmarking Equity Across Speech Recognition Systems,"
*Interspeech 2025*, arXiv:2505.11572. Section 2.2.1, "Word error rate and
fairness model" (verified against the arXiv HTML rendering, 2026-08-29):

> "To assess fairness across demographic groups, we employ mixed-effects
> Poisson regression:
>
> log(WER_i) = β₀ + β₁X_i + β₂Z_i + u_i          (1)
>
> where X_i represents demographic attributes, Z_i represents covariates,
> and β₁ quantifies disparity through disparity ratio = e^(β₁)."

And from §2.2.3, "Statistical adjustment and FAAS":

> "Statistical significance of fairness disparities is assessed via the
> Likelihood Ratio Test comparing full and reduced models... The resulting
> p-value indicates whether disparities are statistically significant. When
> disparities are significant (p < 0.05), we apply a proportional penalty to
> the category score."

**Why this is the load-bearing citation for our pivot, not just a nice-to-have:**

- This is *functionally the same test* we built independently: a Poisson
  model of WER against a demographic covariate, with a likelihood-ratio /
  deviance test and a p < 0.05 threshold for "real" disparity. We did not
  copy this — our Phase 2 pipeline predates knowledge of this paper — but it
  means we can no longer claim the statistical machinery as a contribution.
  We can, and should, cite it, adopt its framing explicitly, and spend our
  novelty budget elsewhere (the multi-system audit itself, and the
  Indian-accent axis it never covers).
- **Their dataset is Fair-Speech (US English, self-reported demographics:
  age, gender, ethnicity, location, native-language status) — not Indian
  accents, and not Svarah.** Their own conclusion (§4) singles out
  Whisper-tiny for "better overall fairness" despite higher WER and flags
  that "fairness issues can outweigh slight WER improvements" for
  fine-tuned Wav2Vec/HuBERT — i.e., their own headline finding is exactly
  our thesis ("aggregate WER hides subgroup movement"), just demonstrated on
  a different population. **The accent-family axis on Indian English is the
  one gap even the paper that already published our statistical test does
  not fill.** That's the sentence for the slide.

---

## 3. The clinical audit paper — system list and axes

**Source:** S. Kumar, P. Shivaprakash, A. Manoharan, A. Kurariya,
D. Mukherjee, L. Shukla, A. Mukherjee, P. Chand, P. Murthy, "ASR Under the
Stethoscope: Evaluating Biases in Clinical Speech Recognition across Indian
Languages," arXiv:2512.10967, submitted 30 Nov 2025 (verified against the
arXiv abstract page and HTML rendering, 2026-08-29).

> "In this study, we conduct the first systematic audit of ASR performance
> on real-world clinical interview data spanning Kannada, Hindi, and Indian
> English, comparing leading models including IndicWhisper, Whisper-large-v3,
> Sarvam, Google speech-to-text, Gemma3n, Omnilingual, Vaani, and Gemini...
> with a particular focus on error patterns affecting patients vs.
> clinicians and gender-based or intersectional disparities."

**System list, exactly as stated in the paper (8 systems):** IndicWhisper,
Whisper-large-v3, Sarvam, Google Speech-to-Text, Gemma3n, Omnilingual, Vaani,
Gemini.

**Why this is the receipt for "third-party ASR audits are a normal,
publishable thing, and our axis is still open":**

- This paper **includes Sarvam** in a multi-system audit and got through
  peer review (submitted to arXiv Nov 2025; a related/extended companion,
  "SamaVaani," arXiv:2606.26901, went further and proposed a debiasing
  method on top of the same audit). A third-party comparative audit of
  Indian ASR systems, including a commercial one, is not a novel genre we're
  inventing — it's a genre with precedent.
- **Its subgroup axes are speaker role (doctor vs. patient) and gender**, in
  a clinical-interview setting, across Kannada/Hindi/Indian English. It does
  **not** stratify by accent or language family, and it does not touch
  Svarah. **The accent-family axis, on the benchmark Sarvam itself uses to
  validate, is still nobody's paper.** That is our slide-3 punchline.

---

## How to use this on the day

Every quote above is reproduced verbatim from the live source, with the
retrieval date noted, so a panel member can pull up the same page or PDF and
verify it in front of you — that's the point of a "receipts pack" instead of
a paraphrase. Suggested framing for slides 2–3:

- **Slide 2 (the encroachment):** the Sarvam quote — underrepresentation
  rationale is verbatim ours, Svarah is their own validation benchmark, and
  they report *no number* for it, let alone a subgroup one.
- **Slide 3 (the literature closes the "we invented the stats" door, and
  opens the real gap):** ASR-FAIRBENCH already ships our exact statistical
  test — cite it, don't compete with it — on a non-Indian population; the
  clinical audit shows third-party Indian-ASR audits are publishable, on
  role/gender, not accent. **Six systems, one frozen protocol, per-family
  WER with a p-value, on the benchmark the flagship model itself uses** is
  the row nobody has published — and (see `audit/README.md`) it is no longer
  a plan, it is six completed rows.
