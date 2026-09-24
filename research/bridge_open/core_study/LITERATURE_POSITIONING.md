# Contribution positioning — 24 September 2026

## Retrieval and reading

Main-agent ORX discovery used one embedding query about open-ASR quality/efficiency evaluation for Indian languages and one focused `Vimarsha` keyword query. No date cutoff; default relevance ranking. Retrieved candidate order: Vimarsha (2609.24199), SraVaani (2608.08235), Indic DiarBench (2607.23808), IndicContextEval (2606.19157), Vaani Benchmark (2606.21408), OpenWER (2606.21237), Responsible ASR (2606.18659), MultiClin (2606.17826). Read the full texts of Vimarsha, OpenWER, and Responsible ASR for claim-level comparison. The other five are discovery candidates, not verified claim support.

## What we cannot claim

[Vimarsha](https://www.alphaxiv.org/pdf/2609.24199) already presents broad Indian-language evaluation, human-checked reference variants, demographic coverage, and difficult natural audio. Its section 4.1 lists IndicConformer alongside commercial/audio-language systems; its section 5 evaluates model ranking, demographics, duration and speaking rate. Our three FLEURS languages and partial Svarah shard cannot supersede that scope. “First comprehensive Indian-ASR benchmark” is unsupported.

[OpenWER](https://www.alphaxiv.org/pdf/2606.21237) studies language-dependent normalization and token alignment, including a 52-language comparison in section 3. Our common scorer provides a reproducibility contract, not a new metric. Single-reference errors cannot automatically be called meaning errors. The primary metric must remain frozen; a future orthographic sensitivity analysis would be a separately declared secondary analysis.

[Responsible ASR](https://www.alphaxiv.org/pdf/2606.18659) already compares open and commercial systems on Hindi and Indian-English telephony and evaluates adaptation. Our FLEURS read speech and available Svarah shard do not establish narrow-band conversational robustness. We are not proposing an adaptation method.

## Defensible candidate contribution

The narrower candidate is an auditable comparison of six specified inference recipes, with traceable original audio and references, complete failure accounting, and measured same-hardware batch-one resource use on a common panel. Link each quality conclusion to its exact language/corpus condition; use the matched panel for quality/resource interpretation. This may support a useful reproducibility or systems-evaluation paper, but novelty and venue acceptance are not established by adding model rows.

The study should answer a concrete practitioner decision: whether choosing a smaller model changes error enough to matter, what compute/memory that choice actually saves under the measured recipe, and whether a favorable average hides unacceptable failures. Already observed anomalies are descriptive findings; they do not justify choosing a repair method retrospectively and presenting it as a prespecified test.

## Claim gates

1. A resource tradeoff requires measured resources and matched panel outputs; parameter counts alone are insufficient.
2. A semantic failure explanation requires language-competent listening, not only Unicode script counts or high WER.
3. Fairness across people requires verified speaker/subgroup labels and adequate independent coverage; those are absent here.
4. The artifact needs enough independent validation for another researcher to reproduce the measurements; pipeline receipts alone do not establish scientific novelty.
5. A final submission needs a clear distinction from prior art and completed, transparently bounded review. Until then, label the manuscript a working audit draft.

## Dataset and model attribution

A second focused retrieval pass located the original Svarah paper (2305.15760). The pinned FLEURS dataset card supplied its original paper identifier (2205.12446). Retrieved source texts and read their dataset descriptions; both corpora predate this study. Svarah itself already compared open and commercial systems, so that combination is not a novelty claim here. Read the introductions/model descriptions of the original Whisper paper (2212.04356) and Qwen3-ASR report (2601.21337) for family attribution. Current variant identities come from the executed pinned model artifacts. The pinned OmniASR-CTC-1B model card provides implementation/model attribution; its generic LLM example is not the CTC decoding recipe used here. The five additional references in the manuscript are attribution, not evidence that our measurements reproduce vendor results.
