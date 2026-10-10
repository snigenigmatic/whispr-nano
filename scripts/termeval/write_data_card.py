"""Render the observed data probe, with the two gated-source access outcomes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/night1/data_summary.json"
OUT = ROOT / "results/night1/data_card.md"


def percentages(counts: dict) -> str:
    total = sum(counts.values())
    return ", ".join(f"{key}: {value:,} ({value / total:.1%})" for key, value in counts.items())


def samples(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    mucs, hiacc = data["mucs_hien"], data["hiacc"]
    text = [
        "# Night 1 public-corpus data card",
        "",
        "[OBSERVED] Generated from `scripts/termeval/probe_data.py` with seed 20261011. Exact machine-readable counts are in `data_summary.json`; full transcripts and audio stay under ignored `data/`.",
        "",
        "## MUCS Hindi-English test",
        "",
        "[OBSERVED] Source: [OpenSLR SLR104](https://www.openslr.org/104/), CC BY-SA 4.0. The downloaded `Hindi-English_test.tar.gz` has SHA-256 `93e358b3bf8233a897fcd353c1f4f98fdda6b8c01b7eed17a70c7dd26e984b37`.",
        f"[OBSERVED] Layout: `test/*.wav` plus `test/transcripts/{{text,segments,utt2spk,spk2utt,wav.scp,spkr_list}}`; {mucs['recordings']} long recordings, {mucs['utterances']} sentence segments, {mucs['speakers']} speaker IDs. The source is 16-kHz audio; segments were cut into mono 16-kHz clips.",
        f"[OBSERVED] Segment hours {mucs['duration_hours']:.3f}; duration min/median/95th/max {mucs['duration_s_min']:.2f}/{mucs['duration_s_median']:.2f}/{mucs['duration_s_p95']:.2f}/{mucs['duration_s_max']:.2f} seconds. The `segments` file is required to turn long recordings into utterances.",
        f"[OBSERVED] Normalized reference token scripts: {percentages(mucs['token_classes'])}.",
        "[OBSERVED] Speaker and recording IDs are present. `utt2spk` is used for cluster bootstrap, with 30 clusters.",
        "",
        "### 20 seed-selected MUCS transcript lines",
        "",
        samples(mucs["samples"]),
        "",
        "## HiACC",
        "",
        "[OBSERVED] Source: [Zenodo record 15551669](https://zenodo.org/records/15551669), CC BY 4.0. Downloaded `Corpus.zip` MD5 `dd6cc9354e1dee5e2f25bc5243df88ac` and SHA-256 `fd83414b3aba28a4ae86c110f4aab0981621f9f3a1b105c161bf2dfd14737ccf` both verified.",
        f"[OBSERVED] {hiacc['total_utterances']} mono 16-kHz utterance WAVs from {hiacc['total_speakers']} speaker IDs, {hiacc['total_hours']:.3f} hours by WAV header duration. Normalized reference token scripts: {percentages(hiacc['token_classes'])}.",
        "[OBSERVED] Folder structure: `Corpus/{adult,children}/audio/{train,val,test}_split/`, transcripts, `metadata/{speaker_info.csv,sentence_stats.csv}`, and `annotations/code_switched_labels.json`.",
        "[OBSERVED] The annotation JSON has one row per utterance with an utterance-level `label` (`none`, `intra-sential`, `inter-sential`); the released JSON does not contain word-level language tags, despite the bundled README describing word-level annotations.",
        "[OBSERVED] The bundled README calls the splits speaker-disjoint. Every adult speaker ID and every child speaker ID occurs in each of train, validation, and test. These splits are therefore **not speaker-disjoint** in the downloaded archive; do not claim unseen-speaker test performance from them.",
        "[OBSERVED] `AD63` appears in 119 adult WAV/transcript rows but not in `adult/metadata/speaker_info.csv`; `AD65` has a speaker-info row but no WAV. Overall ASR scoring retains AD63. Analyses by age/gender/L1 must exclude its missing values and state the reduced support.",
    ]
    for group in ("adult", "children"):
        g = hiacc["groups"][group]
        m = hiacc["metadata_files"][group]
        text += [
            "", f"### {group.title()}", "",
            f"[OBSERVED] {g['utterances']} utterances, {g['speakers']} audio speaker IDs, {g['hours']:.3f} hours; split utterance counts {g['split_counts']}; speaker overlap counts {g['split_speaker_overlap']}.",
            f"[OBSERVED] Metadata columns: `speaker_info.csv` {m['speaker_info.csv']} ({m['speaker_info_rows']} rows); `sentence_stats.csv` {m['sentence_stats.csv']} ({m['sentence_stats_rows']} rows); `code_switched_labels.json` {m['code_switched_labels.json']} ({g['annotation_rows']} rows).",
            f"[OBSERVED] Speaker metadata distributions: {g['speaker_metadata']}. Annotation labels: {g['annotation_labels']}.",
            f"[OBSERVED] Normalized reference token scripts: {percentages(g['token_classes'])}.",
            "", f"#### 20 seed-selected {group} transcript lines", "", samples(g["samples"]),
        ]
    text += [
        "", "## Access checks on remaining planned sources", "",
        "[OBSERVED] [Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) exposes public metadata but its Parquet content returned HTTP 401 without a locally available accepted-access token. No Lahaja audio, rows, group counts, or transcripts were inspected; it is excluded from this run.",
        "[OBSERVED] [VITB-HEBiC](https://huggingface.co/datasets/PalashJain/VITB-HEBiC_Code-switch_corpus) returned HTTP 401 for its dataset API and file tree. No claims from its paper/card were verified against files; it is excluded from this run.",
        "", "## Scope limits", "",
        "[ASSUMED] All released HiACC splits are pooled for a corpus feasibility audit. The original split label is retained in the local manifest so split-specific scores can be reported. The pooled result must not be framed as a held-out speaker benchmark.",
        "[OBSERVED] MUCS speaker IDs and HiACC filename prefixes are available for speaker-cluster bootstrap; demographic metadata is unavailable for MUCS.",
        "[UNVERIFIED] Model training-data overlap with either corpus is unknown.",
        "",
    ]
    OUT.write_text("\n".join(text))
    print(OUT)


if __name__ == "__main__":
    main()
