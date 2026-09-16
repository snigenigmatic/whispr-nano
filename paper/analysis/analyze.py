"""Recompute paired Svarah audits without importing the old ast_asr project.

Example:
    .venv/bin/python -m paper.analysis.analyze \
      --results-dir audit/svarah_fairness_audit \
      --matched-results-dir paper/data/matched_predictions \
      --spelling-map paper/data/whisper_normalizer.json \
      --metadata paper/data/svarah_metadata.csv \
      --cluster-column recording_cluster_id --cluster-kind recording \
      --output-dir paper/results

Missing speaker metadata never triggers an utterance bootstrap. Recording-only
metadata support explicitly labelled diagnostic intervals, never p-values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


BASELINE = "whisper-large-v3"
GROUPS = ("Bodo", "Dravidian", "Indo-Aryan")
SUBGROUP_CONTRASTS = (
    ("Bodo_minus_non_Bodo", "Bodo", "non_Bodo"),
    ("Bodo_minus_Dravidian", "Bodo", "Dravidian"),
    ("Bodo_minus_Indo_Aryan", "Bodo", "Indo-Aryan"),
    ("Indo_Aryan_minus_Dravidian", "Indo-Aryan", "Dravidian"),
)


def normalize_minimal(text: str) -> str:
    """Exactly reproduce all historical reference/hypothesis normalizations."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def word_edit_counts(reference: str, hypothesis: str) -> EditCounts:
    """Word Levenshtein alignment; deterministic diagonal/deletion/insertion ties.

    Inputs are already normalized. Different minimum-cost alignments may allocate
    substitutions/deletions/insertions differently, but total error is invariant.
    """
    ref, hyp = reference.split(), hypothesis.split()
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, 1):
        curr = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, 1):
            cost, sub, delete, insert = prev[j - 1]
            mismatch = int(ref_word != hyp_word)
            diagonal = (cost + mismatch, sub + mismatch, delete, insert)
            cost, sub, delete, insert = prev[j]
            deletion = (cost + 1, sub, delete + 1, insert)
            cost, sub, delete, insert = curr[j - 1]
            insertion = (cost + 1, sub, delete, insert + 1)
            curr.append(min((diagonal, deletion, insertion), key=lambda item: item[0]))
        prev = curr
    _, sub, delete, insert = prev[-1]
    return EditCounts(sub, delete, insert, len(ref))


def corpus_wer(errors: np.ndarray, references: np.ndarray) -> float | None:
    denominator = int(np.sum(references))
    return float(100 * np.sum(errors) / denominator) if denominator else None


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(row: dict) -> tuple:
    return tuple(row.get(key) for key in (
        "reference_text", "primary_language", "family", "gender", "duration_s"
    ))


def load_predictions(directory: Path) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    systems, provenance = {}, []
    for path in sorted(directory.glob("results_*_svarah_clean.csv")):
        system = path.name[len("results_"):-len("_svarah_clean.csv")]
        rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        if not rows:
            raise ValueError(f"Empty prediction file: {path}")
        indexed = {}
        for row in rows:
            for key in ("utt_id", "reference_text", "hypothesis_text", "primary_language", "family"):
                if key not in row:
                    raise ValueError(f"{path}: missing {key}")
            if row["utt_id"] in indexed:
                raise ValueError(f"{path}: duplicate utterance ID {row['utt_id']}")
            if row["family"] == "Sino-Tibetan" and row["primary_language"] != "Bodo":
                raise ValueError("Bodo-only group no longer describes this input; revise protocol")
            indexed[row["utt_id"]] = row
        systems[system] = indexed
        provenance.append({"path": str(path.resolve()), "sha256": hash_file(path), "rows": len(rows)})
    if not systems:
        raise ValueError(f"No prediction CSVs in {directory}")
    return systems, provenance


def validate_panel(systems: dict[str, dict], ids: list[str]) -> None:
    first = systems[next(iter(systems))]
    for utt_id in ids:
        expected = _signature(first[utt_id])
        for system, rows in systems.items():
            if _signature(rows[utt_id]) != expected:
                raise ValueError(f"Conflicting metadata for {utt_id} in {system}")


def validate_frozen_inputs(systems: dict[str, dict], provenance: list[dict], manifest: dict) -> None:
    """A missing model or partial export must not silently redefine the study."""
    actual = {Path(item["path"]).name: item for item in provenance}
    expected = manifest["source_files"]
    if set(actual) != set(expected):
        raise ValueError(f"Historical prediction files differ from frozen manifest: "
                         f"missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}")
    for filename, frozen in expected.items():
        if actual[filename]["sha256"] != frozen["sha256"] or actual[filename]["rows"] != frozen["n_rows"]:
            raise ValueError(f"Historical prediction file changed since protocol freeze: {filename}")
    open_ids, saaras_ids = set(manifest["open_cohort_ids"]), set(manifest["saaras_cohort_ids"])
    if len(open_ids) != 2500 or len(saaras_ids) != 450:
        raise ValueError("Frozen cohort sizes no longer match the 2500/450 protocol")
    if len(systems) != 6:
        raise ValueError("Frozen protocol requires exactly six systems")
    for system, rows in systems.items():
        desired = saaras_ids if "saaras" in system.lower() else open_ids
        if set(rows) != desired:
            raise ValueError(f"{system} utterance IDs differ from the frozen cohort")


def validate_unchanged_saaras(original: dict[str, dict], matched: dict[str, dict]) -> None:
    for system in original:
        if "saaras" in system.lower() and matched.get(system) != original[system]:
            raise ValueError("The matched-panel Saaras rows must be unchanged historical predictions")


def validate_metadata_identity(row: dict, metadata: dict, utt_id: str) -> None:
    for field in ("reference_text", "primary_language", "family", "gender"):
        if metadata.get(field) and str(metadata[field]).strip() != str(row.get(field, "")).strip():
            raise ValueError(f"Metadata identity conflict for {utt_id}: {field}")
    if metadata.get("duration_s") and row.get("duration_s"):
        if abs(float(metadata["duration_s"]) - float(row["duration_s"])) > 0.02:
            raise ValueError(f"Metadata duration conflict for {utt_id}")


def validate_full_panel(full: dict[str, dict], expected_systems: set[str], metadata: dict[str, dict]) -> None:
    """The full benchmark is a census of the pinned 6,656 rows, not a subsample."""
    expected_ids = {f"svarah_{index:05d}" for index in range(6656)}
    if set(metadata) != expected_ids:
        raise ValueError("Full-corpus analysis requires complete pinned metadata for all 6,656 IDs")
    if set(full) != expected_systems or len(expected_systems) != 5:
        raise ValueError("Full-corpus analysis requires exactly the five frozen open systems")
    for system, rows in full.items():
        if set(rows) != expected_ids:
            raise ValueError(f"Full-corpus {system} is missing/adding benchmark utterances")
        for utt_id, row in rows.items():
            recovered = metadata[utt_id]
            if not recovered.get("reference_text") or not recovered.get("primary_language"):
                raise ValueError(f"Full-corpus metadata lack reference/language for {utt_id}")
            validate_metadata_identity(row, recovered, utt_id)
    validate_panel(full, sorted(expected_ids))


def load_metadata(path: Path | None, cluster_column: str) -> tuple[dict[str, dict], dict]:
    if path is None:
        return {}, {"status": "absent", "cluster_column": cluster_column}
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    result = {}
    for row in rows:
        if "utt_id" not in row or cluster_column not in row:
            raise ValueError(f"Metadata need utt_id and {cluster_column}")
        if row["utt_id"] in result:
            raise ValueError(f"Duplicate metadata ID {row['utt_id']}")
        result[row["utt_id"]] = row
    return result, {"path": str(path.resolve()), "sha256": hash_file(path),
                    "rows": len(rows), "cluster_column": cluster_column}


@dataclass
class BootstrapPlan:
    weights: np.ndarray
    inverse: np.ndarray
    cluster_ids: list[str]
    stratum_cluster_counts: dict[str, int]
    kind: str

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self.inverse, weights=values, minlength=len(self.cluster_ids))

    def ratios(self, errors: np.ndarray, references: np.ndarray) -> np.ndarray:
        numerators = self.weights @ self.aggregate(errors)
        denominators = self.weights @ self.aggregate(references)
        return np.divide(100 * numerators, denominators,
                         out=np.full(len(numerators), np.nan), where=denominators > 0)

    def support(self, mask: np.ndarray) -> int:
        return int(len(np.unique(self.inverse[mask])))


def make_bootstrap_plan(cluster_ids: list[str], strata: list[str], resamples: int,
                        seed: int, kind: str = "speaker") -> BootstrapPlan:
    if any(not value for value in cluster_ids):
        raise ValueError("Blank cluster IDs cannot be resampled")
    ordered = sorted(set(cluster_ids))
    index = {key: i for i, key in enumerate(ordered)}
    inverse = np.array([index[key] for key in cluster_ids], dtype=int)
    cluster_strata = {}
    for cluster, stratum in zip(cluster_ids, strata):
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise ValueError(f"Cluster {cluster!r} crosses native-language strata")
        cluster_strata[cluster] = stratum
    by_stratum = defaultdict(list)
    for cluster in ordered:
        by_stratum[cluster_strata[cluster]].append(index[cluster])
    rng = np.random.default_rng(seed)
    weights = np.zeros((resamples, len(ordered)), dtype=np.int32)
    for stratum in sorted(by_stratum):
        indexes = by_stratum[stratum]
        count = len(indexes)
        weights[:, indexes] = rng.multinomial(count, np.full(count, 1 / count), size=resamples)
    return BootstrapPlan(weights, inverse, ordered,
                         {key: len(value) for key, value in by_stratum.items()}, kind)


def interval(samples: np.ndarray | None, n_clusters: int | None,
             kind: str | None) -> tuple[float | None, float | None, str]:
    if samples is None or n_clusters is None:
        return None, None, "unsupported_without_complete_verified_cluster_metadata"
    if n_clusters < 2:
        return None, None, "unsupported_fewer_than_two_clusters"
    finite = samples[np.isfinite(samples)]
    if len(finite) < 0.95 * len(samples):
        return None, None, "unsupported_too_many_zero_denominator_resamples"
    low, high = np.percentile(finite, [2.5, 97.5])
    status = ("diagnostic_filename_block_only" if kind == "recording" else
              "exploratory_unverified_demographic_profile" if kind == "demographic_profile" else
              "exploratory_fewer_than_ten_speakers" if n_clusters < 10 else "speaker_bootstrap")
    return float(low), float(high), status


def centered_bootstrap_p(observed: float, samples: np.ndarray) -> float:
    """Approximate two-sided tail area, not an exact randomization test."""
    finite = samples[np.isfinite(samples)]
    return float((1 + np.sum(np.abs(finite - observed) >= abs(observed))) / (len(finite) + 1))


def holm_adjust(pvalues: list[float | None], family_size: int | None = None) -> list[float | None]:
    """Holm step-down; withheld tests remain in the predeclared family size."""
    ordered = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    size = len(pvalues) if family_size is None else family_size
    adjusted: list[float | None] = [None] * len(pvalues)
    previous = 0.0
    for rank, (pvalue, index) in enumerate(ordered):
        previous = max(previous, min(1.0, (size - rank) * pvalue))
        adjusted[index] = previous
    return adjusted


def _contrast_row(name: str, difference: float | None, samples: np.ndarray | None,
                  n_left: int | None, n_right: int | None, kind: str | None,
                  minimum_test_clusters: int) -> dict:
    n_min = min(n_left, n_right) if n_left is not None and n_right is not None else None
    low, high, status = interval(samples, n_min, kind)
    pvalue = None
    if (kind == "speaker" and n_min is not None and n_min >= minimum_test_clusters
            and samples is not None and difference is not None
            and np.isfinite(samples).all()):
        pvalue = centered_bootstrap_p(difference, samples)
    return {"contrast": name, "difference_pp": difference, "ci_low_pp": low,
            "ci_high_pp": high, "ci_status": status, "n_clusters_left": n_left,
            "n_clusters_right": n_right, "approximate_bootstrap_p": pvalue,
            "holm_adjusted_p": None}


def historical_validation(systems: dict[str, dict]) -> dict:
    bad_normalizations = bad_distances = bad_arithmetic = 0
    total_rows = 0
    for rows in systems.values():
        for row in rows.values():
            total_rows += 1
            ref = normalize_minimal(row["reference_text"])
            hyp = normalize_minimal(row["hypothesis_text"])
            for field, value in (("reference_normalized", ref), ("hypothesis_normalized", hyp)):
                if field in row and row[field] != value:
                    bad_normalizations += 1
            counts = word_edit_counts(ref, hyp)
            if all(key in row for key in ("n_sub", "n_del", "n_ins", "ref_len_words", "wer")):
                total = sum(int(row[key]) for key in ("n_sub", "n_del", "n_ins"))
                bad_distances += int(total != counts.errors)
                bad_arithmetic += int(int(row["ref_len_words"]) != counts.reference_words or
                                      abs(float(row["wer"]) - total / max(counts.reference_words, 1)) > 1e-8)
    if bad_normalizations or bad_distances or bad_arithmetic:
        raise ValueError(f"Input validation failed: normalizations={bad_normalizations}, "
                         f"distance={bad_distances}, arithmetic={bad_arithmetic}")
    return {"rows_verified": total_rows, "normalization_mismatches": bad_normalizations,
            "edit_distance_mismatches": bad_distances, "arithmetic_mismatches": bad_arithmetic}


def poisson_diagnostic(errors: np.ndarray, references: np.ndarray, groups: np.ndarray) -> dict:
    """Closed-form grouped-offset Poisson fits; historical test is diagnostic only."""
    valid = references > 0
    err, ref, grp = errors[valid], references[valid], groups[valid]
    if not len(ref) or not err.sum():
        return {"status": "undefined_zero_reference_or_errors"}
    null = ref * err.sum() / ref.sum()
    alt = np.zeros(len(err), dtype=float)
    unique = np.unique(grp)
    for group in unique:
        mask = grp == group
        alt[mask] = ref[mask] * err[mask].sum() / ref[mask].sum()
    def deviance(mean: np.ndarray) -> float:
        positive = err > 0
        terms = -(err - mean).astype(float)
        terms[positive] += err[positive] * np.log(err[positive] / mean[positive])
        return float(2 * terms.sum())
    difference = max(0.0, deviance(null) - deviance(alt))
    degrees = len(unique) - 1
    positive_mean = alt > 0
    pearson = float(np.sum((err[positive_mean] - alt[positive_mean]) ** 2 / alt[positive_mean]))
    from scipy.stats import chi2
    return {"status": "diagnostic_independent_poisson_not_fairness_inference",
            "deviance_difference": difference, "df": degrees,
            "unadjusted_poisson_p": float(chi2.sf(difference, degrees)),
            "pearson_dispersion": pearson / (len(err) - len(unique)) if len(err) > len(unique) else None,
            "zero_reference_rows_excluded_from_diagnostic": int((~valid).sum())}


def analyze_panel(name: str, systems: dict[str, dict], ids: list[str],
                  normalizers: dict[str, Callable], metadata: dict[str, dict],
                  cluster_column: str, cluster_kind: str, resamples: int, seed: int,
                  minimum_test_clusters: int) -> dict:
    validate_panel(systems, ids)
    first = systems[next(iter(systems))]
    for key in ids:
        validate_metadata_identity(first[key], metadata.get(key, {}), key)
    languages = np.array([first[key]["primary_language"] for key in ids])
    groups = np.array(["Bodo" if first[key]["family"] == "Sino-Tibetan" else
                       first[key]["family"] for key in ids])
    clusters = [metadata.get(key, {}).get(cluster_column, "").strip() for key in ids]
    covered = sum(bool(cluster) for cluster in clusters)
    cluster_status = {"kind": cluster_kind, "column": cluster_column, "covered_utterances": covered,
                      "total_utterances": len(ids), "complete": covered == len(ids),
                      "inference_supported": cluster_kind == "speaker" and covered == len(ids)}
    if cluster_kind == "recording":
        cluster_status["interpretation"] = (
            "Filename-derived blocks; physical recording and speaker identity are unverified. "
            "Intervals are diagnostic and cannot establish person-level disparities."
        )
    elif cluster_kind == "demographic_profile":
        cluster_status["interpretation"] = (
            "Exact matches on eight public demographic fields; these profiles are not verified speakers. "
            "Distinct people may share a profile and one person may have inconsistent profiles. "
            "Intervals are exploratory grouping sensitivity, not speaker-level inference."
        )
    plan = None
    if covered == len(ids) and cluster_kind in ("speaker", "recording", "demographic_profile"):
        plan = make_bootstrap_plan(clusters, languages.tolist(), resamples, seed, cluster_kind)
        cluster_status.update({"unique_clusters": len(plan.cluster_ids),
                               "clusters_per_native_language": plan.stratum_cluster_counts,
                               "singleton_language_strata": [key for key, value in
                                   plan.stratum_cluster_counts.items() if value < 2]})
    supports = []
    group_masks = {group: groups == group for group in GROUPS}
    group_masks["non_Bodo"] = groups != "Bodo"
    selections = [("overall", "all", np.ones(len(ids), dtype=bool))]
    selections += [("group", group, mask) for group, mask in group_masks.items() if group != "non_Bodo"]
    selections += [("language", language, languages == language) for language in sorted(set(languages))]
    for level, group, mask in selections:
        indexes = np.flatnonzero(mask)
        supports.append({"level": level, "group": group, "n_utterances": int(mask.sum()),
                         "n_clusters": plan.support(mask) if plan else None,
                         "cluster_kind": cluster_kind if plan else "unsupported",
                         "n_native_languages": len(set(languages[mask])),
                         "gender_counts": dict(Counter(first[ids[i]].get("gender", "Unknown") for i in indexes)),
                         "duration_seconds": sum(float(first[ids[i]].get("duration_s") or 0) for i in indexes)})
    normalized_results = {}
    for normalizer_name, normalizer in normalizers.items():
        reference_texts = [normalizer(first[key]["reference_text"]) for key in ids]
        references = np.array([len(text.split()) for text in reference_texts], dtype=int)
        model_errors, metric_rows, subgroup_rows, diagnostic_rows = {}, [], [], []
        for system in sorted(systems):
            counts = [word_edit_counts(ref, normalizer(systems[system][key]["hypothesis_text"]))
                      for key, ref in zip(ids, reference_texts)]
            errors = np.array([count.errors for count in counts], dtype=int)
            model_errors[system] = errors
            for level, group, mask in selections:
                estimate = corpus_wer(errors[mask], references[mask])
                bootstrap = plan.ratios(errors * mask, references * mask) if plan else None
                n_clusters = plan.support(mask) if plan else None
                low, high, status = interval(bootstrap, n_clusters, cluster_kind)
                metric_rows.append({"system": system, "level": level, "group": group,
                                    "n_utterances": int(mask.sum()), "reference_words": int(references[mask].sum()),
                                    "errors": int(errors[mask].sum()), "wer_pct": estimate,
                                    "ci_low_pct": low, "ci_high_pct": high, "ci_status": status,
                                    "n_clusters": n_clusters})
            group_wers = [corpus_wer(errors[mask], references[mask]) for group, mask in group_masks.items()
                          if group != "non_Bodo"]
            tail = errors > references
            duplicate_counts = Counter(reference_texts)
            language_wers = {language: corpus_wer(errors[languages == language], references[languages == language])
                             for language in sorted(set(languages))}
            ranked_languages = sorted(language_wers, key=lambda language: (-language_wers[language], language))
            coarse_wers = {group: corpus_wer(errors[mask], references[mask]) for group, mask in group_masks.items()
                           if group != "non_Bodo"}
            diagnostics = {"system": system, "equal_three_group_mean_wer_pct": float(np.mean(group_wers)),
                           "three_group_wer_spread_pp": max(group_wers) - min(group_wers),
                           "substitutions": sum(count.substitutions for count in counts),
                           "deletions": sum(count.deletions for count in counts),
                           "insertions": sum(count.insertions for count in counts),
                           "zero_reference_rows": int((references == 0).sum()),
                           "unique_reference_texts": len(duplicate_counts),
                           "duplicate_reference_extra_rows": sum(value - 1 for value in duplicate_counts.values()),
                           "wer_over_100pct_rows": int(tail.sum()),
                           "error_fraction_in_over_100pct_rows": float(errors[tail].sum() / errors.sum()) if errors.sum() else 0,
                           "tail_excluded_wer_pct_sensitivity_only": corpus_wer(errors[~tail], references[~tail]),
                           "Bodo_native_language_rank_worst_first": 1 + sum(wer > language_wers["Bodo"] for wer in language_wers.values()),
                           "worst_native_language": ranked_languages[0],
                           "worst_native_language_wer_pct": language_wers[ranked_languages[0]],
                           "Bodo_wer_pct": language_wers["Bodo"],
                           "native_language_wer_ranking": ranked_languages,
                           "worst_coarse_group": max(coarse_wers, key=coarse_wers.get),
                           "poisson_diagnostic": poisson_diagnostic(errors, references, groups)}
            diagnostic_rows.append(diagnostics)
            system_contrasts = []
            for contrast, left, right in SUBGROUP_CONTRASTS:
                lm, rm = group_masks[left], group_masks[right]
                lwer, rwer = corpus_wer(errors[lm], references[lm]), corpus_wer(errors[rm], references[rm])
                difference = lwer - rwer if lwer is not None and rwer is not None else None
                samples = (plan.ratios(errors * lm, references * lm) -
                           plan.ratios(errors * rm, references * rm)) if plan else None
                row = _contrast_row(contrast, difference, samples,
                                    plan.support(lm) if plan else None, plan.support(rm) if plan else None,
                                    cluster_kind, minimum_test_clusters)
                row["system"] = system
                system_contrasts.append(row)
            for row, adjusted in zip(system_contrasts, holm_adjust(
                    [row["approximate_bootstrap_p"] for row in system_contrasts])):
                row["holm_adjusted_p"] = adjusted
            subgroup_rows.extend(system_contrasts)
        model_contrasts = []
        if BASELINE in model_errors:
            baseline = model_errors[BASELINE]
            baseline_wer = corpus_wer(baseline, references)
            for system, errors in model_errors.items():
                if system == BASELINE:
                    continue
                samples = (plan.ratios(errors, references) - plan.ratios(baseline, references)) if plan else None
                row = _contrast_row(f"{system}_minus_{BASELINE}", corpus_wer(errors, references) - baseline_wer,
                                    samples, len(plan.cluster_ids) if plan else None,
                                    len(plan.cluster_ids) if plan else None, cluster_kind, minimum_test_clusters)
                row["system"], row["baseline"] = system, BASELINE
                model_contrasts.append(row)
            for row, adjusted in zip(model_contrasts, holm_adjust(
                    [row["approximate_bootstrap_p"] for row in model_contrasts])):
                row["holm_adjusted_p"] = adjusted
        normalized_results[normalizer_name] = {"metrics": metric_rows, "subgroup_contrasts": subgroup_rows,
                                                "model_contrasts": model_contrasts, "diagnostics": diagnostic_rows}
    return {"name": name, "n_utterances": len(ids), "systems": sorted(systems),
            "utterance_ids": ids, "cluster_status": cluster_status, "support": supports,
            "normalizations": normalized_results}


def overlap_diagnostic(old: dict[str, dict], new: dict[str, dict]) -> list[dict]:
    output = []
    for system in sorted(set(old) & set(new)):
        ids = sorted(set(old[system]) & set(new[system]))
        metadata_conflicts = [key for key in ids if _signature(old[system][key]) != _signature(new[system][key])]
        if metadata_conflicts:
            raise ValueError(f"Historical/new metadata drift for {system}: {metadata_conflicts[:5]}")
        output.append({"system": system, "overlapping_utterances": len(ids),
                       "exact_hypothesis_matches": sum(old[system][key]["hypothesis_text"] ==
                                                        new[system][key]["hypothesis_text"] for key in ids),
                       "minimal_normalized_hypothesis_matches": sum(normalize_minimal(old[system][key]["hypothesis_text"]) ==
                            normalize_minimal(new[system][key]["hypothesis_text"]) for key in ids)})
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("audit/svarah_fairness_audit"))
    parser.add_argument("--matched-results-dir", type=Path)
    parser.add_argument("--full-results-dir", type=Path)
    parser.add_argument("--full-input-manifest", type=Path, default=Path("paper/data/full_input_manifest.json"))
    parser.add_argument("--input-manifest", type=Path, default=Path("paper/data/input_manifest.json"))
    parser.add_argument("--spelling-map", type=Path, default=Path("paper/data/whisper_normalizer.json"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--cluster-column", default="speaker_id")
    parser.add_argument("--cluster-kind", choices=("speaker", "recording", "demographic_profile", "none"), default="none")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--minimum-test-clusters", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/results"))
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("Use at least 100 resamples; the paper protocol specifies 2,000")
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    spelling = json.loads(args.spelling_map.read_text())
    normalizers = {"minimal": normalize_minimal, "whisper_english": EnglishTextNormalizer(spelling)}
    original, provenance = load_predictions(args.results_dir)
    manifest = json.loads(args.input_manifest.read_text())
    validate_frozen_inputs(original, provenance, manifest)
    metadata, metadata_provenance = load_metadata(args.metadata, args.cluster_column)
    validation = {"historical": historical_validation(original)}
    open_systems = {key: value for key, value in original.items() if "saaras" not in key.lower()}
    open_ids = sorted(set.intersection(*(set(rows) for rows in open_systems.values())))
    panels = [("historical_open_2500", open_systems, open_ids)]
    common_ids = sorted(set.intersection(*(set(rows) for rows in original.values())))
    if any("saaras" in key.lower() for key in original):
        panels.append(("historical_six_overlap_provisional", original, common_ids))
        saaras = {key: value for key, value in original.items() if "saaras" in key.lower()}
        panels.append(("historical_saaras_450", saaras, sorted(set.intersection(*(set(rows) for rows in saaras.values())))))
    reproduction, full_historical_overlap, full_matched_overlap = [], [], []
    full_manifest_provenance = None
    matched = None
    if args.matched_results_dir:
        matched, matched_provenance = load_predictions(args.matched_results_dir)
        provenance += matched_provenance
        validation["matched"] = historical_validation(matched)
        expected_ids = set(next(rows for key, rows in original.items() if "saaras" in key.lower()))
        if set(matched) != set(original):
            raise ValueError(f"Matched panel systems differ: expected {sorted(original)}, got {sorted(matched)}")
        for system, rows in matched.items():
            if set(rows) != expected_ids:
                raise ValueError(f"Matched {system} is not precisely the original Saaras cohort")
        validate_unchanged_saaras(original, matched)
        panels.append(("matched_six_450", matched, sorted(expected_ids)))
        reproduction = overlap_diagnostic(original, matched)
    if args.full_results_dir:
        full_manifest = json.loads(args.full_input_manifest.read_text())
        if args.metadata is None or full_manifest["metadata_sha256"] != hash_file(args.metadata):
            raise ValueError("Full-corpus metadata differ from the frozen full-input manifest")
        frozen_full_rows = {row["utt_id"]: row for row in full_manifest["manifest"]}
        if len(frozen_full_rows) != 6656 or full_manifest["n_utterances"] != 6656 or set(frozen_full_rows) != set(metadata):
            raise ValueError("Full-input manifest no longer enumerates all pinned metadata IDs")
        if {model.split("/")[-1] for model in full_manifest["model_revisions"]} != set(open_systems):
            raise ValueError("Full-input model revisions differ from the five frozen open models")
        for utt_id, frozen_row in frozen_full_rows.items():
            validate_metadata_identity(metadata[utt_id], frozen_row, utt_id)
        full, full_provenance = load_predictions(args.full_results_dir)
        validate_full_panel(full, set(open_systems), metadata)
        provenance += full_provenance
        validation["full"] = historical_validation(full)
        panels.append(("full_open_6656", full, sorted(metadata)))
        full_historical_overlap = overlap_diagnostic(open_systems, full)
        if matched is not None:
            full_matched_overlap = overlap_diagnostic(matched, full)
        full_manifest_provenance = {"path": str(args.full_input_manifest.resolve()),
                                    "sha256": hash_file(args.full_input_manifest),
                                    "dataset_revision": full_manifest["dataset_revision"],
                                    "n_utterances": full_manifest["n_utterances"]}
    result = {"protocol": str(Path(__file__).with_name("PROTOCOL.md").resolve()),
              "protocol_sha256": hash_file(Path(__file__).with_name("PROTOCOL.md")),
              "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "inputs": provenance, "metadata": metadata_provenance, "validation": validation,
              "frozen_manifest": {"path": str(args.input_manifest.resolve()), "sha256": hash_file(args.input_manifest)},
              "full_frozen_manifest": full_manifest_provenance,
              "normalizer": {"transformers_version": importlib.metadata.version("transformers"),
                             "spelling_map_path": str(args.spelling_map.resolve()),
                             "spelling_map_sha256": hash_file(args.spelling_map)},
              "historical_new_overlap": reproduction,
              "full_historical_overlap": full_historical_overlap,
              "full_matched_overlap": full_matched_overlap, "cohorts": {}}
    for name, systems, ids in panels:
        print(f"Analyzing {name}: {len(ids)} utterances, {len(systems)} systems", flush=True)
        result["cohorts"][name] = analyze_panel(name, systems, ids, normalizers, metadata,
            args.cluster_column, args.cluster_kind, args.resamples, args.seed, args.minimum_test_clusters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    metrics, contrasts, support, diagnostics = [], [], [], []
    for cohort, panel in result["cohorts"].items():
        support += [{"cohort": cohort, **row} for row in panel["support"]]
        for normalizer, outputs in panel["normalizations"].items():
            prefix = {"cohort": cohort, "normalizer": normalizer}
            metrics += [{**prefix, **row} for row in outputs["metrics"]]
            contrasts += [{**prefix, "contrast_type": kind, **row}
                          for kind in ("subgroup_contrasts", "model_contrasts") for row in outputs[kind]]
            diagnostics += [{**prefix, **row} for row in outputs["diagnostics"]]
    for filename, rows in (("metrics.csv", metrics), ("contrasts.csv", contrasts), ("support.csv", support)):
        _write_csv(args.output_dir / filename, rows)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output_dir / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
