"""Freeze a reference-supervised intervention on teacher-based data selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jiwer import process_words
from paper.analysis.analyze import normalize_minimal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_manifest(rows):
    ids, partitions = set(), defaultdict(lambda: defaultdict(set))
    speaker_languages = {}
    for r in rows:
        if r['utt_id'] in ids:
            raise ValueError('Duplicate utterance ID')
        ids.add(r['utt_id'])
        if r['split'] not in ('train', 'dev', 'test'):
            raise ValueError('Only primary speaker/content-disjoint sets allowed')
        if not r['speaker_id'] or not r['prompt_id'] or not r['reference_text'].strip():
            raise ValueError('Identity or reference missing')
        if r['speaker_id'] in speaker_languages and speaker_languages[r['speaker_id']] != r['native_language']:
            raise ValueError('Inconsistent native language for one speaker')
        speaker_languages[r['speaker_id']] = r['native_language']
        if not 0 < float(r['duration_s']) <= 30:
            raise ValueError('Unexpected duration: automatic truncation prohibited')
        for key, value in [('speaker', r['speaker_id']), ('prompt', r['prompt_id']),
                           ('reference', normalize_minimal(r['reference_text']))]:
            partitions[key][r['split']].add(value)
    for key, sets in partitions.items():
        for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
            if sets[a] & sets[b]:
                raise ValueError(f'{key} leakage between {a} and {b}')
    return {'rows': len(rows), 'counts': {s: sum(r['split'] == s for r in rows)
                                        for s in ('train', 'dev', 'test')},
            'speaker_counts': {s: len(partitions['speaker'][s]) for s in ('train', 'dev', 'test')},
            'speaker_content_disjoint': True}


def length_bin(n):
    return 0 if n < 5 else 1 if n < 10 else 2 if n < 20 else 3


def duration_matched_random(pool, n, target_seconds, rng):
    """Random initialization plus sum-duration swaps, never reads teacher errors.

    Exact speaker/word-bin counts are maintained. Duration matching is approximate
    and its achieved balance is reported. Rejected examples remain eligible.
    """
    if n == 0:
        return []
    if n == len(pool):
        return list(pool)
    chosen = set(int(x) for x in rng.choice(len(pool), n, replace=False))
    durations = np.array([float(r['duration_s']) for r in pool])
    current = sum(durations[i] for i in chosen)
    # Match total duration to within0.5% or0.05s, preserving randomness in
    # composition instead of matching each filtered row to itself.
    tolerance = max(0.05, target_seconds * 0.005)
    for _ in range(300):
        gap = target_seconds - current
        if abs(gap) <= tolerance:
            break
        inside = np.array(sorted(chosen))
        outside = np.array(sorted(set(range(len(pool))) - chosen))
        sample_in = rng.choice(inside, min(32, len(inside)), replace=False)
        sample_out = rng.choice(outside, min(64, len(outside)), replace=False)
        delta = durations[sample_out][None, :] - durations[sample_in][:, None]
        i, j = np.unravel_index(np.argmin(abs(gap - delta)), delta.shape)
        change = float(delta[i, j])
        if abs(gap - change) >= abs(gap):
            break
        chosen.remove(int(sample_in[i])); chosen.add(int(sample_out[j])); current += change
    return [pool[i] for i in sorted(chosen)]


def build(manifest, predictions, normalizer, seed=20260909, threshold=10):
    validation = validate_manifest(manifest)
    train = sorted([r for r in manifest if r['split'] == 'train'], key=lambda r: r['utt_id'])
    indexed = {}
    for r in predictions:
        if r['utt_id'] in indexed:
            raise ValueError('Duplicate teacher prediction ID')
        indexed[r['utt_id']] = r
    if set(indexed) != {r['utt_id'] for r in train}:
        raise ValueError('Teacher prediction IDs must cover train exactly; dev/test scoring prohibited here')
    scored, strata = [], defaultdict(list)
    for r in train:
        p = indexed[r['utt_id']]
        if p.get('reference_text', r['reference_text']) != r['reference_text']:
            raise ValueError('Teacher reference differs from frozen manifest')
        ref, hyp = normalizer(r['reference_text']), normalizer(p['hypothesis_text'])
        n = len(ref.split())
        if not n:
            raise ValueError('Zero-word normalized reference')
        a = process_words(ref, hyp)
        errors = a.substitutions + a.deletions + a.insertions
        row = {**r, 'teacher_errors': errors, 'reference_words': n,
               'teacher_wer_pct': 100 * errors / n,
               'accepted': 100 * errors <= threshold * n, 'length_bin': length_bin(n)}
        scored.append(row)
        strata[(r['speaker_id'], row['length_bin'])].append(row)
    rng = np.random.default_rng(seed)
    random, balances = [], []
    filtered = [r for r in scored if r['accepted']]
    if not filtered:
        raise ValueError('Teacher filter retained zero training examples')
    for (speaker, b), pool in sorted(strata.items()):
        accepted = [r for r in pool if r['accepted']]
        target_seconds = sum(float(r['duration_s']) for r in accepted)
        control = duration_matched_random(pool, len(accepted), target_seconds, rng)
        assert len(control) == len(accepted)
        random.extend(control)
        control_seconds = sum(float(r['duration_s']) for r in control)
        balances.append({'speaker_id': speaker, 'length_bin': b,
                         'native_language': pool[0]['native_language'], 'candidate_n': len(pool),
                         'filtered_n': len(accepted), 'random_n': len(control),
                         'filtered_seconds': target_seconds,
                         'random_seconds': control_seconds,
                         'duration_match_within_tolerance': abs(control_seconds - target_seconds) <= max(0.05, target_seconds * 0.005),
                         'overlap_with_filtered_n': len({r['utt_id'] for r in accepted} & {r['utt_id'] for r in control}),
                         'random_includes_rejected_n': sum(not r['accepted'] for r in control)})
    selected = {'full_gold': scored, 'filtered10_gold': filtered, 'random_matched_gold': random}
    details = {}
    for name, rows in selected.items():
        counts = defaultdict(int)
        for r in rows:
            counts[r['native_language']] += 1
        details[name] = {'n': len(rows), 'hours': sum(float(r['duration_s']) for r in rows) / 3600,
                         'native_language_counts': dict(counts),
                         'teacher_wer_pct': 100 * sum(r['teacher_errors'] for r in rows) / sum(r['reference_words'] for r in rows),
                         'teacher_rejected_included_n': sum(not r['accepted'] for r in rows)}
    return selected, {'manifest_validation': validation, 'seed': seed, 'threshold_pct': threshold,
                      'status': 'selection_frozen_before_training', 'target_source': 'provided_reference_in_every_arm',
                      'matching': 'exact speaker x reference-word-bin counts; approximate duration; errors never used to optimize random control',
                      'arms': details, 'stratum_balance': balances}, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('research/pilot/data/manifest.jsonl'))
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('research/pilot/selections'))
    args = parser.parse_args()
    from research.pilot.frozen_english_normalizer import EnglishTextNormalizer
    map_path = Path('paper/data/whisper_normalizer.json')
    selections, report, scores = build(load_jsonl(args.manifest), load_jsonl(args.predictions),
                                       EnglishTextNormalizer(json.loads(map_path.read_text())))
    report['manifest_sha256'] = sha(args.manifest)
    report['teacher_predictions_sha256'] = sha(args.predictions)
    report['normalizer_map_sha256'] = sha(map_path)
    import inspect, importlib.metadata
    report['normalizer_implementation_sha256'] = sha(Path(inspect.getfile(EnglishTextNormalizer)))
    report['selection_code_sha256'] = sha(Path(__file__))
    report['runtime_versions'] = {p: importlib.metadata.version(p) for p in ['numpy', 'jiwer', 'regex']}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in {**selections, 'teacher_scored_train': scores}.items():
        path = args.output / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        report.setdefault('output_sha256', {})[name] = sha(path)
    (args.output / 'selection_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['arms'], indent=2))


if __name__ == '__main__':
    main()
