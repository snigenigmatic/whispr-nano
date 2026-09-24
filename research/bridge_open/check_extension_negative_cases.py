"""Corrupt copies of real evidence to verify rejection without changing originals."""
import json
import tempfile
from pathlib import Path
from research.bridge_open.analyze_omni_extension import validate as validate_omni
from research.bridge_open.analyze_language_conditioning import validate as validate_language

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research/bridge_open/artifacts'


def check(source, validator, name, mutate, file='predictions.jsonl'):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        for original in source.iterdir():
            if original.is_file(): (p / original.name).symlink_to(original.resolve())
        target = p / file
        data = ([json.loads(x) for x in target.read_text().splitlines()] if file.endswith('.jsonl')
                else json.loads(target.read_text()))
        mutate(data)
        target.unlink()
        target.write_text('\n'.join(json.dumps(x) for x in data) + '\n' if file.endswith('.jsonl')
                          else json.dumps(data))
        try:
            validator(p)
        except (AssertionError, ValueError, KeyError):
            return dict(case=name, rejected=True)
        raise AssertionError('Invalid evidence accepted: ' + name)


def main():
    omni = ART / 'omni_fleurs_full_as_v1'
    language = ART / 'whisper_language_conditioning_v1'
    checks = [
        check(omni, validate_omni, 'omni_missing_record', lambda d: d.pop()),
        check(omni, validate_omni, 'omni_wrong_source_hash', lambda d: d[0].update(audio_sha256='0'*64)),
        check(omni, validate_omni, 'omni_corrupt_ctc_collapse', lambda d: d[0]['chunks'][0]['ctc_collapsed_token_ids'].append(987654)),
        check(omni, validate_omni, 'omni_training_mode', lambda d: d['runtime_evidence'].update(model_training=True), 'environment.json'),
        check(language, validate_language, 'language_missing_pair', lambda d: d.pop()),
        check(language, validate_language, 'language_wrong_arm_options', lambda d: d[0]['effective_options'].update(language=None)),
        check(language, validate_language, 'language_fabricated_score', lambda d: d[0]['metrics'].update(wer=999)),
        check(language, validate_language, 'language_wrong_precision', lambda d: d.update(actual_dtype='torch.float16'), 'environment.json'),
    ]
    p = ART / 'expanded_study_v1/negative_validation_checks.json'
    p.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__': main()
