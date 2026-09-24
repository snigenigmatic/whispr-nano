"""Compare primary and resource-panel texts on the identical twelve audio files."""
import hashlib
import json
from pathlib import Path

from research.bridge_open.scoring import normalize_v1
from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "research/bridge_open/artifacts"
PRIMARY = {
    "whisper_small": "core_whisper_small_svarah_v1",
    "whisper_turbo": "core_whisper_turbo_svarah_v1",
    "whisper_large_v3": "core_whisper_large_svarah_v1",
    "qwen_0_6b": "core_qwen06_svarah_v1",
    "qwen_1_7b": "qwen_svarah_full_v1",
    "omni": "omni_svarah_full_v1",
}


def compare():
    models = {}
    for key, directory in PRIMARY.items():
        prediction_path = ART / directory / "predictions.jsonl"
        primary_rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        primary = {row["conversation_id"]: row for row in primary_rows}
        if len(primary) != len(primary_rows):
            raise ValueError("Duplicate primary IDs")
        profile_path = (ART / "core_resource_omni_v1/resource_profile.json" if key == "omni"
                        else ART / f"core_resource_whisper_qwen_v1/{key}_resource_profile.json")
        calls = [row for row in json.loads(profile_path.read_text())["timed_calls"] if row["pass_index"] == 3]
        if tuple(row["conversation_id"] for row in calls) != FIXED_PANEL_IDS:
            raise ValueError("Different resource panel")
        raw_different, normalized_different = [], []
        errors = words = 0
        for call in calls:
            row = primary[call["conversation_id"]]
            audio_hash = row.get("audio_sha256", row.get("evidence", {}).get("input_audio_sha256"))
            if audio_hash != call["audio_sha256"]:
                raise ValueError("Different primary/resource audio")
            if row["text_raw"] != call["text"]:
                raw_different.append(call["conversation_id"])
            if normalize_v1(row["text_raw"]) != normalize_v1(call["text"]):
                normalized_different.append(call["conversation_id"])
            errors += sum(row["metrics"][kind] for kind in ("substitutions", "deletions", "insertions"))
            words += row["metrics"]["ref_words"]
        models[key] = {
            "raw_text_different_ids": raw_different,
            "normalized_text_different_ids": normalized_different,
            "primary_panel_word_errors": errors, "primary_panel_reference_words": words,
            "primary_panel_wer": errors / words,
            "primary_predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "resource_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        }
    return {"scope": "Post-result exact-ID/audio-hash comparison; no independent samples added. Resource and primary conditions remain separately reported.", "models": models}


if __name__ == "__main__":
    result = compare()
    Path(__file__).with_name("primary_panel_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: len(value["raw_text_different_ids"]) for key, value in result["models"].items()}))
