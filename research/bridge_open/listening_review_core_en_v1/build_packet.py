"""Build the separate, blinded English review packet for the fixed Svarah resource panel.

The only transcripts released here are pass-three outputs from the six already
accepted batch-one resource profiles.  This script adds no recordings, ratings,
or independent observations.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.bridge_open.core_study import accept_resources
from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS, read_fixed_panel


OUT = ROOT / "research/bridge_open/listening_review_core_en_v1"
MANIFEST = ROOT / "data/bridge_open/svarah_native_shard02_v2/bundle/manifest.jsonl"
BUNDLE = MANIFEST.parent
OMNI_RUN = ROOT / "research/bridge_open/artifacts/core_resource_omni_v1"
SUITE_RUN = ROOT / "research/bridge_open/artifacts/core_resource_whisper_qwen_v1"
SEED = 20260924
LABELS = ("system_A", "system_B", "system_C", "system_D", "system_E", "system_F")
PROFILE_SOURCES = (
    ("omni", "OmniASR CTC 1B", OMNI_RUN / "resource_profile.json"),
    ("whisper_small", "Whisper Small", SUITE_RUN / "whisper_small_resource_profile.json"),
    ("whisper_turbo", "Whisper Turbo", SUITE_RUN / "whisper_turbo_resource_profile.json"),
    ("whisper_large_v3", "Whisper Large-v3", SUITE_RUN / "whisper_large_v3_resource_profile.json"),
    ("qwen_0_6b", "Qwen3-ASR 0.6B", SUITE_RUN / "qwen_0_6b_resource_profile.json"),
    ("qwen_1_7b", "Qwen3-ASR 1.7B", SUITE_RUN / "qwen_1_7b_resource_profile.json"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _final_pass(profile_path: Path) -> dict[str, dict[str, Any]]:
    profile = _load_json(profile_path)
    calls = [call for call in profile["timed_calls"] if call.get("pass_index") == 3]
    if [call.get("conversation_id") for call in calls] != list(FIXED_PANEL_IDS):
        raise ValueError(f"{profile_path}: pass-three panel differs from the accepted fixed order")
    if any(call.get("status") != "success" or not isinstance(call.get("text"), str) for call in calls):
        raise ValueError(f"{profile_path}: a final-pass output is missing or failed")
    for call in calls:
        if call.get("text_sha256") != hashlib.sha256(call["text"].encode("utf-8")).hexdigest():
            raise ValueError(f"{profile_path}: final-pass transcript hash mismatch for {call['conversation_id']}")
    return {call["conversation_id"]: call for call in calls}


def build() -> dict[str, Any]:
    """Validate accepted sources and build public and sealed artifacts in this directory."""
    accepted = accept_resources.validate(OMNI_RUN, SUITE_RUN)
    if accepted.get("valid") is not True:
        raise ValueError("resource profiles are not accepted")
    rows = read_fixed_panel()
    if tuple(row["conversation_id"] for row in rows) != FIXED_PANEL_IDS:
        raise ValueError("source manifest differs from the fixed resource panel")
    final_outputs = {key: _final_pass(path) for key, _name, path in PROFILE_SOURCES}

    public_items: list[dict[str, Any]] = []
    sealed_mapping: list[dict[str, Any]] = []
    for source in rows:
        review_id = source["conversation_id"]
        audio = BUNDLE / "wav" / Path(source["audio_path"]).name
        reference = BUNDLE / "references" / Path(source["reference_path"]).name
        if _sha256(audio) != source["audio_sha256"] or _sha256(reference) != source["reference_sha256"]:
            raise ValueError(f"{review_id}: source audio or reference hash differs from accepted manifest")
        reference_payload = _load_json(reference)
        labels = list(LABELS)
        random.Random(f"{SEED}:{review_id}").shuffle(labels)
        systems: list[dict[str, str]] = []
        for label, (key, model_name, profile_path) in zip(labels, PROFILE_SOURCES, strict=True):
            output = final_outputs[key][review_id]
            systems.append({"system_label": label, "transcript": output["text"]})
            sealed_mapping.append(
                {
                    "review_id": review_id,
                    "system_label": label,
                    "system_name": model_name,
                    "profile_source": str(profile_path.relative_to(ROOT)),
                    "timed_pass_index": 3,
                    "transcript_sha256": output["text_sha256"],
                    "audio_sha256": output["audio_sha256"],
                }
            )
        # The label assignment is deliberately randomized, but the public
        # order is label-only so a card position cannot reveal a source model.
        systems.sort(key=lambda card: card["system_label"])
        public_items.append(
            {
                "review_id": review_id,
                "selection_kinds": ["preexposed_resource_panel"],
                "language": "en",
                "exposure_scope": "preexposed_exploratory_corpus",
                "audio_url": "../../../data/bridge_open/svarah_native_shard02_v2/bundle/wav/" + audio.name,
                "audio_path_relative": "data/bridge_open/svarah_native_shard02_v2/bundle/wav/" + audio.name,
                "audio_sha256": source["audio_sha256"],
                "reference_sha256": source["reference_sha256"],
                "reference_text": reference_payload["reference_text"],
                "systems": systems,
                "review_status": "UNREVIEWED",
                "reviewer_response": {
                    "reference_correct": "",
                    "audio_quality": "",
                    "language_or_script_by_system": {},
                    "meaning_preserved_by_system": {},
                    "notes": "",
                },
            }
        )
    if len(public_items) != 12 or len(sealed_mapping) != 72:
        raise ValueError("English extension must contain 12 items and 72 anonymous cards")

    _write_json(OUT / "review_items.json", public_items)
    (OUT / "review_items.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in public_items) + "\n", encoding="utf-8"
    )
    _write_json(
        OUT / "sealed_mapping.json",
        {
            "warning": "SEALED: do not share with blinded reviewer",
            "packet_name": "listening_review_core_en_v1",
            "seed": SEED,
            "mapping": sealed_mapping,
        },
    )
    _write_json(
        OUT / "packet_metadata.json",
        {
            "packet_name": "listening_review_core_en_v1",
            "packet_status": "UNREVIEWED",
            "fixed_panel_source": "svarah_native_shard02_v2_fixed_12",
            "fixed_panel_manifest_sha256": accepted["panel_manifest_sha256"],
            "item_count": 12,
            "anonymous_card_count": 72,
            "language_counts": {"en": 12},
            "source_outputs": "accepted batch-one resource-profile timed pass 3 only",
            "source_profiles_accepted": True,
            "source_profile_sha256": {key: _sha256(path) for key, _name, path in PROFILE_SOURCES},
            "preexposure_note": "The fixed 12-item Svarah panel is pre-exposed and adds no independent observations.",
            "comparability_note": "These batch-one resource-profile transcripts are separate from primary batch-eight quality conditions.",
            "reviewer_view_excludes_true_system_names_and_scores": True,
        },
    )
    return {
        "items": len(public_items),
        "anonymous_cards": len(sealed_mapping),
        "all_ratings_pending": all(item["review_status"] == "UNREVIEWED" for item in public_items),
        "accepted_sources": accepted["valid"],
    }


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
