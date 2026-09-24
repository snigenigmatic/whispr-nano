"""Build the static core-study evidence viewer from independently accepted cells.

The page embeds transcript evidence only.  Audio remains in the verified local
corpus and is linked relative to the generated artifact, so the viewer works
offline when opened inside this workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
QUALITY_RESULTS = ROOT / "research/bridge_open/core_study/quality_results.json"
OUTPUT = ROOT / "research/bridge_open/artifacts/core_study_v1/index.html"
FLEURS_MANIFEST = ROOT / "data/bridge_open/fleurs_test_v1/corpus-v2.jsonl"
SVARAH_MANIFEST = ROOT / "data/bridge_open/svarah_native_shard02_full_v1/bundle/manifest.jsonl"

TRACKS = {
    "fleurs-hi": {"label": "FLEURS Hindi", "language": "hi"},
    "fleurs-bn": {"label": "FLEURS Bengali", "language": "bn"},
    "fleurs-as": {"label": "FLEURS Assamese", "language": "as"},
    "svarah-recovered-en": {"label": "Svarah English", "language": "en"},
}
MODELS = {
    "whisper-small": "Whisper Small",
    "whisper-turbo": "Whisper Turbo",
    "whisper-large-v3": "Whisper Large-v3",
    "qwen3-asr-0.6b": "Qwen3-ASR 0.6B",
    "qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
    "omniasr-ctc-1b": "OmniASR CTC-1B",
}
SUPPORTED_TRACKS = {
    "whisper-small": set(TRACKS),
    "whisper-turbo": set(TRACKS),
    "whisper-large-v3": set(TRACKS),
    "qwen3-asr-0.6b": {"fleurs-hi", "svarah-recovered-en"},
    "qwen3-asr-1.7b": {"fleurs-hi", "svarah-recovered-en"},
    "omniasr-ctc-1b": set(TRACKS),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_value(prediction: dict[str, Any], key: str) -> Any:
    if key in prediction:
        return prediction[key]
    evidence = prediction.get("evidence")
    if isinstance(evidence, dict):
        return evidence.get("input_audio_sha256" if key == "audio_sha256" else key)
    return None


def source_rows() -> list[dict[str, Any]]:
    fleurs = jsonl(FLEURS_MANIFEST)
    rows = []
    for track, metadata in TRACKS.items():
        if track.startswith("fleurs-"):
            rows.extend({**row, "_track": track} for row in fleurs if row["language"] == metadata["language"])
    rows.extend({**row, "_track": "svarah-recovered-en"} for row in jsonl(SVARAH_MANIFEST))
    ids = [row["conversation_id"] for row in rows]
    require(len(rows) == 2840 and len(ids) == len(set(ids)), "expected exactly 2,840 unique FLEURS and Svarah recordings")
    return rows


def reference_path(source: dict[str, Any]) -> Path:
    value = Path(source["reference_path"])
    if source["_track"] == "svarah-recovered-en":
        return SVARAH_MANIFEST.parent / "references" / value.name
    return value if value.is_absolute() else ROOT / value


def audio_path(source: dict[str, Any]) -> Path:
    value = Path(source["audio_path"])
    if source["_track"] == "svarah-recovered-en":
        return SVARAH_MANIFEST.parent / "wav" / value.name
    return value if value.is_absolute() else ROOT / value


def accepted_predictions(results: dict[str, Any], sources: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    require(results.get("schema") == "core-asr-quality-results/v1", "unexpected quality summary schema")
    accepted: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cell in results.get("cells", []):
        if cell.get("status") != "accepted":
            continue
        model, track = cell.get("model"), cell.get("track")
        require(model in MODELS and track in TRACKS, "quality summary has an unknown model or track")
        prediction_path = ROOT / cell["source_predictions"]
        receipt_path = ROOT / cell["acceptance_receipt"]
        require(prediction_path.is_file() and digest(prediction_path) == cell["source_predictions_sha256"],
                f"accepted prediction hash differs: {model} {track}")
        require(receipt_path.is_file() and digest(receipt_path) == cell["acceptance_receipt_sha256"],
                f"accepted receipt hash differs: {model} {track}")
        cohort = {cid: row for cid, row in sources.items() if row["_track"] == track}
        rows = [row for row in jsonl(prediction_path) if row.get("conversation_id") in cohort]
        ids = [row.get("conversation_id") for row in rows]
        require(len(rows) == cell["records"] and len(ids) == len(set(ids)) and set(ids) == set(cohort),
                f"accepted prediction cohort differs from source: {model} {track}")
        rendered: dict[str, dict[str, Any]] = {}
        for prediction in rows:
            source = cohort[prediction["conversation_id"]]
            require(source_value(prediction, "audio_sha256") == source["audio_sha256"],
                    f"accepted prediction audio hash differs: {prediction['conversation_id']}")
            require(source_value(prediction, "reference_sha256") == source["reference_sha256"],
                    f"accepted prediction reference hash differs: {prediction['conversation_id']}")
            require(isinstance(prediction.get("text_raw"), str) and isinstance(prediction.get("metrics"), dict),
                    f"accepted prediction lacks text or metrics: {prediction['conversation_id']}")
            rendered[prediction["conversation_id"]] = {
                "availability": "accepted",
                "status": str(prediction.get("status", "unknown")),
                "text": prediction["text_raw"],
                "metrics": prediction["metrics"],
                "route": str(prediction.get("route", prediction.get("inference_route", prediction.get("condition_id", "unknown")))),
                "prediction_sha256": cell["source_predictions_sha256"],
                "acceptance_sha256": cell["acceptance_receipt_sha256"],
            }
        accepted[(model, track)] = rendered
    return accepted


def build(output: Path = OUTPUT) -> dict[str, int]:
    results = json.loads(QUALITY_RESULTS.read_text())
    rows = source_rows()
    sources = {row["conversation_id"]: row for row in rows}
    accepted = accepted_predictions(results, sources)
    pending = {(cell["model"], cell["track"]): cell for cell in results["cells"] if cell["status"] == "pending"}
    items = []
    for source in rows:
        ref_path = reference_path(source)
        audio = audio_path(source)
        require(ref_path.is_file() and digest(ref_path) == source["reference_sha256"],
                f"reference bytes/hash differ: {source['conversation_id']}")
        require(audio.is_file() and digest(audio) == source["audio_sha256"],
                f"audio bytes/hash differ: {source['conversation_id']}")
        reference = json.loads(ref_path.read_text()).get("reference_text")
        require(isinstance(reference, str), f"reference is malformed: {source['conversation_id']}")
        systems = {}
        for model in MODELS:
            key = (model, source["_track"])
            if source["_track"] not in SUPPORTED_TRACKS[model]:
                systems[model] = {"availability": "unsupported", "status": "unsupported for this track"}
            elif key in accepted:
                systems[model] = accepted[key][source["conversation_id"]]
            else:
                cell = pending.get(key)
                require(cell is not None, f"supported model/track is missing from quality summary: {model} {source['_track']}")
                systems[model] = {"availability": "pending", "status": "pending independent acceptance", "reason": cell.get("reason", "accepted prediction evidence is not yet available")}
        items.append({
            "id": source["conversation_id"], "track": source["_track"], "language": source["language"],
            "duration": source["duration_seconds"], "audio": os.path.relpath(audio, output.parent),
            "audio_sha256": source["audio_sha256"], "reference": reference,
            "reference_sha256": source["reference_sha256"], "systems": systems,
        })
    payload = {
        "items": items, "tracks": TRACKS, "models": MODELS,
        "accepted_cells": results["accepted_cell_count"], "pending_cells": results["pending_cell_count"],
        "quality_summary_sha256": digest(QUALITY_RESULTS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output.write_text(HTML.replace("PAYLOAD", safe_payload))
    return {"recordings": len(items), "accepted_cells": results["accepted_cell_count"], "pending_cells": results["pending_cell_count"]}


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Core ASR evidence viewer</title><style>
:root{font:16px/1.48 system-ui,sans-serif;color:#172936;background:#f5f7f8}body{max-width:1320px;margin:auto;padding:24px}h1{margin:0;font-size:29px}h2{font-size:20px}h3{margin:.1rem 0}.muted{color:#536776}.notice{border-left:4px solid #08767a;background:#e7f2f1;padding:12px 16px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}label{display:flex;flex-direction:column;gap:3px}select,input{font:inherit;padding:8px;border:1px solid #adbdc7;border-radius:6px;background:#fff;min-width:210px}article,.reference{background:#fff;border:1px solid #d1dce1;border-radius:8px;padding:15px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:12px}.text{white-space:pre-wrap;overflow-wrap:anywhere}.metric{font-weight:650}.badge{font-size:12px;padding:3px 7px;border-radius:4px;background:#e9eff4}.pending{background:#f7ecd0}.unsupported{background:#e9e9e9}.hash{font:12px/1.4 ui-monospace,monospace;overflow-wrap:anywhere}audio{width:min(700px,100%)}a{color:#006b70}details{margin-top:12px}footer{margin:32px 0 4px}
</style><body><h1>Core ASR evidence viewer</h1>
<p id="scope" class="muted"></p><p class="notice">Each displayed model output comes only from a cell with local independent acceptance. Pending cells deliberately have no transcript or metric. “Unsupported” means that model was not evaluated on that track.</p>
<p class="notice">The resource panel is separate from primary quality evidence: <a href="../core_resource_omni_v1/resource_profile.json">Omni batch-one resource profile</a> · <a href="../core_resource_whisper_qwen_v1/overall_summary.json">five-model batch-one resource summary</a> · <a href="../../core_study/RESOURCE_RESULTS.md">resource interpretation</a>.</p>
<div class="controls"><label>Track<select id="track"></select></label><label>Search recordings, references, or accepted outputs<input id="search" type="search" placeholder="ID or transcript text"></label><label>Recording<select id="clip"></select></label></div><p id="count" class="muted"></p><main id="detail"></main>
<footer class="muted">Sources are the frozen FLEURS official-test and recovered Svarah manifests. Audio playback uses the complete local waveform. Text is rendered with text nodes; the embedded payload escapes HTML-significant characters. This viewer reports saved evidence and does not rank systems or combine resource timing with primary quality.</footer>
<script type="application/json" id="payload">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id);let visible=[];
function node(tag,value,klass){const e=document.createElement(tag);e.textContent=value;if(klass)e.className=klass;return e}
function percent(value){return typeof value==='number'?(value*100).toFixed(2)+'%':'—'}
function renderSystem(model,s){const card=document.createElement('article');card.append(node('h3',data.models[model]));const badge=node('span',s.status,'badge '+(s.availability==='accepted'?'':s.availability));card.append(badge);if(s.availability!=='accepted'){card.append(node('p',s.reason||'No output is present for this track.','muted'));return card}const m=s.metrics||{};card.append(node('p','WER '+percent(m.wer)+' · CER '+percent(m.cer),'metric'),node('p','S '+(m.substitutions??'—')+' · D '+(m.deletions??'—')+' · I '+(m.insertions??'—'),'muted'),node('p',s.route,'muted'),node('p',s.text||'(empty output)','text'));const evidence=document.createElement('details');evidence.append(node('summary','Accepted evidence hashes'),node('p','Prediction SHA-256: '+s.prediction_sha256,'hash'),node('p','Acceptance receipt SHA-256: '+s.acceptance_sha256,'hash'));card.append(evidence);return card}
function render(){const item=data.items.find(x=>x.id===$('clip').value),detail=$('detail');detail.replaceChildren();if(!item)return;detail.append(node('h2',item.id),node('p',data.tracks[item.track].label+' · '+item.duration.toFixed(2)+' seconds · complete local waveform','muted'));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=item.audio;detail.append(audio);const reference=node('section','','reference');reference.append(node('h3','Dataset reference'),node('p',item.reference,'text'));detail.append(reference);const grid=node('section','','grid');for(const model of Object.keys(data.models))grid.append(renderSystem(model,item.systems[model]));detail.append(grid);const hashes=document.createElement('details');hashes.append(node('summary','Fixed source hashes'),node('p','Audio SHA-256: '+item.audio_sha256,'hash'),node('p','Reference SHA-256: '+item.reference_sha256,'hash'));detail.append(hashes)}
function searchable(item){return [item.id,item.reference,...Object.values(item.systems).filter(s=>s.availability==='accepted').map(s=>s.text)].join('\n').toLocaleLowerCase()}
function filter(){const track=$('track').value,query=$('search').value.toLocaleLowerCase(),old=$('clip').value;visible=data.items.filter(item=>(track==='all'||item.track===track)&&searchable(item).includes(query));$('clip').replaceChildren();for(const item of visible){const option=node('option',item.id+' · '+item.duration.toFixed(2)+' s');option.value=item.id;$('clip').append(option)}if(visible.some(item=>item.id===old))$('clip').value=old;$('count').textContent=visible.length+' recordings shown';render()}
const all=node('option','All tracks · 2,840 recordings');all.value='all';$('track').append(all);for(const [key,track] of Object.entries(data.tracks)){const option=node('option',track.label);option.value=key;$('track').append(option)}const completeness=data.pending_cells?'partial':'complete';$('scope').textContent=completeness+' evidence viewer · 2,840 unique recordings · '+data.accepted_cells+' accepted quality cells · '+data.pending_cells+' pending cells · quality summary SHA-256 '+data.quality_summary_sha256;$('track').onchange=filter;$('search').oninput=filter;$('clip').onchange=render;filter();
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
