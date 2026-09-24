"""Plot observed system WER on complete FLEURS language cohorts."""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'research/bridge_open/artifacts/fleurs_qwen_comparison_v1/comparison.json').is_file())
ART = ROOT / 'research/bridge_open/artifacts'
STYLE = ART / 'fleurs_whisper_corrected_v3/figures/orx_figstyle.py'
sys.path.insert(0, str(STYLE.parent))
from orx_figstyle import use_style, figure, save, WIDE, PALETTE, MUTED


def main():
    omni = json.loads((ART / 'fleurs_omni_comparison_v1/comparison.json').read_text())
    hindi = json.loads((ART / 'fleurs_qwen_comparison_v1/comparison.json').read_text())
    from research.bridge_open.analyze_fleurs import _aggregate
    systems = {}
    for name, file in [('Whisper Turbo', 'turbo_predictions.jsonl'), ('Whisper Large-v3', 'large-v3_predictions.jsonl')]:
        rows = [json.loads(x) for x in (ART / 'fleurs_whisper_corrected_v3' / file).read_text().splitlines()]
        systems[name] = {lang: _aggregate([r for r in rows if r['language'] == lang])['micro_wer'] * 100 for lang in ('hi', 'bn', 'as')}
    systems['Qwen3-ASR-1.7B'] = {'hi': hindi['summary']['Qwen3-ASR-1.7B']['micro_wer'] * 100}
    systems['OmniASR CTC-1B'] = {l: g['micro_wer'] * 100 for l, g in omni['groups'].items()}
    out = ART / 'fleurs_omni_comparison_v1/figures'; out.mkdir(exist_ok=True)
    use_style(); fig, ax = figure(width=WIDE, ratio=.50)
    colors = [MUTED, PALETTE['blue'], PALETTE['green'], PALETTE['orange']]
    patterns = ['', '//', '..', 'xx']
    for i, (name, vals) in enumerate(systems.items()):
        x = [j + (i - 1.5)*.2 for j, lang in enumerate(('hi', 'bn', 'as')) if lang in vals]
        y = [vals[lang] for lang in ('hi', 'bn', 'as') if lang in vals]
        bars = ax.bar(x, y, width=.19, label=name + (' (Hindi only)' if name.startswith('Qwen') else ''), color=colors[i], hatch=patterns[i], edgecolor='black', linewidth=.4)
        ax.bar_label(bars, fmt='%.2f', fontsize=6.5, padding=3)
    ax.set_xticks([0, 1, 2], ['Hindi (418)', 'Bengali (920)', 'Assamese (984)'])
    ax.set_ylabel('Micro WER (%)'); ax.set_xlabel('FLEURS official-test language (recordings)')
    ax.set_ylim(0, max(v for s in systems.values() for v in s.values()) * 1.18)
    ax.legend(ncol=2, loc='upper left', fontsize=7, frameon=True, framealpha=1, edgecolor='black')
    save(fig, str(out / 'system_comparison'), close=False)
    fig.savefig(out / 'system_comparison_preview.png', dpi=180)
    with (out / 'values.csv').open('w', newline='') as f:
        w=csv.writer(f);w.writerow(['system','language','micro_wer_percent'])
        for name, vals in systems.items():
            for lang, v in vals.items():w.writerow([name, lang, v])
    (out / 'CAPTION.md').write_text('OmniASR CTC improves Bengali and Assamese WER relative to the evaluated Whisper systems; Qwen retains the lowest Hindi WER. Complete cohorts and one run per declared condition are shown. Qwen was not evaluated on Bengali or Assamese. Precision and decoding differ by system; paired sentence-cluster uncertainty is reported separately. These are descriptive benchmark results, not a new training method or population-level fairness result.\n')
    if Path(__file__).resolve() != (out / 'plot_expanded_benchmark.py').resolve():
        shutil.copy2(__file__, out / 'plot_expanded_benchmark.py')
    shutil.copy2(STYLE, out / 'orx_figstyle.py')


if __name__ == '__main__': main()
