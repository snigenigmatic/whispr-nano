"""Render validated Hindi recipe comparison; no model execution.

Run with a Python environment containing Matplotlib:
    python research/bridge_open/plot_expansion.py
"""
import json
from pathlib import Path

from orx_figstyle import TEXT, PALETTE, family, figure, save, use_style

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts/expanded_benchmark'
LABELS = {
    'openai/whisper-small': 'Whisper Small · beam 5',
    'openai/whisper-large-v3-turbo': 'Whisper Turbo · beam 5',
    'openai/whisper-large-v3': 'Whisper Large-v3 · beam 5',
    'Qwen/Qwen3-ASR-0.6B': 'Qwen3-ASR 0.6B · greedy',
    'Qwen/Qwen3-ASR-1.7B': 'Qwen3-ASR 1.7B · greedy',
    'facebook/omniASR-CTC-1B': 'OmniASR CTC 1B · greedy',
}


def main():
    data = json.loads((OUT / 'comparison.json').read_text())
    rows = []
    for run in data['runs']:
        subset = [c for c in run['conversations'] if c['language'] == 'hi']
        assert {c['conversation_id'] for c in subset} == {'humyn-13-00', 'humyn-13-01', 'humyn-14-00'}
        rows.append((run['model_id'], 100 * run['summary']['languages']['hi']['micro_wer']))
    rows.sort(key=lambda r: r[1])
    blues = family('blue', 3)
    colors = {'openai/whisper-small': blues[0], 'openai/whisper-large-v3-turbo': blues[1],
              'openai/whisper-large-v3': blues[2], 'Qwen/Qwen3-ASR-0.6B': '#9A9A9A',
              'Qwen/Qwen3-ASR-1.7B': '#555555', 'facebook/omniASR-CTC-1B': PALETTE['orange']}
    use_style()
    fig, ax = figure(width=TEXT, ratio=.52)
    bars = ax.barh(range(len(rows)), [v for _, v in rows],
                   color=[colors[m] for m, _ in rows], edgecolor='black', linewidth=.4)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=7)
    ax.set_yticks(range(len(rows)), [LABELS[m] for m, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(v for _, v in rows) * 1.18)
    ax.set_xlabel('Word error rate (%) — lower is better')
    ax.set_ylabel('Inference recipe')
    ax.grid(axis='x', alpha=.25); ax.grid(axis='y', visible=False)
    save(fig, str(OUT / 'hindi_recipes'))
    (OUT / 'hindi_recipes.caption.md').write_text(
        'Matched Hindi development results for six open ASR recipes. Three conversations '
        '(6,503 reference words), fixed 30-second windows; single inference seed 12. '
        'Capped and repeated output is retained. No confidence intervals: only three '
        'conversation clusters. Precision and decoding differ by family; this is not '
        'an isolated architecture comparison or held-out model ranking.\n')


if __name__ == '__main__':
    main()
