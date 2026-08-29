"""Generates the per-family WER bar chart used on the Demonstration slide.

Numbers are copied directly from handoff_ast_asr_audit/summary_*.json --
not re-derived here, so this script has no dependency on the audit code.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

systems = ["whisper\ntiny", "whisper\nsmall", "whisper\nlarge-v3", "distil\nlarge-v3", "Saaras\nV3", "Qwen3-ASR\n0.6B"]
families = ["Indo-Aryan", "Dravidian", "Sino-Tibetan"]
data = {
    "whisper\ntiny":       [21.14, 17.09, 31.28],
    "whisper\nsmall":      [11.01, 10.06, 13.24],
    "whisper\nlarge-v3":   [7.06, 6.60, 8.98],
    "distil\nlarge-v3":    [9.93, 9.60, 13.62],
    "Saaras\nV3":          [5.15, 4.41, 7.86],
    "Qwen3-ASR\n0.6B":     [15.82, 14.16, 17.73],
}

x = np.arange(len(systems))
width = 0.25
colors = ["#3B82C4", "#4CAF7D", "#D9534F"]

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
for i, fam in enumerate(families):
    vals = [data[s][i] for s in systems]
    ax.bar(x + (i - 1) * width, vals, width, label=fam, color=colors[i], edgecolor="white", linewidth=0.5)

ax.set_ylabel("Word Error Rate (%)", fontsize=12)
ax.set_title("Per-family WER on Svarah -- Sino-Tibetan is worst in all six systems (all p < 0.05)", fontsize=12.5, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=10.5)
ax.legend(fontsize=10.5, frameon=False, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 36)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("/workspace/docs/deck_assets/family_wer_chart.png", transparent=False, facecolor="white")
print("wrote /workspace/docs/deck_assets/family_wer_chart.png")
