"""Generates the audit-pipeline architecture diagram for the Architecture slide."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12.9, 5.8), dpi=200)
ax.set_xlim(0, 12.9)
ax.set_ylim(0, 5.8)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.3,rounding_size=0.14", linewidth=1.5)

def box(cx, cy, w, h, text, fc, ec, fontsize=9.8, weight="normal"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, **box_style, facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=4, linespacing=1.4)
    return (cx, cy, w, h)

def harrow(b1, b2, gap=0.18):
    x1 = b1[0] + b1[2] / 2 + gap
    x2 = b2[0] - b2[2] / 2 - gap
    y = (b1[1] + b2[1]) / 2
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=18,
                         linewidth=1.6, color="#666666", zorder=2)
    ax.add_patch(a)

def varrow(b1, b2, gap=0.15):
    x = (b1[0] + b2[0]) / 2
    y1 = b1[1] + b1[3] / 2 + gap
    y2 = b2[1] - b2[3] / 2 - gap
    a = FancyArrowPatch((b1[0], y1), (b2[0], y2), arrowstyle="-|>", mutation_scale=18,
                         linewidth=1.6, color="#666666", zorder=2)
    ax.add_patch(a)

col_x = [1.65, 4.85, 8.05, 11.25]
col_w = [2.55, 2.55, 2.55, 2.1]

y_main = 4.3
w0 = box(col_x[0], y_main, col_w[0], 1.5,
         "6 ASR systems\nWhisper tiny / small /\nlarge-v3 / distil-large-v3,\nSaaras V3, Qwen3-ASR",
         "#DCEBFA", "#3B82C4", fontsize=9.3)

w1 = box(col_x[1], y_main, col_w[1], 1.5,
         "Modal T4 GPU +\nvendor REST APIs\n(cost-guardrailed)",
         "#E4DCF5", "#7C5CBF", fontsize=9.3)

w2 = box(col_x[2], y_main, col_w[2], 1.5,
         "Greedy transcripts\nnormalize_for_wer\n(ast_asr.metrics)",
         "#DCEBFA", "#3B82C4", fontsize=9.3)

w3 = box(col_x[3], y_main, col_w[3], 1.5,
         "Poisson\ndrop-in-deviance GLM\n(statsmodels)",
         "#DFF5E1", "#4CAF7D", fontsize=9.1)

harrow(w0, w1, gap=0.22)
harrow(w1, w2, gap=0.22)
harrow(w2, w3, gap=0.22)

y_low = 1.55
s0 = box(col_x[0], y_low, col_w[0], 1.35,
         "Svarah (ai4bharat)\n19 accents -> 3 language\nfamilies (taxonomy)",
         "#FCE8D6", "#D9822B", fontsize=9.1)

s1 = box(col_x[1], y_low, col_w[1], 1.35,
         "Pre-flight + measured-\nrate + hard-timeout\nbudget enforcement",
         "#FDE4E1", "#D9534F", fontsize=9.1)

s2 = box(col_x[2], y_low, col_w[2], 1.35,
         "Per-family WER\n(word_edit_counts)",
         "#DCEBFA", "#3B82C4", fontsize=9.1)

s3 = box(col_x[3], y_low, col_w[3], 1.35,
         "{Delta}DP + p-value\n->  structural\nor noise",
         "#DFF5E1", "#4CAF7D", fontsize=8.9)

varrow(w0, s0)
varrow(w1, s1)
varrow(w2, s2)
varrow(w3, s3)
harrow(s2, s3)

fig.text(0.5, 0.035,
         "Reuses Phase 2's own taxonomy + metrics end to end, so every audit row is directly comparable to every other Svarah number in this project.",
         ha="center", fontsize=10.2, color="#333333", style="italic")

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("/workspace/docs/deck_assets/architecture.png", facecolor="white")
print("wrote /workspace/docs/deck_assets/architecture.png")
