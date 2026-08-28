"""Small matplotlib helpers turning an ArmResult / run summary into PNG bytes,
so the Modal remote function can ship plots back over the wire without a
separate results Volume round-trip for something this small."""

from __future__ import annotations

import io


def loss_curve_png(arms: dict, title: str = "Reverse KL over training") -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for label, arm in arms.items():
        steps = [s["step"] for s in arm["step_log"]]
        losses = [s["loss"] for s in arm["step_log"]]
        ax.plot(steps, losses, label=label, linewidth=1.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("mean per-token reverse KL (nats)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def wer_bar_png(arms: dict, title: str = "WER before vs. after distillation") -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(arms.keys())
    before = [arms[k]["baseline_eval"]["student_wer"] * 100 for k in labels]
    after = [arms[k]["final_eval"]["student_wer"] * 100 for k in labels]
    teacher = arms[labels[0]]["baseline_eval"]["teacher_wer"] * 100 if labels else 0.0

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.bar(x - width / 2, before, width, label="student before")
    ax.bar(x + width / 2, after, width, label="student after")
    ax.axhline(teacher, color="black", linestyle="--", linewidth=1, label="teacher (large-v3)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WER (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
