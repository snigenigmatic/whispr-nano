"""Publication figures generated only from machine-readable analysis outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["whisper-tiny", "whisper-small", "whisper-large-v3", "distil-large-v3", "Qwen3-ASR-0.6B-hf", "saarasv3"]
LABELS = {"whisper-tiny":"Whisper tiny", "whisper-small":"Whisper small", "whisper-large-v3":"Whisper large-v3",
          "distil-large-v3":"Distil large-v3", "Qwen3-ASR-0.6B-hf":"Qwen3-ASR 0.6B", "saarasv3":"Saaras V3"}
BLUE, ORANGE = "#1F5C8F", "#CB651B"


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build(results: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":10,
                         "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42})
    metrics, contrasts = read(results/"metrics.csv"), read(results/"contrasts.csv")
    historical = "historical_open_2500"
    systems = [x for x in ORDER if any(r["system"]==x and r["cohort"]==historical for r in metrics)]
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 6.6), gridspec_kw={"hspace":0.42})
    for ax, kind, title, xlabel in zip(axes, ("wer", "gap"),
            ("A  Overall word error rate", "B  Bodo minus pooled non-Bodo"),
            ("WER (%)", "WER difference (percentage points)")):
        for y, system in enumerate(systems):
            vals=[]
            for normalizer in ("minimal", "whisper_english"):
                if kind=="wer":
                    row=next(r for r in metrics if r["cohort"]==historical and r["normalizer"]==normalizer and r["system"]==system and r["level"]=="overall")
                    vals.append(float(row["wer_pct"]))
                else:
                    row=next(r for r in contrasts if r["cohort"]==historical and r["normalizer"]==normalizer and r["system"]==system and r["contrast"]=="Bodo_minus_non_Bodo")
                    vals.append(float(row["difference_pp"]))
            ax.plot(vals, [y,y], color="#A5AFB8", linewidth=1.5, zorder=1)
            for x, color, marker in zip(vals, (BLUE,ORANGE), ("o","s")):
                ax.scatter(x,y,color=color,marker=marker,s=40,zorder=3)
            ax.annotate(f"{vals[0]:.2f}",(vals[0],y),xytext=(0,8),textcoords="offset points",ha="center",color=BLUE,fontsize=7.4)
            ax.annotate(f"{vals[1]:.2f}",(vals[1],y),xytext=(0,-13),textcoords="offset points",ha="center",color=ORANGE,fontsize=7.4)
        ax.set_yticks(range(len(systems)), [LABELS[x] for x in systems])
        ax.invert_yaxis(); ax.set_ylim(len(systems)-0.15,-0.55)
        ax.set_title(title,loc="left",fontweight="bold",pad=12)
        ax.set_xlabel(xlabel);ax.grid(axis="x",color="#E5E8EA",linewidth=.6)
        ax.set_axisbelow(True); ax.margins(x=.15)
        if kind=="gap":ax.axvline(0,color="#7B8389",linewidth=.7)
    axes[0].scatter([],[],color=BLUE,label="Minimal",marker="o")
    axes[0].scatter([],[],color=ORANGE,label="English",marker="s")
    axes[0].legend(loc="upper right",bbox_to_anchor=(1,1.31),frameon=False,ncol=2,fontsize=8,handletextpad=.3,columnspacing=.8)
    save(fig,destination/"normalization_sensitivity")

    # All native-language point estimates, retaining small groups and their n.
    languages=sorted({r["group"] for r in metrics if r["cohort"]==historical and r["level"]=="language"})
    matrix=np.array([[float(next(r["wer_pct"] for r in metrics if r["cohort"]==historical and r["normalizer"]=="whisper_english" and r["system"]==system and r["level"]=="language" and r["group"]==language)) for system in systems] for language in languages])
    support={language:next(r["n_utterances"] for r in metrics if r["cohort"]==historical and r["level"]=="language" and r["group"]==language) for language in languages}
    fig,ax=plt.subplots(figsize=(5.5,7.1))
    mesh=ax.imshow(matrix,cmap="YlGnBu",aspect="auto",vmin=0,vmax=float(matrix.max()))
    ax.set_xticks(range(len(systems)),[LABELS[x].replace("Whisper ","W. ").replace("Distil ","D. ").replace("Qwen3-ASR ","Qwen ") for x in systems],rotation=40,ha="right")
    ax.set_yticks(range(len(languages)),[f"{x} ({support[x]})" for x in languages])
    for i in range(len(languages)):
        for j in range(len(systems)):
            ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",fontsize=7.7,
                    color="white" if matrix[i,j]>.58*matrix.max() else "#183040")
    for label in ax.get_yticklabels():
        if label.get_text().startswith("Bodo"):
            label.set_fontweight("bold");label.set_color(ORANGE)
    ax.set_title("Native-language WER: Open-2500\nShared English normalization",loc="left",fontweight="bold",pad=12)
    ax.set_ylabel("Native language (utterances)")
    fig.colorbar(mesh,ax=ax,fraction=.035,pad=.04,label="WER (%)")
    save(fig,destination/"native_language_wer")

    profile_path=results/"demographic_profile"/"contrasts.csv"
    if profile_path.exists():
        profile=read(profile_path)
        fig,ax=plt.subplots(figsize=(4.4,3.8))
        for y,system in enumerate(systems):
            for rows,offset,color,marker in ((contrasts,-.13,BLUE,"o"),(profile,.13,ORANGE,"s")):
                row=next(r for r in rows if r["cohort"]==historical and r["normalizer"]=="whisper_english" and r["system"]==system and r["contrast"]=="Bodo_minus_non_Bodo")
                if row["ci_low_pp"] and row["ci_high_pp"]:
                    value,lo,hi=map(float,(row["difference_pp"],row["ci_low_pp"],row["ci_high_pp"]))
                    ax.plot([lo,hi],[y+offset,y+offset],color=color,lw=1.4)
                    ax.scatter(value,y+offset,color=color,marker=marker,s=25,zorder=3)
        ax.set_yticks(range(len(systems)),[LABELS[x] for x in systems]);ax.invert_yaxis()
        ax.axvline(0,color="#7B8389",lw=.8);ax.grid(axis="x",color="#E5E8EA",lw=.6)
        ax.set_xlabel("Bodo - non-Bodo WER (percentage points)")
        ax.set_title("Dependence-unit sensitivity\nOpen-2500, English normalization",loc="left",fontweight="bold")
        ax.plot([],[],color=BLUE,marker="o",label="Filename blocks")
        ax.plot([],[],color=ORANGE,marker="s",label="Demographic profiles (unverified)")
        ax.legend(loc="upper center",bbox_to_anchor=(.45,-.19),frameon=False,fontsize=8)
        save(fig,destination/"cluster_sensitivity")
    print(f"Wrote figures to {destination}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"paper/results")
    p.add_argument("--output",type=Path,default=ROOT/"paper/figures")
    args=p.parse_args();build(args.results,args.output)
