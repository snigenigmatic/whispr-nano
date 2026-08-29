"""Fills the college-supplied Review1-phase3_ppt template with the real
project content: the Phase 2 -> Phase 3 pivot story and the six-system
Svarah fairness audit. Run with the project's own .venv (needs python-pptx).

Usage: python docs/deck_assets/build_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

TEMPLATE = "/home/ubuntu/.cursor/projects/workspace/uploads/Review1-phase3_ppt_8023.pptx"
OUT = "/workspace/docs/PW25_BJD_05_Phase3_Review1.pptx"
ASSETS = Path(__file__).parent

TITLE = "Whose Indian Accent? A Significance-Tested Subgroup Audit of Indian and Frontier ASR Systems"
SHORT_TITLE = "ASR Fairness Audit"
PROJECT_ID = "PW25_BJD_05"
GUIDE = "Dr. Bhaskarjyoti Das"
TEAM_FULL = "Aditya Sharma, Adithya V Holla, C Kaustubh, Aditi Mangala Udaya"
TEAM_TAG = "Aditya_Adithya_Kaustubh_Aditi"


def find_shape(slide, name):
    for shp in slide.shapes:
        if shp.name == name:
            return shp
    raise KeyError(name)


def no_bullet(paragraph):
    """Strip any inherited auto-number/bullet marker from a paragraph."""
    pPr = paragraph._p.get_or_add_pPr()
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    buNone = pPr.makeelement(qn("a:buNone"), {})
    pPr.append(buNone)


def set_paragraph_text(paragraph, text, size=None, bold=None):
    """Overwrite a paragraph's text while keeping its first run's formatting."""
    if not paragraph.runs:
        run = paragraph.add_run()
    else:
        run = paragraph.runs[0]
        for extra in paragraph.runs[1:]:
            extra.text = ""
    run.text = text
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def clear_frame(tf):
    tf.clear()


BULLET_CHARS = {0: "\u2022  ", 1: "\u2013  "}


def add_bullets(tf, items, base_size=15, first_clear=True):
    """items: list of (level, text, bold) tuples. Manual bullet characters are
    prepended (and any inherited auto-number/bullet stripped) so formatting is
    consistent regardless of which placeholder's original list style we reuse."""
    if first_clear:
        tf.clear()
    for i, (level, text, bold) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        no_bullet(p)
        prefix = BULLET_CHARS.get(level, "") if text else ""
        indent = "     " * level
        p.text = indent + prefix + text
        for r in p.runs:
            r.font.size = Pt(base_size - level * 1.0)
            r.font.bold = bold


def replace_footer(slide, title_shape_name, team_shape_name, short_title=SHORT_TITLE, team_tag=TEAM_TAG):
    t = find_shape(slide, title_shape_name)
    p = t.text_frame.paragraphs[0]
    set_paragraph_text(p, short_title, size=12)
    n = find_shape(slide, team_shape_name)
    set_paragraph_text(n.text_frame.paragraphs[0], team_tag)


def main():
    prs = Presentation(TEMPLATE)
    slides = prs.slides

    # ---------------- Slide 1: Title ----------------
    s1 = slides[0]
    info_box = find_shape(s1, "Google Shape;108;g3785cdd0013_0_0")
    tf = info_box.text_frame
    lines = [
        f"Project Title   :  {TITLE}",
        "Project ID       :  " + PROJECT_ID,
        "Project Guide :  " + GUIDE,
        "Project Team  :  " + TEAM_FULL,
    ]
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        for r in p.runs:
            r.font.size = Pt(15)

    # ---------------- Slide 2: Outline ----------------
    s2 = slides[1]
    outline_body = find_shape(s2, "Google Shape;115;g3785cdd0013_0_8")
    items = [
        (0, "Abstract and Scope of the Project", False),
        (0, "", False),
        (0, "Recap: Capstone Project Phase \u2013 2", False),
        (1, "The evaluation pipeline (\u0394DP, \u0394EO, Poisson significance)", False),
        (1, "GRL failed; GRPO/FR-CISPO on whisper-tiny \u2014 no publication-valid win yet", False),
        (1, "August: the pivot trigger \u2014 Sarvam Saaras V3 encroaches our thesis", False),
        (0, "", False),
        (0, "List of Tasks/Modules with Individual Contribution", False),
        (0, "", False),
        (0, "Design of overall Architecture", False),
        (0, "", False),
        (0, "Demonstration: the six-system Svarah fairness audit", False),
        (0, "", False),
        (0, "References", False),
    ]
    add_bullets(outline_body.text_frame, items, base_size=15)
    replace_footer(s2, "Google Shape;118;g3785cdd0013_0_8", "Google Shape;119;g3785cdd0013_0_8")

    # ---------------- Slide 3: Abstract and Scope ----------------
    s3 = slides[2]
    body = find_shape(s3, "Google Shape;125;g3785cdd0013_0_18")
    abstract_paras = [
        "Indian-accented ASR systems look fine on aggregate Word Error Rate, but nobody "
        "checks whether the errors fall evenly across accents, or whether an observed gap "
        "is real structure versus statistical noise. Phase 2 built an evaluation protocol "
        "for exactly this on Svarah (19 accents \u2192 3 language families), pairing every "
        "gap with a Poisson significance test.",
        "In August, we found Sarvam's flagship Saaras V3 had been evaluated on our exact "
        "benchmark \u2014 shipped by a funded lab, with only an aggregate WER, no subgroup "
        "breakdown and no significance test. Rather than race a training result we cannot "
        "out-resource, Phase 3 points our own protocol at every ASR system we can reach: "
        "the open Whisper family, Saaras V3, and Qwen3-ASR. Six systems, one frozen decode "
        "and scoring path, six statistically significant per-family gaps.",
    ]
    tf = body.text_frame
    tf.clear()
    p0 = tf.paragraphs[0]
    no_bullet(p0)
    p0.text = abstract_paras[0]
    for r in p0.runs:
        r.font.size = Pt(15)
    pblank = tf.add_paragraph()
    p1 = tf.add_paragraph()
    no_bullet(p1)
    p1.text = abstract_paras[1]
    for r in p1.runs:
        r.font.size = Pt(15)
    replace_footer(s3, "Google Shape;128;g3785cdd0013_0_18", "Google Shape;129;g3785cdd0013_0_18")

    # ---------------- Slide 4: Summary of Phase 2 + pivot ----------------
    s4 = slides[3]
    title4 = find_shape(s4, "Google Shape;136;g3785cdd0013_0_28")
    set_paragraph_text(title4.text_frame.paragraphs[0], "Summary of Work Done in Phase \u2013 2, and Why We Pivoted", size=22)
    body4 = find_shape(s4, "Google Shape;135;g3785cdd0013_0_28")
    items4 = [
        (0, "Built a Svarah evaluation pipeline: \u0394DP, \u0394EO, per-family noise-robustness, Poisson drop-in-deviance p-value", False),
        (0, "GRL (Wav2Vec2 + LoRA) adversarial de-biasing WIDENED the accent gap \u2014 SPIRE-SIES has zero Sino-Tibetan speakers, so the adversary cannot reverse a gradient for a group it never sees", False),
        (0, "Pivoted to GRPO / FR-CISPO RL post-training on whisper-tiny \u2014 safety-gated (per-token KL, ratio caps): H0 (300-cycle run at 3 learning rates) refuted, H6 replication failed at seed 2028; no publication-valid result yet on Svarah (authoritative speaker IDs unobtainable)", False),
        (0, "One banked, real result: the SPIRE-SIES cross-corpus ladder \u2014 29.89% \u2192 24.79% overall WER, worst-20%-speaker WER \u201223.87 pp (4.7\u00d7 the average gain) \u2014 but the explicit fairness term is provably inert", False),
        (0, "August literature pass (63 records \u2192 12 kept): Sarvam's Saaras V3 was validated on our exact Svarah benchmark; ASR-FAIRBENCH already publishes our Poisson-test idea; a Nov-2025 paper shows third-party ASR audits get through review", False),
        (0, "\u2192 Pivot: stop competing on a training result, ship the audit our own pipeline was always capable of", True),
    ]
    add_bullets(body4.text_frame, items4, base_size=13.5)
    replace_footer(s4, "Google Shape;138;g3785cdd0013_0_28", "Google Shape;139;g3785cdd0013_0_28")

    # ---------------- Slide 5: Architecture ----------------
    s5 = slides[4]
    body5 = find_shape(s5, "Google Shape;145;g3785cdd0013_0_38")
    body5.top = Emu(1300000)
    body5.height = Emu(450000)
    p5 = body5.text_frame.paragraphs[0]
    no_bullet(p5)
    set_paragraph_text(p5, "One frozen pipeline, six systems: acquire \u2192 transcribe \u2192 score \u2192 test \u2192 verdict", size=15)
    p5.alignment = PP_ALIGN.CENTER
    for extra_p in body5.text_frame.paragraphs[1:]:
        for r in extra_p.runs:
            r.text = ""
    # Fit the diagram into the space between the caption and the footer band.
    img_top = Emu(1900000)
    img_height = Emu(4300000)
    img = s5.shapes.add_picture(str(ASSETS / "architecture.png"), Emu(0), img_top, height=img_height)
    img.left = Emu(int((prs.slide_width - img.width) / 2))
    replace_footer(s5, "Google Shape;147;g3785cdd0013_0_38", "Google Shape;148;g3785cdd0013_0_38")

    # ---------------- Slide 6: List of Tasks/Modules ----------------
    s6 = slides[5]
    body6 = find_shape(s6, "Google Shape;154;g3785cdd0013_0_47")
    items6 = [
        (0, "Data module \u2014 Svarah loader (ai4bharat/svarah), 19-accent \u2192 3-family taxonomy, per-family stratified sampling", False),
        (0, "Whisper-family audit module \u2014 whisper-tiny / small / large-v3 / distil-large-v3 on Modal T4 GPUs, 2,500 clips each", False),
        (0, "External-API module \u2014 Sarvam Saaras V3 REST client (cached, rate-limited) and Qwen3-ASR-0.6B via Transformers", False),
        (0, "Scoring module \u2014 ast_asr.metrics.normalize_for_wer / word_edit_counts, reused unchanged from Phase 2", False),
        (0, "Statistics module \u2014 Poisson drop-in-deviance GLM (statsmodels) \u2192 \u0394DP + p-value + structural/noise verdict", False),
        (0, "Cost-guardrail module \u2014 pre-flight estimate, in-run measured-rate re-plan, hard Modal timeout on every job", False),
        (0, "Results-assembly module \u2014 schema-hard-fail assembler (9/9 pytest green) that recomputes every reported number straight from the per-utterance CSVs and cross-checks it against each summary JSON", False),
        (0, "Side-quest module \u2014 on-policy distillation (whisper-tiny \u2190 whisper-large-v3) PoC on Modal, parked as a future direction", False),
        (0, "", False),
        (0, "Tools / tech: Modal (serverless GPU), PyTorch + Transformers, HF Datasets/torchcodec, statsmodels, Sarvam Saaras V3 API, Qwen3-ASR \u2014 all open-source or free-tier API access", False),
    ]
    add_bullets(body6.text_frame, items6, base_size=13)
    replace_footer(s6, "Google Shape;156;g3785cdd0013_0_47", "Google Shape;157;g3785cdd0013_0_47")

    # ---------------- Slide 7: Individual Contribution ----------------
    s7 = slides[6]
    old7 = find_shape(s7, "Google Shape;163;g3785cdd0013_0_56")
    left, top = old7.left, old7.top
    old7._element.getparent().remove(old7._element)

    rows = [
        ("Member", "Task / Module", "Contribution", "Status"),
        ("Aditya Sharma", "Whisper-family fleet",
         "Ported Phase 2's eval pipeline to Modal; whisper-tiny/small/large-v3/distil-large-v3, 2,500 clips each",
         "Complete"),
        ("Adithya V Holla", "External systems",
         "Sarvam Saaras V3 client (cache + backoff) and Qwen3-ASR-0.6B feasibility + run",
         "Complete"),
        ("Aditi Mangala Udaya", "Assembly & receipts",
         "Schema-hard-fail results assembler (9/9 tests green, reproduces all 6 rows exactly); 3-source receipts pack (PDF); fixed the HuBERT-row and [17]-citation report bugs",
         "Complete"),
        ("C Kaustubh", "Pivot lead + audit",
         "Literature search, pivot memo, blockers A/B, on-policy distillation side-quest (whispr-nano), and ran all six audit systems end-to-end to hit this review's deadline",
         "Complete"),
    ]
    n_rows, n_cols = len(rows), 4
    tbl_width, tbl_height = Emu(9067800), Emu(2650000)
    gframe = s7.shapes.add_table(n_rows, n_cols, left, top, tbl_width, tbl_height)
    table = gframe.table
    col_widths = [Emu(1900000), Emu(1650000), Emu(4400000), Emu(1117800)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11.5 if r > 0 else 12.5)
                if r == 0:
                    p.font.bold = True

    note = s7.shapes.add_textbox(left, Emu(top + 2750000), tbl_width, Emu(500000))
    ntf = note.text_frame
    ntf.word_wrap = True
    ntf.paragraphs[0].text = (
        "Note: given the review deadline, Kaustubh personally ran all six audit systems (including the "
        "Whisper-family and external-API tracks) and completed the assembler/receipts/report-fix tasks "
        "on Modal + the vendor APIs; ownership above reflects the planned task split and hand-off "
        "documentation, which the team is now reconciling in parallel."
    )
    ntf.paragraphs[0].font.size = Pt(11)
    ntf.paragraphs[0].font.italic = True
    replace_footer(s7, "Google Shape;165;g3785cdd0013_0_56", "Google Shape;166;g3785cdd0013_0_56")

    # ---------------- Slide 8: Demonstration ----------------
    s8 = slides[7]
    title8 = find_shape(s8, "Google Shape;171;g3785cdd0013_0_65")
    set_paragraph_text(title8.text_frame.paragraphs[0], "Demonstration: the Six-System Svarah Fairness Audit")
    old8 = find_shape(s8, "Google Shape;172;g3785cdd0013_0_65")
    left8, top8 = old8.left, old8.top
    old8._element.getparent().remove(old8._element)

    audit_rows = [
        ("System", "Overall WER", "Worst family", "\u0394DP (pp)", "Poisson p", "n", "Cost"),
        ("whisper-tiny", "20.60%", "Sino-Tibetan", "14.19", "3.3e-24", "2,500", "$0.034"),
        ("whisper-small", "10.88%", "Sino-Tibetan", "3.18", "0.0045", "2,500", "$0.095"),
        ("whisper-large-v3", "7.04%", "Sino-Tibetan", "2.38", "0.0152", "2,500", "$0.411"),
        ("distil-large-v3", "10.03%", "Sino-Tibetan", "4.02", "0.00025", "2,500", "$0.285"),
        ("Saaras V3", "5.81% (best)", "Sino-Tibetan", "3.45", "0.00023", "450", "$0.226"),
        ("Qwen3-ASR-0.6B", "15.49%", "Sino-Tibetan", "3.57", "0.0013", "2,500", "$0.142"),
    ]
    n_rows, n_cols = len(audit_rows), 7
    tbl_w, tbl_h = Emu(9067800), Emu(2000000)
    gframe8 = s8.shapes.add_table(n_rows, n_cols, left8, top8, tbl_w, tbl_h)
    table8 = gframe8.table
    widths8 = [Emu(1650000), Emu(1450000), Emu(1400000), Emu(1000000), Emu(1200000), Emu(750000), Emu(1617800)]
    for i, w in enumerate(widths8):
        table8.columns[i].width = w
    for r, row in enumerate(audit_rows):
        for c, val in enumerate(row):
            cell = table8.cell(r, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10.5 if r > 0 else 11.5)
                p.alignment = PP_ALIGN.CENTER
                if r == 0:
                    p.font.bold = True

    caption = s8.shapes.add_textbox(left8, Emu(top8 + tbl_h + 50000), tbl_w, Emu(350000))
    ctf = caption.text_frame
    ctf.word_wrap = True
    ctf.paragraphs[0].text = ("All six systems: Sino-Tibetan worst, all p < 0.05 (structural). "
                               "Total cost across all six: $1.193. Best overall WER (Saaras V3) does "
                               "not close the fairness gap \u2014 aggregate WER hides subgroup movement.")
    ctf.paragraphs[0].font.size = Pt(11)
    ctf.paragraphs[0].font.italic = True

    chart_top = Emu(top8 + tbl_h + 480000)
    chart_height = Emu(int(prs.slide_height - chart_top - 500000))
    chart_pic = s8.shapes.add_picture(str(ASSETS / "family_wer_chart.png"), Emu(300000), chart_top, height=chart_height)

    bonus = s8.shapes.add_textbox(Emu(int(300000 + chart_pic.width + 250000)), chart_top,
                                   Emu(int(prs.slide_width - (300000 + chart_pic.width + 250000) - 300000)),
                                   chart_height)
    btf = bonus.text_frame
    btf.word_wrap = True
    btf.paragraphs[0].text = "Bonus \u2014 whispr-nano (side-quest, $0.16):"
    btf.paragraphs[0].font.size = Pt(11.5)
    btf.paragraphs[0].font.bold = True
    for line in [
        "On-policy WER: 10.09% \u2192 7.89%",
        "On-policy KL: 0.633 \u2192 0.496",
        "Off-policy WER: 10.09% \u2192 7.57%",
        "Off-policy KL: 0.633 \u2192 0.524",
        "On-policy wins on calibration \u2014 matches the LLM-distillation literature (GKD).",
    ]:
        p = btf.add_paragraph()
        p.text = line
        p.font.size = Pt(10.5)

    replace_footer(s8, "Google Shape;174;g3785cdd0013_0_65", "Google Shape;175;g3785cdd0013_0_65")

    # ---------------- Slide 9: References ----------------
    s9 = slides[8]
    body9 = find_shape(s9, "Google Shape;181;g3785cdd0013_0_74")
    refs = [
        "[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, \u201cRobust Speech Recognition via Large-Scale Weak Supervision,\u201d arXiv:2212.04356, 2022.",
        "[2] T. Javed et al., \u201cSvarah: Evaluating English ASR Systems on Indian Accents,\u201d in Proc. INTERSPEECH 2023, pp. 5087\u20135091, 2023.",
        "[3] A. K. Rai, S. Rahangdale, U. Anand, and A. Mukherjee, \u201cASR-FAIRBENCH: Measuring and Benchmarking Equity Across Speech Recognition Systems,\u201d in Proc. INTERSPEECH 2025, 2025.",
        "[4] S. Kumar et al., \u201cASR Under the Stethoscope: Evaluating Biases in Clinical Speech Recognition across Indian Languages,\u201d arXiv:2512.10967, 2025.",
        "[5] A. Koenecke et al., \u201cRacial disparities in automated speech recognition,\u201d Proc. Natl. Acad. Sci., vol. 117, no. 14, pp. 7684\u20137689, 2020.",
        "[6] Z. Liu, I.-E. Veliche, and F. Peng, \u201cModel-Based Approach for Measuring the Fairness in ASR,\u201d in Proc. ICASSP, pp. 6532\u20136536, 2022.",
        "[7] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, \u201cDistributionally Robust Neural Networks for Group Shifts,\u201d arXiv:1911.08731, 2020.",
        "[8] P. G. Shivakumar, Y. Gu, A. Gandhe, and I. Bulyko, \u201cGroup Relative Policy Optimization for Speech Recognition,\u201d in Proc. IEEE ASRU, 2025.",
        "[9] R. Agarwal et al., \u201cOn-Policy Distillation of Language Models: Learning from Self-Generated Mistakes,\u201d in Proc. ICLR, 2024.",
        "[10] Thinking Machines Lab, \u201cOn-Policy Distillation,\u201d Company blog, 2025.",
        "[11] S. Gandhi, P. von Platen, and A. M. Rush, \u201cDistil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling,\u201d arXiv:2311.00430, 2023.",
        "[12] Sarvam AI, \u201cIntroducing Saaras V3,\u201d Company blog, Feb. 2026. [Online]. Available: sarvam.ai/blogs/asr",
    ]
    tf9 = body9.text_frame
    tf9.clear()
    for i, ref in enumerate(refs):
        p = tf9.paragraphs[0] if i == 0 else tf9.add_paragraph()
        p.text = ref
        for r in p.runs:
            r.font.size = Pt(10.5)
    replace_footer(s9, "Google Shape;183;g3785cdd0013_0_74", "Google Shape;184;g3785cdd0013_0_74")

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
