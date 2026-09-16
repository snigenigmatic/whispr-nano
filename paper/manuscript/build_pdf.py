"""Render the verified Markdown manuscript as a compact two-column paper.

Uses local fonts and ReportLab. Figures/tables remain reproducible artifacts;
the Markdown and BibTeX sources are suitable for transfer to a venue template.
"""
from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, XPreformatted,
)

ROOT = Path(__file__).resolve().parents[2]
PAGE_W, PAGE_H = A4
MARGIN, GUTTER = 37.0, 19.0
WIDTH = PAGE_W - 2*MARGIN
COL = (WIDTH-GUTTER)/2


def fonts():
    directory = Path("/System/Library/Fonts/Supplemental")
    paths = {"Paper": "Times New Roman.ttf", "Paper-Bold": "Times New Roman Bold.ttf",
             "Paper-Italic": "Times New Roman Italic.ttf", "Paper-BoldItalic": "Times New Roman Bold Italic.ttf"}
    for name, filename in paths.items():
        pdfmetrics.registerFont(TTFont(name, str(directory/filename)))
    pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="Paper-Bold", italic="Paper-Italic", boldItalic="Paper-BoldItalic")
    pdfmetrics.registerFont(TTFont("Mono", "/System/Library/Fonts/Menlo.ttc", subfontIndex=0))


def clean(text: str) -> str:
    return text.translate(str.maketrans({"\u2010":"-", "\u2011":"-", "\u2012":"-", "\u2013":"-", "\u2014":"-", "\u2212":"-", "\u00a0":" "}))


def inline(element: ET.Element) -> str:
    parts = [html.escape(clean(element.text or ""))]
    for child in element:
        content = inline(child)
        if child.tag in ("strong", "b"):
            parts.append(f"<b>{content}</b>")
        elif child.tag in ("em", "i"):
            parts.append(f"<i>{content}</i>")
        elif child.tag == "code":
            parts.append(f'<font name="Mono" size="7.4">{content}</font>')
        elif child.tag == "a":
            href = html.escape(child.attrib.get("href", ""), quote=True)
            parts.append(f'<a href="{href}" color="#234D72">{content}</a>')
        elif child.tag == "br":
            parts.append("<br/>")
        elif child.tag != "img":
            parts.append(content)
        parts.append(html.escape(clean(child.tail or "")))
    return "".join(parts)


def build(source: Path, output: Path, allow_draft: bool = False):
    text = source.read_text()
    if not allow_draft and re.search(r"<!--\s*(?:INSERT|INTERNAL DRAFT)|\b(?:TODO|TBD)\b", text):
        raise ValueError("Manuscript contains draft markers; refusing a final PDF")
    fonts()
    parsed = ET.fromstring("<root>" + markdown.markdown(clean(text), extensions=["tables", "fenced_code", "sane_lists"]) + "</root>")
    elements = list(parsed)
    title = "".join(elements.pop(0).itertext())
    subtitle = ""
    if elements and elements[0].tag == "h2" and not "".join(elements[0].itertext()).startswith(("Abstract", "1.")):
        subtitle = "".join(elements.pop(0).itertext())
    if elements and "".join(elements[0].itertext()).strip() == "Anonymous authors":
        elements.pop(0)
    body = ParagraphStyle("body", fontName="Paper", fontSize=9.25, leading=11.45,
                          spaceAfter=5.0, alignment=TA_JUSTIFY, splitLongWords=True,
                          allowWidows=0, allowOrphans=0)
    heading = ParagraphStyle("heading", parent=body, fontName="Paper-Bold", fontSize=10.7,
                             leading=12.7, spaceBefore=8, spaceAfter=4, keepWithNext=True,
                             alignment=TA_LEFT)
    subheading = ParagraphStyle("subheading", parent=heading, fontSize=9.6, leading=11.5,
                                spaceBefore=5, spaceAfter=3)
    caption = ParagraphStyle("caption", parent=body, fontSize=8.1, leading=9.6,
                              alignment=TA_LEFT, spaceBefore=3, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.7, leading=9.0, spaceAfter=0,
                            alignment=TA_LEFT)
    references = ParagraphStyle("references", parent=body, fontSize=8.15, leading=9.7,
                                 alignment=TA_LEFT, spaceAfter=4)
    title_style = ParagraphStyle("title", fontName="Paper-Bold", fontSize=17.2, leading=19.5,
                                  alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("subtitle", fontName="Paper", fontSize=11.5, leading=14,
                                     alignment=TA_CENTER)
    title_para = Paragraph(html.escape(title), title_style)
    _, title_h = title_para.wrap(WIDTH, 100)
    subtitle_para = Paragraph(html.escape(subtitle), subtitle_style)
    _, subtitle_h = subtitle_para.wrap(WIDTH, 50)
    title_area = title_h + subtitle_h + 39

    def decorate(canvas, doc):
        canvas.setTitle(title + (": " + subtitle if subtitle else ""))
        canvas.setAuthor("Anonymous authors")
        canvas.setStrokeColor(colors.HexColor("#A7ADB4"))
        canvas.setLineWidth(0.35)
        canvas.line(MARGIN, 30, PAGE_W-MARGIN, 30)
        canvas.setFont("Paper", 8)
        canvas.setFillColor(colors.HexColor("#5C6269"))
        canvas.drawString(MARGIN, 19, "Research manuscript | 9 September 2026")
        canvas.drawRightString(PAGE_W-MARGIN, 19, str(doc.page))
        if doc.page == 1:
            top = PAGE_H - MARGIN
            title_para.drawOn(canvas, MARGIN, top-title_h)
            subtitle_para.drawOn(canvas, MARGIN, top-title_h-7-subtitle_h)
            canvas.setFont("Paper", 9)
            canvas.drawCentredString(PAGE_W/2, top-title_h-7-subtitle_h-19, "Anonymous authors")
        else:
            canvas.setFont("Paper-Italic", 7.6)
            canvas.drawString(MARGIN, PAGE_H-22, title)
        if allow_draft:
            canvas.setFont("Paper", 7)
            canvas.drawCentredString(PAGE_W/2, 8, "INTERNAL LAYOUT DRAFT")

    def frames(first=False):
        height = PAGE_H-MARGIN-43-(title_area if first else 0)
        return [Frame(MARGIN+i*(COL+GUTTER), 42, COL, height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                      id=("first" if first else "normal")+str(i)) for i in range(2)]

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=A4, title=title, author="Anonymous authors",
                          pageCompression=1)
    doc.addPageTemplates([PageTemplate(id="first", frames=frames(True), onPage=decorate,
                                       autoNextPageTemplate="normal"),
                          PageTemplate(id="normal", frames=frames(), onPage=decorate)])
    story, in_references = [], False
    for element in elements:
        tag, plain = element.tag, "".join(element.itertext()).strip()
        if tag in ("h1", "h2", "h3", "h4"):
            if plain == "References": in_references=True
            story.append(Paragraph(inline(element), heading if tag in ("h1", "h2") else subheading))
        elif tag == "p":
            img = element.find("img")
            if img is not None:
                path = Path(img.attrib["src"])
                if not path.is_absolute(): path = (source.parent/path).resolve()
                image = Image(str(path))
                scale = min(COL/image.imageWidth, 295/image.imageHeight)
                image.drawWidth, image.drawHeight = image.imageWidth*scale, image.imageHeight*scale
                story.append(KeepTogether([Spacer(1,3), image, Spacer(1,3)]))
            elif plain:
                is_caption = re.match(r"(?:Table|Figure)\s+\d+[.:]", plain)
                paragraph = Paragraph(inline(element), references if in_references else caption if is_caption else body)
                if is_caption and plain.startswith("Table"): paragraph.keepWithNext=True
                story.append(paragraph)
        elif tag == "table":
            rows = element.findall(".//tr")
            cells = [[inline(cell) for cell in row] for row in rows]
            if not cells: continue
            n = len(cells[0])
            lengths = [max(min(len(re.sub("<[^>]*>", "", row[j])), 28) for row in cells) for j in range(n)]
            weights = [max(5, length)**0.7 for length in lengths]
            widths = [COL*weight/sum(weights) for weight in weights]
            first_header = re.sub("<[^>]*>", "", cells[0][0]).lower()
            if n > 2 and first_header in ("panel", "system", "model"):
                first_width = 53 if first_header == "panel" else 72
                widths = [first_width] + [(COL-first_width)*weight/sum(weights[1:]) for weight in weights[1:]]
            data = [[Paragraph(cell, small) for cell in row] for row in cells]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", spaceAfter=6)
            table.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 2.5), ("RIGHTPADDING", (0,0), (-1,-1), 2.5),
                ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EDF1F4")),
                ("LINEABOVE", (0,0), (-1,0), 0.7, colors.HexColor("#3E4851")),
                ("LINEBELOW", (0,0), (-1,0), 0.35, colors.HexColor("#7D8994")),
                ("LINEBELOW", (0,-1), (-1,-1), 0.7, colors.HexColor("#3E4851")),
            ]))
            story.append(table)
        elif tag in ("ul", "ol"):
            for index, item in enumerate(element.findall("li"), 1):
                prefix = str(index)+". " if tag=="ol" else "• "
                story.append(Paragraph(html.escape(prefix)+inline(item), body))
        elif tag == "pre":
            code_style = ParagraphStyle("code", fontName="Mono", fontSize=6.6, leading=8,
                                        spaceBefore=3, spaceAfter=5)
            story.append(XPreformatted(html.escape(plain), code_style, maxLineLength=55))
    doc.build(story)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"paper/manuscript/manuscript.md")
    parser.add_argument("--output", type=Path, default=ROOT/"output/pdf/svarah_reliability_paper.pdf")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, args.draft)
