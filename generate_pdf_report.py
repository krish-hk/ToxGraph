"""Render REPORT.md as a four-page PDF; no model execution or duplicated report prose."""
import argparse
from html import escape
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parent


def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)
    return text


def build_pdf_report(output_filename=None):
    output = Path(output_filename or ROOT / "REPORT.pdf")
    styles = {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.4, leading=12.4, spaceAfter=7),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24, leading=27, spaceAfter=7),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceAfter=9),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=5, spaceAfter=6, keepWithNext=True),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.2, leading=11),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=12, spaceAfter=8),
    }
    story = []
    lines = (ROOT / "REPORT.md").read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            story.append(Preformatted("\n".join(block), styles["code"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [x.strip() for x in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", x) for x in row):
                    rows.append([Paragraph(inline(x), styles["cell"]) for x in row])
                i += 1
            table = Table(rows, colWidths=[190, 72, 45, 90, 90], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8EDF2")),
                ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#D9D9D9")),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.extend([table, Spacer(1,8)])
            continue
        elif line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            path = ROOT / match.group(2)
            if not path.is_file():
                raise FileNotFoundError(path)
            img = Image(str(path))
            img.drawHeight = 475 * img.imageHeight / img.imageWidth
            img.drawWidth = 475
            story.append(img)
        else:
            kind = "title" if line.startswith("# ") else "subtitle" if line.startswith("## ") else "heading" if line.startswith("### ") else "body"
            text = re.sub(r"^#{1,3} ", "", line)
            # Built-in Helvetica lacks arrow/en-dash glyphs reliably on all viewers.
            text = text.replace("→", " / ").replace("–", "-").replace("×", "x")
            story.append(Paragraph(inline(text), styles[kind]))
        i += 1

    def footer(canvas, doc):
        if doc.page > 4:
            raise ValueError("Report exceeds the hackathon four-page limit")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#505050"))
        canvas.drawString(48, 27, "ToxGraph | Track 3 | Frozen benchmark")
        canvas.drawRightString(A4[0]-48, 27, f"{doc.page} / 4")

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
                            topMargin=40, bottomMargin=42, title="ToxGraph - Track 3 Scientific Report")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Built {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "REPORT.pdf"))
    build_pdf_report(parser.parse_args().output)
