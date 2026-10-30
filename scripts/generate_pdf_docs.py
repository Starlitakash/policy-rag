"""
scripts/generate_pdf_docs.py
----------------------------
Converts DOCUMENTATION.md into a professional, publication-ready PDF document
preserving all ASCII diagrams, architecture charts, code snippets, tables,
and typography using ReportLab.
"""

import os
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer,
    KeepTogether, Preformatted, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# Numbered Canvas for "Page X of Y" and Running Header
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        page_width, page_height = letter
        margin = 36  # 0.5 inch

        # Footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Footer line
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(margin, 28, page_width - margin, 28)
        
        # Footer text
        self.drawString(margin, 16, "AI-Based QA for Policy Documents • Technical Architecture & Documentation")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_width - margin, 16, page_str)

        # Header on pages 2+
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawString(margin, page_height - 24, "AI-Based QA for Policy Documents (RAG + Qdrant + Groq)")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#94A3B8"))
            self.drawRightString(page_width - margin, page_height - 24, "Enterprise & Organizational Policy Intelligence")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(margin, page_height - 28, page_width - margin, page_height - 28)

        self.restoreState()


def register_fonts():
    """Registers standard TrueType fonts available on Windows."""
    fonts = {
        'Consolas': 'C:/Windows/Fonts/consola.ttf',
        'Consolas-Bold': 'C:/Windows/Fonts/consolab.ttf',
        'Arial': 'C:/Windows/Fonts/arial.ttf',
        'Arial-Bold': 'C:/Windows/Fonts/arialbd.ttf',
        'Arial-Italic': 'C:/Windows/Fonts/ariali.ttf',
    }
    for name, path in fonts.items():
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception as e:
                print(f"Warning: Could not register font {name}: {e}")


def sanitize_text(text: str) -> str:
    """Replaces or escapes HTML and special characters for ReportLab Paragraphs."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Convert markdown bold and italics
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.*?)`', r'<font face="Consolas" color="#1E293B"><b>\1</b></font>', text)
    return text


def parse_markdown_table(lines: list[str]) -> list[list[str]]:
    """Parses markdown table lines into a 2D array of cleaned cell contents."""
    table_data = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        # Skip divider rows like |---|---|
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        table_data.append(cells)
    return table_data


def generate_pdf(input_md_path: str, output_pdf_path: str):
    """Parses DOCUMENTATION.md and generates DOCUMENTATION.pdf."""
    register_fonts()

    # Document Template with 0.5 inch margins (36 points)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=36
    )

    printable_width = letter[0] - 72  # 612 - 72 = 540 points

    # Styles
    styles = getSampleStyleSheet()

    body_font = 'Arial' if 'Arial' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
    mono_font = 'Consolas' if 'Consolas' in pdfmetrics.getRegisteredFontNames() else 'Courier'

    title_style = ParagraphStyle(
        'DocTitle',
        fontName=f'{body_font}-Bold' if f'{body_font}-Bold' in pdfmetrics.getRegisteredFontNames() else body_font,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName=body_font,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        fontName=body_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        fontName=f'{body_font}-Bold' if f'{body_font}-Bold' in pdfmetrics.getRegisteredFontNames() else body_font,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        fontName=f'{body_font}-Bold' if f'{body_font}-Bold' in pdfmetrics.getRegisteredFontNames() else body_font,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'SectionH3',
        fontName=f'{body_font}-Bold' if f'{body_font}-Bold' in pdfmetrics.getRegisteredFontNames() else body_font,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        fontName=body_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        fontName=body_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    mono_style = ParagraphStyle(
        'MonoBox',
        fontName=mono_font,
        fontSize=6.8,
        leading=8.6,
        textColor=colors.HexColor("#0F172A"),
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName=body_font,
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#1E293B")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName=f'{body_font}-Bold' if f'{body_font}-Bold' in pdfmetrics.getRegisteredFontNames() else body_font,
        fontSize=8.0,
        leading=10.5,
        textColor=colors.white
    )

    callout_style = ParagraphStyle(
        'Callout',
        fontName=body_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=4,
        spaceAfter=4
    )

    # Read markdown content
    with open(input_md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    story = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Horizontal Rule
        if stripped in ["---", "***", "___"]:
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#E2E8F0"), spaceBefore=3, spaceAfter=6))
            i += 1
            continue

        # Title (# )
        if line.startswith("# "):
            title_text = sanitize_text(stripped[2:].replace("📄", "").strip())
            story.append(Paragraph(title_text, title_style))
            i += 1
            continue

        # Subtitle (### ) at top or Section H3
        if line.startswith("### ") and i < 15:
            sub_text = sanitize_text(stripped[4:].strip())
            story.append(Paragraph(sub_text, subtitle_style))
            i += 1
            continue

        # Section H1 (## )
        if line.startswith("## "):
            h1_text = sanitize_text(stripped[3:].strip())
            story.append(Paragraph(h1_text, h1_style))
            i += 1
            continue

        # Section H2 (### )
        if line.startswith("### "):
            h2_text = sanitize_text(stripped[4:].strip())
            story.append(Paragraph(h2_text, h2_style))
            i += 1
            continue

        # Section H3 (#### )
        if line.startswith("#### "):
            h3_text = sanitize_text(stripped[5:].strip())
            story.append(Paragraph(h3_text, h3_style))
            i += 1
            continue

        # Code block or ASCII Diagram (``` ... ```)
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1  # skip closing ```

            code_text = "\n".join(code_lines)
            
            # Wrap in Preformatted with a shaded table container
            pre = Preformatted(code_text, mono_style)
            container_table = Table([[pre]], colWidths=[printable_width])
            container_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 3))
            story.append(container_table)
            story.append(Spacer(1, 5))
            continue

        # Markdown Table (| ... |)
        if line.startswith("|") and "|" in line[1:]:
            table_lines = [line]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            raw_table_data = parse_markdown_table(table_lines)
            if raw_table_data:
                num_cols = len(raw_table_data[0])
                
                # Determine proportional column widths
                if num_cols == 2:
                    col_widths = [printable_width * 0.32, printable_width * 0.68]
                elif num_cols == 3:
                    col_widths = [printable_width * 0.22, printable_width * 0.28, printable_width * 0.50]
                elif num_cols == 4:
                    col_widths = [printable_width * 0.22, printable_width * 0.24, printable_width * 0.24, printable_width * 0.30]
                elif num_cols == 5:
                    col_widths = [printable_width * 0.22, printable_width * 0.18, printable_width * 0.16, printable_width * 0.18, printable_width * 0.26]
                else:
                    col_widths = [printable_width / num_cols] * num_cols

                formatted_table_data = []
                for row_idx, row in enumerate(raw_table_data):
                    formatted_row = []
                    for c_idx, cell in enumerate(row):
                        cell_p_style = table_header_style if row_idx == 0 else table_cell_style
                        sanitized_cell = sanitize_text(cell)
                        formatted_row.append(Paragraph(sanitized_cell, cell_p_style))
                    formatted_table_data.append(formatted_row)

                t = Table(formatted_table_data, colWidths=col_widths, repeatRows=1)
                t_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]
                
                # Alternating row colors
                for r in range(1, len(raw_table_data)):
                    if r % 2 == 0:
                        t_style.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor("#F8FAFC")))
                    else:
                        t_style.append(('BACKGROUND', (0, r), (-1, r), colors.white))

                t.setStyle(TableStyle(t_style))
                story.append(Spacer(1, 3))
                story.append(t)
                story.append(Spacer(1, 5))
            continue

        # Bullet List Item (- or * or •)
        if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
            item_text = sanitize_text(stripped[2:].strip())
            bullet_char = "&bull;&nbsp;&nbsp;"
            story.append(Paragraph(f"{bullet_char}{item_text}", bullet_style))
            i += 1
            continue

        # Numbered List Item (1. 2. etc)
        match_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if match_num:
            num_idx = match_num.group(1)
            item_text = sanitize_text(match_num.group(2))
            story.append(Paragraph(f"<b>{num_idx}.</b>&nbsp;&nbsp;{item_text}", bullet_style))
            i += 1
            continue

        # Blockquote (> )
        if stripped.startswith("> "):
            quote_text = sanitize_text(stripped[2:].strip())
            callout_box = Table([[Paragraph(quote_text, callout_style)]], colWidths=[printable_width])
            callout_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#93C5FD")),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(Spacer(1, 3))
            story.append(callout_box)
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Regular Body Paragraph
        p_text = sanitize_text(stripped)
        story.append(Paragraph(p_text, body_style))
        i += 1

    # Build Document
    print(f"Building PDF '{output_pdf_path}' with {len(story)} flowable elements...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {output_pdf_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    md_file = base_dir / "DOCUMENTATION.md"
    pdf_file = base_dir / "DOCUMENTATION.pdf"

    generate_pdf(str(md_file), str(pdf_file))
