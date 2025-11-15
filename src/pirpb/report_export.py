"""
report_export.py
----------------
Stage G — Final PDF compilation of metrics and plots.
Combines summary stats and figures into results/final_report.pdf.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd


def _table_from_csv(path: Path, title: str):
    if not path.exists():
        return [Paragraph(f"<b>{title}:</b> File not found ({path})", getSampleStyleSheet()["Normal"])]
    df = pd.read_csv(path)
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    return [Paragraph(f"<b>{title}</b>", getSampleStyleSheet()["Heading4"]), table, Spacer(1, 12)]


def main():
    project_root = Path(__file__).resolve().parents[2]
    results = project_root / "results"
    out_path = results / "final_report.pdf"

    doc = SimpleDocTemplate(str(out_path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Search Perturbation Robustness and Bias (PIR + PB)</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Section 1: PIR summary
    elements += _table_from_csv(results / "pir_google.csv", "Stage D – Global PIR Results")

    # Section 2: Item-level PIR
    elements += _table_from_csv(results / "item_pir_summary.csv", "Stage E – Item-Level PIR Summary")

    # Section 3: PB summary
    elements += _table_from_csv(results / "pb_summary.csv", "Stage F – Perturbation Bias Summary")

    # Add images
    for img_name in ["pir_hist_google.png", "pb_bar_google.png"]:
        img_path = results / img_name
        if img_path.exists():
            elements.append(Spacer(1, 12))
            elements.append(Image(str(img_path), width=400, height=250))
            elements.append(Paragraph(f"<i>{img_name}</i>", styles["Normal"]))
            elements.append(Spacer(1, 12))

    elements.append(Paragraph("Report auto-generated from project pipeline.", styles["Italic"]))
    doc.build(elements)
    print(f"[OK] Final report exported → {out_path}")


if __name__ == "__main__":
    main()
