"""
Generates SYNTHETIC sample data (loosely modeled on Apple's public filing
structure, with made-up numbers) so the whole pipeline runs end-to-end
without needing to download real filings first.

To use REAL data instead: drop real 10-K PDFs and quarterly-financials
Excel files into data/raw/ (any public company), matching this naming
convention isn't required - just re-run scripts/build_index.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from openpyxl import Workbook

import config

os.makedirs(config.RAW_DATA_DIR, exist_ok=True)


def _write_pdf_pages(path: str, pages: list[str]):
    c = canvas.Canvas(path, pagesize=letter)
    for page_text in pages:
        y = 750
        for line in page_text.split("\n"):
            # wrap long lines crudely
            while len(line) > 100:
                c.drawString(50, y, line[:100])
                line = line[100:]
                y -= 15
            c.drawString(50, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                y = 750
        c.showPage()
    c.save()


def make_10k(year: str, revenue: str, net_income: str, headcount: str, comp_expense: str) -> list[str]:
    page1 = f"""ANNUAL REPORT (FORM 10-K) - FISCAL YEAR {year}
Item 1. Business Overview

The Company designs, manufactures, and markets smartphones, personal computers,
tablets, wearables, and accessories, and sells a range of related services.
Our strategic priorities for fiscal {year} included expanding our Services
segment, deepening our silicon roadmap, and pursuing selective acquisitions
to strengthen our product ecosystem. Management believes the competitive
landscape remains intense, particularly in the smartphone and tablet
categories, and continues to invest in differentiation through hardware,
software, and services integration.

Item 7. Management's Discussion and Analysis

Total net sales for fiscal {year} were {revenue}, compared to the prior year.
Net income for fiscal {year} was {net_income}. Gross margin remained strong,
driven by growth in our Services segment and disciplined cost management in
our Products segment. Segment results: iPhone revenue grew driven by strong
demand; Mac and iPad revenue were roughly flat year over year; Wearables,
Home and Accessories revenue increased on strong unit sales; Services revenue
grew double digits, reflecting an expanding installed base.
"""
    page2 = f"""ANNUAL REPORT (FORM 10-K) - FISCAL YEAR {year}
Item 8. Financial Statements and Supplementary Data - Human Capital

As of the end of fiscal {year}, the Company had approximately {headcount}
full-time equivalent employees worldwide. Total compensation and benefits
expense, including salaries, wages, bonuses, and stock-based compensation,
was approximately {comp_expense} for fiscal {year}. The Company offers
competitive compensation packages designed to attract and retain talent in a
highly competitive labor market for engineering and design talent.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk - Outlook

Looking ahead, management expects continued investment in research and
development, with an outlook toward modest revenue growth in the following
fiscal year, subject to macroeconomic conditions, foreign exchange
headwinds, and component supply constraints. The Company anticipates
Services revenue will continue to be a key growth driver going into next
fiscal year.
"""
    return [page1, page2]


def make_quarterly_excel(path: str, quarter_label: str, segment_revenue: dict, headcount: int, comp_expense_musd: int):
    wb = Workbook()

    ws1 = wb.active
    ws1.title = "Revenue by Segment"
    ws1.append(["Quarter", "Segment", "Revenue (USD millions)"])
    for segment, revenue in segment_revenue.items():
        ws1.append([quarter_label, segment, revenue])

    ws2 = wb.create_sheet("Headcount and Compensation")
    ws2.append(["Quarter", "Metric", "Value"])
    ws2.append([quarter_label, "Full-time employees", headcount])
    ws2.append([quarter_label, "Total compensation expense (USD millions)", comp_expense_musd])
    ws2.append([quarter_label, "Stock-based compensation (USD millions)", round(comp_expense_musd * 0.18)])

    wb.save(path)


def main():
    # --- 3 years of synthetic 10-Ks (PDF) ---
    fy_data = [
        ("2022", "$394,328 million", "$99,803 million", "164,000", "$22,800 million"),
        ("2023", "$383,285 million", "$96,995 million", "161,000", "$24,900 million"),
        ("2024", "$391,035 million", "$93,736 million", "150,000", "$26,300 million"),
    ]
    for year, revenue, net_income, headcount, comp in fy_data:
        pages = make_10k(year, revenue, net_income, headcount, comp)
        out_path = os.path.join(config.RAW_DATA_DIR, f"sample_10k_FY{year}.pdf")
        _write_pdf_pages(out_path, pages)
        print(f"Wrote {out_path}")

    # --- Quarterly Excel financials ---
    quarters = [
        ("Q1_2024", {"iPhone": 69700, "Mac": 7780, "iPad": 7020, "Wearables": 11950, "Services": 20770}, 152000, 6300),
        ("Q2_2024", {"iPhone": 45960, "Mac": 6100, "iPad": 5560, "Wearables": 7910, "Services": 21200}, 151000, 6100),
        ("Q3_2024", {"iPhone": 39300, "Mac": 6610, "iPad": 5790, "Wearables": 8100, "Services": 21870}, 150500, 6050),
        ("Q4_2024", {"iPhone": 46220, "Mac": 7740, "iPad": 6950, "Wearables": 9040, "Services": 24960}, 150000, 6200),
    ]
    for label, seg_rev, headcount, comp in quarters:
        out_path = os.path.join(config.RAW_DATA_DIR, f"sample_financials_{label}.xlsx")
        make_quarterly_excel(out_path, label.replace("_", " "), seg_rev, headcount, comp)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
