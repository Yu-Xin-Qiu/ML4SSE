"""Build the formatted SI descriptor workbook with pandas/openpyxl."""
from pathlib import Path
from copy import copy

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "final_si_tables" / "SI_descriptor_dictionary_RF_RFE_final.csv"
XLSX = ROOT / "reports" / "final_si_tables" / "SI_descriptor_dictionary_RF_RFE_final.xlsx"
SELECTED_CSV = ROOT / "reports" / "final_si_tables" / "SI_selected_116_descriptors.csv"
SELECTED_XLSX = ROOT / "reports" / "final_si_tables" / "SI_selected_116_descriptors.xlsx"


def write_workbook(table, path, sheet_name):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.book[sheet_name]
        thin = Side(style="thin", color="E2E2E2")
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="3C5488")
            cell.font = Font(name="Arial", size=9, bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows():
            for cell in row:
                font = copy(cell.font); font.name = "Arial"; font.sz = 9; cell.font = font
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        for column in ws.columns:
            letter = column[0].column_letter
            width = min(55, max(10, max(len(str(c.value or "")) for c in column) + 2))
            ws.column_dimensions[letter].width = width
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    print(f"saved {path}")


def main():
    table = pd.read_csv(CSV)
    write_workbook(table, XLSX, "All descriptors")
    selected = table[table["selected_final"].astype(bool)].copy()
    if len(selected) != 116:
        raise ValueError(f"Expected 116 selected descriptors, found {len(selected)}")
    selected.to_csv(SELECTED_CSV, index=False, encoding="utf-8-sig")
    write_workbook(selected, SELECTED_XLSX, "Selected descriptors")


if __name__ == "__main__":
    main()
