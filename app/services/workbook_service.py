from __future__ import annotations

from pathlib import Path

import pandas as pd

MSCORE_SHEET = "Mscore"
TEAM_COLUMN = "Team"
REQUIRED_COLUMNS = [TEAM_COLUMN, "Rank", "Mscore", "W", "L"]


def _clean_team_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy().dropna(how="all")

    if TEAM_COLUMN not in cleaned.columns:
        return cleaned

    cleaned[TEAM_COLUMN] = cleaned[TEAM_COLUMN].astype(str).str.strip()
    cleaned = cleaned[(cleaned[TEAM_COLUMN] != "") & (cleaned[TEAM_COLUMN].str.lower() != "nan")]
    return cleaned


def load_rankings(workbook_path: Path) -> pd.DataFrame:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    df = pd.read_excel(workbook_path, sheet_name=MSCORE_SHEET)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns in '{MSCORE_SHEET}': {missing_str}")

    rankings = _clean_team_frame(df)

    for column in rankings.columns:
        if column == TEAM_COLUMN:
            continue
        rankings[column] = pd.to_numeric(rankings[column], errors="coerce")

    rankings = rankings.dropna(subset=[TEAM_COLUMN])
    return rankings


def load_additional_team_sheets(workbook_path: Path) -> dict[str, pd.DataFrame]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    xls = pd.ExcelFile(workbook_path)
    sheets: dict[str, pd.DataFrame] = {}

    for sheet_name in xls.sheet_names:
        if sheet_name == MSCORE_SHEET:
            continue

        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        if TEAM_COLUMN not in df.columns:
            continue

        cleaned = _clean_team_frame(df)
        if cleaned.empty:
            continue

        sheets[sheet_name] = cleaned

    return sheets
