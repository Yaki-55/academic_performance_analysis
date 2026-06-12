import pandas as pd
import numpy as np


def sanitize_grade_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up structural inconsistencies
    and filters out corrupt database records.
    Converts values less than 0 into proper NaN types.

    Args:
        raw_df (pd.DataFrame): raw_grade_history

    Returns:
        pd.DataFrame: cleaned_grade_history
    """
    print("Sanitizing data records...")
    cleaned_df = raw_df.copy()

    # 1. Target columns to search for invalid < 0 values
    academic_metrics = [
        "p1",
        "p2",
        "p3",
        "o",
        "pf",
        "e1",
        "e2",
        "esp",
        "a1",
        "a2",
        "a3",
        "oa",
        "pa",
    ]

    # 2. Convert any database placeholder (< 0) to standard NaN values
    for metric in academic_metrics:
        if metric in cleaned_df.columns:
            cleaned_df.loc[cleaned_df[metric] < 0, metric] = np.nan

    # 3. Drop rows that do not have a valid student identifier (matricula)
    cleaned_df = cleaned_df.dropna(subset=["matricula"])

    print("Data sanitization step complete.")
    return cleaned_df
