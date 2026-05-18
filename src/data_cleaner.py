import pandas as pd
import numpy as np


def sanitize_grade_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up structural inconsistencies
    and filters out corrupt database records.
    Converts -10 values into proper NaN types.

    Args:
        raw_df (pd.DataFrame): raw_grade_history

    Returns:
        pd.DataFrame: cleaned_grade_history
    """
    print("Sanitizing data records...")
    cleaned_df = raw_df.copy()

    # 1. Target columns to search for invalid -10 values
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

    # 2. Convert database placeholder -10 to standard NaN values
    for metric in academic_metrics:
        if metric in cleaned_df.columns:
            cleaned_df[metric] = cleaned_df[metric].replace(-10.0, np.nan)

    # 3. Drop rows that do not have a valid student identifier (matricula)
    cleaned_df = cleaned_df.dropna(subset=["matricula"])

    print("Data sanitization step complete.")
    return cleaned_df
