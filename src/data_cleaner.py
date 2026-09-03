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


def resolve_blocked_parcial_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distingue, para cada parcial (P1/P2/P3), entre "el parcial aún no ocurre" y
    "el parcial ocurrió pero la calificación quedó bloqueada" (p. ej. por no
    entregar el reporte de lectura exigido antes de cada periodo de exámenes
    parciales -- Artículos 29-XIII y 50-II del Reglamento de Estudiantes de
    Licenciatura de la UTM). Debe ejecutarse DESPUÉS de sanitize_grade_history.

    Regla verificada empíricamente contra la base de datos: cuando la
    calificación de un parcial es NaN (originalmente negativa) pero la
    asistencia registrada de ESE MISMO parcial es un valor real, el parcial sí
    ocurrió -- el estudiante asistió pero por algún motivo administrativo la
    calificación no se asentó -- y la calificación bloqueada se resuelve como un
    0 real (reprobado), no como dato faltante. Si tanto la calificación como la
    asistencia del parcial son NaN, el parcial genuinamente no ha ocurrido
    todavía (patrón concentrado casi en su totalidad en el periodo más reciente
    del dataset).
    """
    print("Resolviendo parciales bloqueados (reporte de lectura no entregado, etc.)...")
    resolved_df = df.copy()

    for calif_col, asistencia_col in [("p1", "a1"), ("p2", "a2"), ("p3", "a3")]:
        bloqueado = resolved_df[calif_col].isna() & resolved_df[asistencia_col].notna()
        resolved_df.loc[bloqueado, calif_col] = 0.0

    return resolved_df
