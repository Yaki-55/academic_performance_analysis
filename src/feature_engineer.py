import hashlib
import pandas as pd
from config.settings import PERIOD_NAMES, PERIOD_ORDER


def generate_student_major_hash(matricula: str, id_carrera: int) -> str:
    """Combines matricula and major ID into a secure, unique hash string."""
    combined_string = f"{str(matricula).strip()}_{str(id_carrera)}"
    return hashlib.sha256(combined_string.encode("utf-8")).hexdigest()


def enrich_academic_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Adds academic context using Spanish column names for consistency."""
    print("Enriching dataset with academic time horizons...")
    enriched_df = df.copy()

    # Map raw numeric periods to readable tags
    period_mapping = dict(zip(PERIOD_ORDER, PERIOD_NAMES))
    enriched_df["nombre_periodo"] = enriched_df["periodo"].map(period_mapping)

    # Parse timeline details into Spanish keys
    enriched_df["año_academico"] = enriched_df["nombre_periodo"].str[:9]
    enriched_df["tipo_periodo"] = enriched_df["nombre_periodo"].str[9:]
    enriched_df["tipo_semestre"] = enriched_df["semestre"].apply(
        lambda x: "Par" if x % 2 == 0 else "Impar"
    )
    enriched_df["es_periodo_regular"] = enriched_df["tipo_periodo"].isin(["A", "B"])

    # Create the unified, anonymized student profile tracking token
    enriched_df["alumno_carrera_hash"] = enriched_df.apply(
        lambda row: generate_student_major_hash(row["matricula"], row["id_carrera"]),
        axis=1,
    )

    return enriched_df


def apply_graduation_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and marks students who successfully completed the 10th semester."""
    print("Computing student retention targets (Graduated vs. Dropout)...")

    # Determine the absolute highest semester achieved per unique student profile
    max_semester_per_student = df.groupby("alumno_carrera_hash")["semestre"].max()
    graduated_hashes = set(max_semester_per_student[max_semester_per_student >= 10].index)

    # Label the tracking column (1 = Término, 0 = Desertó)
    df["resultado_final"] = df["alumno_carrera_hash"].isin(graduated_hashes).astype(int)
    return df


def build_progress_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms transactional grade rows into cumulative historical snapshots per semester.
    Imputes missing academic values with 0 for predictive model alignment.
    Also categorizes attendance into 5 risk blocks.
    """
    print("Building progress snapshots for model processing...")

    # Fill remaining NaNs to ready data for the mathematical aggregations
    fill_cols = ["p1", "p2", "p3", "o", "pf", "e1", "e2", "esp", "a1", "a2", "a3", "oa", "pa"]
    df[fill_cols] = df[fill_cols].fillna(0)

    snapshot_records = []
    grouped_students = df.groupby("alumno_carrera_hash")

    for _, student_history in grouped_students:
        target_status = student_history["resultado_final"].iloc[0]
        sorted_history = student_history.sort_values(by="periodo")

        for active_semester in sorted(sorted_history["semestre"].unique()):
            # Isolate cumulative data up to the current evaluated snapshot semester
            cumulative_window = sorted_history[sorted_history["semestre"] <= active_semester]

            snapshot = {
                "semestre_actual": active_semester,
                "promedio_calificacion_final": cumulative_window["pf"].mean(),
                "promedio_asistencia_final": cumulative_window["pa"].mean(),
                "materias_cursadas": cumulative_window.shape[0],
                "materias_reprobadas": (cumulative_window["pf"] < 6.0).sum(),
                "periodos_verano_cursados": (cumulative_window["tipo_periodo"] == "V").nunique(),
                "std_calificacion_final": cumulative_window["pf"].std(ddof=0),
                "resultado_final": target_status,
            }
            snapshot_records.append(snapshot)

    # Convertir a DataFrame y rellenar nulos derivados de varianzas sin suficientes datos
    snapshot_df = pd.DataFrame(snapshot_records).fillna(0)

    # --- NUEVO: Convertir Asistencias a Datos Categóricos ---
    print("--- Convirtiendo Asistencias a Categorías de Riesgo ---")
    bins = [-1, 79.99, 84.99, 89.99, 94.99, 101]
    labels = [
        "1. Riesgo (<80%)",
        "2. Regular (80-84%)",
        "3. Bueno (85-89%)",
        "4. Muy Bueno (90-94%)",
        "5. Excelente (>=95%)",
    ]

    snapshot_df["categoria_asistencia"] = pd.cut(
        snapshot_df["promedio_asistencia_final"], bins=bins, labels=labels, right=True
    )

    return snapshot_df
