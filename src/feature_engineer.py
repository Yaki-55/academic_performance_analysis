import os
import re
import hashlib
import unicodedata
import pandas as pd
import numpy as np
from config.settings import PERIOD_NAMES, PERIOD_ORDER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def apply_graduation_labels(
    df: pd.DataFrame, grace_periods: int = 3, ventana_reglamentaria_periodos: int = 21
) -> pd.DataFrame:
    """
    Determina el resultado institucional final POR ALUMNO (matrícula), no por carrera:
    un alumno que se cambia de carrera y termina la segunda cuenta como retenido, no
    como desertor de la primera. La deserción se define a nivel institución porque es
    la pregunta que plantea la tesis, no a nivel programa.

    Un alumno que aún no se ha graduado se marca como CENSURADO (resultado todavía
    desconocido, se excluye del entrenamiento) solo si se cumplen A LA VEZ dos
    condiciones -- de lo contrario se considera un desertor confirmado:

    1. Actividad reciente: su última actividad cae dentro de `grace_periods`
       respecto al periodo más reciente del dataset (podría seguir activo).
    2. Ventana reglamentaria vigente: no ha excedido el tiempo límite de
       permanencia como estudiante, fijado por el Artículo 23 del Reglamento de
       Estudiantes de Licenciatura de la UTM en "el plan de estudios más dos años
       adicionales" (~7 años, aproximado aquí como `ventana_reglamentaria_periodos`
       periodos A/B/V desde su primer periodo de actividad). Si ya se excedió,
       por reglamento la universidad ya lo habría dado de baja definitiva
       (Artículo 19-II), así que aunque haya estado activo recientemente ya no
       calza como "resultado desconocido".
    """
    print("Computing institutional-level retention targets (Graduated vs. Dropout)...")

    period_position = {periodo: idx for idx, periodo in enumerate(PERIOD_ORDER)}
    period_idx = df["periodo"].map(period_position)
    dataset_last_idx = period_idx.max()

    student_stats = (
        df.assign(_periodo_idx=period_idx)
        .groupby("matricula")
        .agg(
            max_semestre=("semestre", "max"),
            primer_periodo_idx=("_periodo_idx", "min"),
            ultimo_periodo_idx=("_periodo_idx", "max"),
            carreras_cursadas=("id_carrera", "nunique"),
        )
    )

    student_stats["graduo"] = student_stats["max_semestre"] >= 10
    actividad_reciente = (
        dataset_last_idx - student_stats["ultimo_periodo_idx"]
    ) < grace_periods
    dentro_de_ventana_reglamentaria = (
        dataset_last_idx - student_stats["primer_periodo_idx"]
    ) < ventana_reglamentaria_periodos
    student_stats["es_censurado"] = (
        (~student_stats["graduo"]) & actividad_reciente & dentro_de_ventana_reglamentaria
    )
    student_stats["cambio_carrera"] = student_stats["carreras_cursadas"] > 1
    student_stats["resultado_final"] = student_stats["graduo"].astype(int)

    df = df.merge(
        student_stats[["resultado_final", "es_censurado", "cambio_carrera"]],
        left_on="matricula",
        right_index=True,
        how="left",
    )

    n_total = student_stats.shape[0]
    n_censurados = int(student_stats["es_censurado"].sum())
    n_cambio = int(student_stats["cambio_carrera"].sum())
    print(f"Alumnos totales: {n_total} | Censurados (resultado aún desconocido): {n_censurados}")
    print(f"Alumnos con cambio de carrera detectado: {n_cambio}")

    return df


def flag_career_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para alumnos que cambiaron de carrera, identifica cuál(es) carrera(s) fueron
    abandonadas (todas menos la última cursada cronológicamente) y mide el rendimiento
    académico justo antes del cambio (promedio de calificación y asistencia en los
    últimos 2 periodos activos de esa carrera).

    Esto no asume un motivo para el cambio: convierte "¿el cambio fue por bajo
    rendimiento o por preferencia?" en una señal medible a partir de los propios
    registros académicos, disponible para las pruebas de asociación de la Fase 2.

    Debe ejecutarse después de `apply_graduation_labels` (requiere la columna
    `cambio_carrera`).
    """
    print("Analizando historial de cambios de carrera...")
    df = df.copy()

    period_position = {periodo: idx for idx, periodo in enumerate(PERIOD_ORDER)}
    df["_periodo_idx"] = df["periodo"].map(period_position)

    # La carrera "vigente" de cada alumno es aquella con actividad más reciente
    hash_last_activity = df.groupby("alumno_carrera_hash").agg(
        matricula=("matricula", "first"),
        ultimo_periodo_idx=("_periodo_idx", "max"),
    )
    hashes_vigentes = set(
        hash_last_activity.groupby("matricula")["ultimo_periodo_idx"].idxmax()
    )

    df["carrera_abandonada"] = df["cambio_carrera"] & (
        ~df["alumno_carrera_hash"].isin(hashes_vigentes)
    )

    def _rendimiento_previo_al_cambio(grupo: pd.DataFrame, n_periodos: int = 2) -> pd.Series:
        ultimos_periodos = grupo["_periodo_idx"].drop_duplicates().nlargest(n_periodos)
        ventana = grupo[grupo["_periodo_idx"].isin(ultimos_periodos)]
        return pd.Series(
            {
                "pf_previo_a_cambio": ventana["pf"].mean(),
                "pa_previo_a_cambio": ventana["pa"].mean(),
            }
        )

    if df["carrera_abandonada"].any():
        rendimiento_abandono = (
            df[df["carrera_abandonada"]]
            .groupby("alumno_carrera_hash")
            .apply(_rendimiento_previo_al_cambio, include_groups=False)
        )
        df = df.merge(rendimiento_abandono, on="alumno_carrera_hash", how="left")
    else:
        df["pf_previo_a_cambio"] = np.nan
        df["pa_previo_a_cambio"] = np.nan

    df = df.drop(columns=["_periodo_idx"])

    n_abandonadas = df.loc[df["carrera_abandonada"], "alumno_carrera_hash"].nunique()
    print(f"Carreras abandonadas por cambio detectadas: {n_abandonadas}")

    return df


_CONECTORES_MATERIA = {
    "DE", "DEL", "LA", "LAS", "EL", "LOS", "PARA", "Y", "EN", "A", "AL",
    "CON", "SU", "SUS", "UNA", "UN",
}


def _normalizar_nombre_materia(nombre: str) -> str:
    """
    Normaliza el nombre de una materia (mayúsculas, sin acentos, sin
    puntuación, sin conectores) para poder comparar coincidencias EXACTAS
    entre periodos. No resuelve renombres semánticos (ej. "Física 1" ->
    "Física para la Ingeniería 1") -- eso requeriría una tabla de
    equivalencias oficial que no existe en la base de datos; se prefiere
    subestimar el recursamiento en esos casos raros a arriesgar falsos
    positivos por coincidencia difusa de palabras genéricas (ver nota en
    flag_subject_approval).
    """
    if pd.isna(nombre):
        return ""
    texto = unicodedata.normalize("NFKD", str(nombre).upper())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^A-Z0-9 ]", " ", texto)
    tokens = [t for t in texto.split() if t not in _CONECTORES_MATERIA]
    return " ".join(tokens)


def flag_subject_approval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determina si una inscripción a materia fue aprobada, siguiendo las reglas del
    Reglamento de Estudiantes de Licenciatura de la UTM:

    - Vía ordinario: calificación final >= 6 Y asistencia >= 85%
      (Artículos 52 y 29-XI/50-I).
    - Vía examen extraordinario (1 o 2): calificación >= 6 (Artículos 62-65).
    - Vía examen especial: calificación >= 6, y solo aplica después de recursar
      la asignatura (Artículo 69).

    También marca `es_recursamiento`: si el alumno ya había cursado antes una
    materia con el MISMO NOMBRE NORMALIZADO (no el mismo `id_materia`: se
    verificó contra la base de datos que `id_materia` se regenera en cada
    periodo/oferta, por lo que casi nunca coincide entre un curso y su
    recursamiento -- es una deuda técnica del sistema de origen, no algo que
    podamos corregir ahí). La comparación de nombres es EXACTA tras normalizar
    mayúsculas/acentos/conectores, deliberadamente sin similitud difusa: se
    probó con coincidencia por tokens (Jaccard/subconjunto) y produjo fusiones
    incorrectas (ej. "Cálculo Diferencial" con "Cálculo Integral", "Estructuras
    de Acero" con "Estructuras de Concreto") al compartir palabras genéricas.
    Sin una tabla de equivalencias oficial, se prefiere subestimar el
    recursamiento en casos de renombre semántico real que arriesgar falsos
    positivos.

    Debe ejecutarse después de `resolve_blocked_parcial_grades` para que `pf`
    refleje correctamente los parciales bloqueados.
    """
    print("Determinando aprobación de materias según reglamento institucional...")
    df = df.copy()

    df["_materia_normalizada"] = df["nombre_materia"].apply(_normalizar_nombre_materia)

    period_position = {periodo: idx for idx, periodo in enumerate(PERIOD_ORDER)}
    orden_cronologico = df["periodo"].map(period_position)
    df = df.assign(_orden_cronologico=orden_cronologico).sort_values(
        ["matricula", "_materia_normalizada", "_orden_cronologico"]
    )

    df["es_recursamiento"] = df.duplicated(
        subset=["matricula", "_materia_normalizada"], keep="first"
    ) & (df["_materia_normalizada"] != "")

    aprobo_ordinario = (df["pf"] >= 6) & (df["pa"] >= 85)
    aprobo_extraordinario = (df["e1"] >= 6) | (df["e2"] >= 6)
    aprobo_especial = (df["esp"] >= 6) & df["es_recursamiento"]

    df["aprobo_materia"] = aprobo_ordinario | aprobo_extraordinario | aprobo_especial

    df = df.drop(columns=["_orden_cronologico", "_materia_normalizada"]).sort_index()

    n_recursamientos = int(df["es_recursamiento"].sum())
    n_aprobadas = int(df["aprobo_materia"].sum())
    print(f"Inscripciones en recursamiento detectadas: {n_recursamientos}")
    print(f"Inscripciones aprobadas (de {len(df)} totales): {n_aprobadas}")

    return df


def merge_subject_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the static CSV dictionary
    of categorized subjects into the main dataframe.
    """
    print("Integrando diccionario de categorías de conocimiento...")
    mapping_path = os.path.join(
        BASE_DIR, "data", "mappings", "materias_clasificadas.csv"
    )

    try:
        df_mapeo = pd.read_csv(mapping_path)
        df_con_categorias = pd.merge(df, df_mapeo, on="id_materia", how="left")

        df_con_categorias["categoria_materia"] = df_con_categorias[
            "categoria_materia"
        ].fillna("Otra")
        return df_con_categorias
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el mapeo en {mapping_path}.")
        print("Ejecuta 'python scripts/build_subject_mapping.py' primero.")
        df["categoria_materia"] = "Otra"
        return df


def build_progress_snapshots_with_granular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms transactional rows into cumulative snapshots per semester.
    Generates granular features per Subject Category (e.g., GPA in Math).
    """
    print("Building progress snapshots for model processing...")

    fill_cols = [
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
    df[fill_cols] = df[fill_cols].fillna(0)

    categorias_conocimiento = df["categoria_materia"].unique()

    # Materias distintas por categoria de conocimiento, contadas POR CARRERA:
    # usar un total unico para toda la universidad sobre-representaria categorias
    # que no existen en algunas carreras y distorsionaria la exposicion real de
    # un alumno a su propio plan de estudios. Limitacion conocida: la base de
    # datos no distingue version de plan de estudios dentro de una misma
    # carrera, asi que se mezclan todas las eras historicas de esa carrera.
    materias_por_categoria_carrera = (
        df.groupby(["id_carrera", "categoria_materia"])["id_materia"].nunique().to_dict()
    )

    snapshot_records = []
    grouped_students = df.groupby("alumno_carrera_hash")

    for _, student_history in grouped_students:
        target_status = student_history["resultado_final"].iloc[0]
        censurado_status = student_history["es_censurado"].iloc[0]
        cambio_carrera_status = student_history["cambio_carrera"].iloc[0]
        carrera_abandonada_status = student_history["carrera_abandonada"].iloc[0]
        pf_previo_a_cambio = student_history["pf_previo_a_cambio"].iloc[0]
        pa_previo_a_cambio = student_history["pa_previo_a_cambio"].iloc[0]
        id_carrera_status = student_history["id_carrera"].iloc[0]
        sorted_history = student_history.sort_values(by="periodo")
        cohorte_ingreso_status = sorted_history["año_academico"].iloc[0]

        for active_semester in sorted(sorted_history["semestre"].unique()):
            cumulative_window = sorted_history[
                sorted_history["semestre"] <= active_semester
            ]
            materias_este_semestre = sorted_history[
                sorted_history["semestre"] == active_semester
            ]

            snapshot = {
                "semestre_actual": active_semester,
                "promedio_calificacion_final": cumulative_window["pf"].mean(),
                "promedio_asistencia_final": cumulative_window["pa"].mean(),
                "materias_cursadas_totales": cumulative_window.shape[0],
                "materias_reprobadas_totales": (cumulative_window["pf"] < 6.0).sum(),
                "periodos_verano_cursados": (
                    cumulative_window["tipo_periodo"] == "V"
                ).nunique(),
                "std_calificacion_final": cumulative_window["pf"].std(ddof=0),
                "resultado_final": target_status,
                "es_censurado": censurado_status,
                "cambio_carrera": cambio_carrera_status,
                "carrera_abandonada": carrera_abandonada_status,
                "pf_previo_a_cambio": pf_previo_a_cambio,
                "pa_previo_a_cambio": pa_previo_a_cambio,
                "id_carrera": id_carrera_status,
                "cohorte_ingreso": cohorte_ingreso_status,
                # Target de corto plazo: aprobo TODAS las materias de ESTE
                # semestre segun las reglas institucionales (ver
                # flag_subject_approval), no acumulado con semestres previos.
                "materias_reprobadas_este_semestre": (
                    ~materias_este_semestre["aprobo_materia"]
                ).sum(),
                "aprobo_semestre": bool(materias_este_semestre["aprobo_materia"].all()),
            }

            for cat in categorias_conocimiento:
                safe_cat_name = cat.replace(" y ", "_").replace(" ", "_").lower()

                cat_data = cumulative_window[
                    cumulative_window["categoria_materia"] == cat
                ]

                # Cobertura: fraccion de las materias distintas de esta categoria,
                # dentro del plan de SU carrera, que el alumno ya curso hasta este
                # punto. A diferencia de un corte por semestre minimo, distingue
                # directamente a un alumno que curso 1 de 5 materias tempranas de
                # una categoria de otro que ya curso las 5 -- sin necesidad de
                # inferir en que semestre "empieza" la categoria.
                total_categoria_carrera = materias_por_categoria_carrera.get(
                    (id_carrera_status, cat), 0
                )
                materias_cursadas_categoria = cat_data["id_materia"].nunique()
                if total_categoria_carrera > 0:
                    cobertura = min(
                        materias_cursadas_categoria / total_categoria_carrera, 1.0
                    )
                else:
                    cobertura = 0.0
                snapshot[f"cobertura_{safe_cat_name}"] = cobertura

                if cat_data.empty:
                    snapshot[f"promedio_pf_{safe_cat_name}"] = 0.0
                    snapshot[f"materias_reprobadas_{safe_cat_name}"] = 0
                else:
                    snapshot[f"promedio_pf_{safe_cat_name}"] = cat_data["pf"].mean()
                    snapshot[f"materias_reprobadas_{safe_cat_name}"] = (
                        cat_data["pf"] < 6.0
                    ).sum()

            snapshot_records.append(snapshot)

    snapshot_df = pd.DataFrame(snapshot_records).fillna(0)

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


# Momentos de corte ("landmarks") dentro de un semestre en curso: tutorías llama a
# cada alumno tres veces por semestre, una por parcial, antes de que exista el
# ordinario. En cada landmark solo se conocen los parciales ya presentados.
_LANDMARKS_PARCIAL = [
    (1, ["p1"], ["a1"]),
    (2, ["p1", "p2"], ["a1", "a2"]),
    (3, ["p1", "p2", "p3"], ["a1", "a2", "a3"]),
]


def build_landmark_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera fotografías a nivel PARCIAL (no solo por semestre completo): al
    concluir P1, P1+P2 y P1+P2+P3 de cada semestre en curso -- los 3 momentos
    en que Tutorías llama a cada alumno para revisar su rendimiento, antes del
    periodo de evaluación ordinaria.

    Distingue dos fuentes de información en cada fotografía:
    - Semestres YA CONCLUIDOS: usan la calificación final real (`pf`) y la
      asistencia final real (`pa`), como en `build_progress_snapshots_with_
      granular_features`.
    - El semestre EN CURSO: como el ordinario todavía no ha ocurrido en
      ninguno de los 3 landmarks, se usa el promedio de los parciales ya
      presentados (`p1`, o promedio(p1,p2), o promedio(p1,p2,p3)) como proxy
      del desempeño hasta ese momento -- NO es la calificación final.

    El target de corto plazo (`aprobo_semestre`, `materias_reprobadas_este_
    semestre`) se calcula con el `aprobo_materia` REAL (el desenlace conocido
    solo en retrospectiva) -- es la etiqueta a predecir, nunca debe filtrarse
    hacia las variables predictoras del landmark.
    """
    print("Building parcial-level landmark snapshots for early-warning modeling...")

    fill_cols = [
        "p1", "p2", "p3", "o", "pf", "e1", "e2", "esp", "a1", "a2", "a3", "oa", "pa",
    ]
    df[fill_cols] = df[fill_cols].fillna(0)

    categorias_conocimiento = df["categoria_materia"].unique()

    materias_por_categoria_carrera = (
        df.groupby(["id_carrera", "categoria_materia"])["id_materia"].nunique().to_dict()
    )

    snapshot_records = []
    grouped_students = df.groupby("alumno_carrera_hash")

    for alumno_hash, student_history in grouped_students:
        target_status = student_history["resultado_final"].iloc[0]
        censurado_status = student_history["es_censurado"].iloc[0]
        cambio_carrera_status = student_history["cambio_carrera"].iloc[0]
        carrera_abandonada_status = student_history["carrera_abandonada"].iloc[0]
        pf_previo_a_cambio = student_history["pf_previo_a_cambio"].iloc[0]
        pa_previo_a_cambio = student_history["pa_previo_a_cambio"].iloc[0]
        id_carrera_status = student_history["id_carrera"].iloc[0]
        sorted_history = student_history.sort_values(by="periodo")
        cohorte_ingreso_status = sorted_history["año_academico"].iloc[0]

        for active_semester in sorted(sorted_history["semestre"].unique()):
            historial_previo = sorted_history[sorted_history["semestre"] < active_semester]
            materias_actuales_base = sorted_history[
                sorted_history["semestre"] == active_semester
            ]

            for landmark_parcial, columnas_pf, columnas_pa in _LANDMARKS_PARCIAL:
                # Vista PARCIAL del semestre en curso: pf/pa son un proxy del
                # desempeño hasta este corte, no la calificacion final real.
                materias_actuales = materias_actuales_base.copy()
                materias_actuales["pf"] = materias_actuales[columnas_pf].mean(axis=1)
                materias_actuales["pa"] = materias_actuales[columnas_pa].mean(axis=1)

                cumulative_window = pd.concat(
                    [historial_previo, materias_actuales], ignore_index=True
                )

                snapshot = {
                    "alumno_carrera_hash": alumno_hash,
                    "semestre_actual": active_semester,
                    "landmark_parcial": landmark_parcial,
                    "promedio_calificacion_final": cumulative_window["pf"].mean(),
                    "promedio_asistencia_final": cumulative_window["pa"].mean(),
                    "materias_cursadas_totales": cumulative_window.shape[0],
                    "materias_reprobadas_totales": (cumulative_window["pf"] < 6.0).sum(),
                    "periodos_verano_cursados": (
                        cumulative_window["tipo_periodo"] == "V"
                    ).nunique(),
                    "std_calificacion_final": cumulative_window["pf"].std(ddof=0),
                    "resultado_final": target_status,
                    "es_censurado": censurado_status,
                    "cambio_carrera": cambio_carrera_status,
                    "carrera_abandonada": carrera_abandonada_status,
                    "pf_previo_a_cambio": pf_previo_a_cambio,
                    "pa_previo_a_cambio": pa_previo_a_cambio,
                    "id_carrera": id_carrera_status,
                    "cohorte_ingreso": cohorte_ingreso_status,
                    # Target de corto plazo: desenlace REAL (conocido solo en
                    # retrospectiva), nunca el proxy parcial usado arriba.
                    "materias_reprobadas_este_semestre": (
                        ~materias_actuales_base["aprobo_materia"]
                    ).sum(),
                    "aprobo_semestre": bool(
                        materias_actuales_base["aprobo_materia"].all()
                    ),
                }

                for cat in categorias_conocimiento:
                    safe_cat_name = cat.replace(" y ", "_").replace(" ", "_").lower()

                    cat_data = cumulative_window[
                        cumulative_window["categoria_materia"] == cat
                    ]

                    total_categoria_carrera = materias_por_categoria_carrera.get(
                        (id_carrera_status, cat), 0
                    )
                    materias_cursadas_categoria = cat_data["id_materia"].nunique()
                    if total_categoria_carrera > 0:
                        cobertura = min(
                            materias_cursadas_categoria / total_categoria_carrera, 1.0
                        )
                    else:
                        cobertura = 0.0
                    snapshot[f"cobertura_{safe_cat_name}"] = cobertura

                    if cat_data.empty:
                        snapshot[f"promedio_pf_{safe_cat_name}"] = 0.0
                        snapshot[f"materias_reprobadas_{safe_cat_name}"] = 0
                    else:
                        snapshot[f"promedio_pf_{safe_cat_name}"] = cat_data["pf"].mean()
                        snapshot[f"materias_reprobadas_{safe_cat_name}"] = (
                            cat_data["pf"] < 6.0
                        ).sum()

                snapshot_records.append(snapshot)

    snapshot_df = pd.DataFrame(snapshot_records).fillna(0)

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


def build_progress_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms transactional grade rows into cumulative historical snapshots per semester.
    Imputes missing academic values with 0 for predictive model alignment.
    Also categorizes attendance into 5 risk blocks.
    """
    print("Building progress snapshots for model processing...")

    # Fill remaining NaNs to ready data for the mathematical aggregations
    fill_cols = [
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
    df[fill_cols] = df[fill_cols].fillna(0)

    snapshot_records = []
    grouped_students = df.groupby("alumno_carrera_hash")

    for _, student_history in grouped_students:
        target_status = student_history["resultado_final"].iloc[0]
        censurado_status = student_history["es_censurado"].iloc[0]
        cambio_carrera_status = student_history["cambio_carrera"].iloc[0]
        carrera_abandonada_status = student_history["carrera_abandonada"].iloc[0]
        pf_previo_a_cambio = student_history["pf_previo_a_cambio"].iloc[0]
        pa_previo_a_cambio = student_history["pa_previo_a_cambio"].iloc[0]
        id_carrera_status = student_history["id_carrera"].iloc[0]
        sorted_history = student_history.sort_values(by="periodo")
        cohorte_ingreso_status = sorted_history["año_academico"].iloc[0]

        for active_semester in sorted(sorted_history["semestre"].unique()):
            # Isolate cumulative data up to the current evaluated snapshot semester
            cumulative_window = sorted_history[
                sorted_history["semestre"] <= active_semester
            ]

            snapshot = {
                "semestre_actual": active_semester,
                "promedio_calificacion_final": cumulative_window["pf"].mean(),
                "promedio_asistencia_final": cumulative_window["pa"].mean(),
                "materias_cursadas": cumulative_window.shape[0],
                "materias_reprobadas": (cumulative_window["pf"] < 6.0).sum(),
                "periodos_verano_cursados": (
                    cumulative_window["tipo_periodo"] == "V"
                ).nunique(),
                "std_calificacion_final": cumulative_window["pf"].std(ddof=0),
                "resultado_final": target_status,
                "es_censurado": censurado_status,
                "cambio_carrera": cambio_carrera_status,
                "carrera_abandonada": carrera_abandonada_status,
                "pf_previo_a_cambio": pf_previo_a_cambio,
                "pa_previo_a_cambio": pa_previo_a_cambio,
                "id_carrera": id_carrera_status,
                "cohorte_ingreso": cohorte_ingreso_status,
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
