import os
import re
import unicodedata
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "mappings")

os.makedirs(DATA_DIR, exist_ok=True)


def normalize_text(text: str) -> str:
    """Normaliza el texto: minúsculas, sin acentos y sin caracteres especiales."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def classify_subject(materia: str) -> str:
    """Clasifica la materia en un área de conocimiento usando Regex."""
    m = materia

    if re.search(
        r"administracion|gestion|direccion|liderazgo|calidad|compras|"
        r"inventarios|recursos humanos|mercadotecnia|mercados|ventas|"
        r"financiera|costos|presupuestos|empresarial",
        m,
    ):
        return "Administración y Gestión"

    if re.search(
        r"calculo|algebra|estadistica|probabilidad|numerico|matematic|"
        r"vectorial|convexo|funcional|fourier|analisis",
        m,
    ):
        return "Matemáticas y Estadística"

    if re.search(
        r"programacion|algoritmos|computacion|base de datos|compilador|"
        r"software|arquitectura de computadoras|sistemas operativos|redes|"
        r"inteligencia artificial|informatica|computadora",
        m,
    ):
        return "Programación y Computación"

    if re.search(
        r"electronica|circuitos|control|potencia|microcontrolador|senal|"
        r"instrumentacion|telecomunicacion",
        m,
    ):
        return "Electrónica y Control"

    if re.search(
        r"vibraciones|mecanica|cinematica|dinamica|termofluid|estructuras|"
        r"mantenimiento|manufactura",
        m,
    ):
        return "Ingeniería Mecánica"

    if re.search(
        r"materiales|procesos|balance de materia|energia|quimica de alimentos|"
        r"biotecnologia|analisis de alimentos|inocuidad",
        m,
    ):
        return "Materiales y Procesos"

    if re.search(
        r"fisica|quimica|termodinamica|optica|electromagnetismo|"
        r"celdas de combustible",
        m,
    ):
        return "Física y Química"

    if re.search(
        r"etica|comunicacion|sociedad|humanidades|desarrollo humano|"
        r"psicologia|responsabilidad social",
        m,
    ):
        return "Humanidades y Comunicación"

    if re.search(r"finanzas|economia|mercado|costos|presupuesto|contabilidad", m):
        return "Economía y Finanzas"

    if re.search(r"automatizacion|robotica|automa|plc|sistemas embebidos", m):
        return "Automatización y Robótica"

    if re.search(r"dibujo|cimentacion|diseno|arquitectura|maquetas|planos", m):
        return "Diseño y Arquitectura"

    return "Otra"


def run_mapping():
    input_file = os.path.join(
        ROOT_DIR, "data", "original", "materias_sin_clasificacion.csv"
    )
    output_file = os.path.join(DATA_DIR, "materias_clasificadas.csv")

    try:
        df = pd.read_csv(input_file)
        print(f"Cargando {len(df)} materias desde la base SQL...")

        df["materia_normalizada"] = df["materia"].apply(normalize_text)
        df["categoria_materia"] = df["materia_normalizada"].apply(classify_subject)

        df_unicas = (
            df[["id_materia", "categoria_materia"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df_unicas.to_csv(output_file, index=False, encoding="utf-8")

        print("=" * 50)
        print(" Estadísticas de Clasificación (Materias Únicas) ")
        print("=" * 50)
        print(df_unicas["categoria_materia"].value_counts())
        print("-" * 50)
        print(f"Total de materias únicas clasificadas: {len(df_unicas)}")
        print(
            f"Archivo guardado exitosamente en: data/mappings/materias_clasificadas.csv"
        )

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {input_file}")


if __name__ == "__main__":
    run_mapping()
