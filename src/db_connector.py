import pandas as pd
from sqlalchemy import create_engine
from config.settings import DATABASE_CONFIG


def get_db_engine():
    """Creates and returns a SQLAlchemy connection engine."""
    connection_url = (
        f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['dbname']}"
    )
    return create_engine(connection_url)


def fetch_raw_grade_history() -> pd.DataFrame:
    """
    Queries the database and returns the raw academic historical records.

    Returns:
        pd.DataFrame: raw_grade_history
    """
    print("Executing database extraction query...")
    engine = get_db_engine()

    # Using optimized SQL query
    query = """
        SELECT
            c.matricula,
            g.id_carrera,
            c.id_grupo,
            c.id_materia,
            g.semestre,
            g.id_periodo AS periodo,
            -- Calificaciones
            c.p1, c.p2, c.p3, c.o, c.pf, c.e1, c.e2, c.esp,
            -- Asistencias
            c.a1, c.a2, c.a3, c.oa, c.pa
        FROM
            nes_calificaciones AS c
        JOIN
            nes_grupos AS g ON c.id_grupo = g.id_grupo;
        """

    with engine.connect() as connection:
        raw_grade_history = pd.read_sql_query(query, connection)

    print(f"Successfully extracted {len(raw_grade_history)} raw records.")
    return raw_grade_history
