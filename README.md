# Academic Performance & Retention Analysis Pipeline

Este proyecto contiene el ecosistema de procesamiento de datos, ingeniería de características (Feature Engineering) y análisis exploratorio automatizado (EDA) enfocado en la predicción de la deserción escolar con base en factores institucionales. 

El sistema está diseñado bajo una arquitectura modular de software en inglés para la lógica e infraestructura, manteniendo consistencia semántica en español en las capas de visualización y estructuración de datos para alineación directa con el documento de tesis.

---

## Estructura del Proyecto

```bash
academic_performance_analysis/
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Carga de variables de entorno (.env) y constantes académicas
│
├── database/
│   └── docker-compose.yml   # Infraestructura local de PostgreSQL en contenedor Docker
│
├── src/                     # Código fuente modularizado ("Tools")
│   ├── __init__.py
│   ├── db_connector.py      # Conexión optimizada a la BD mediante SQLAlchemy y Psycopg2
│   ├── data_cleaner.py      # Saneamiento de datos crudos (conversión de flags -10 a NaNs)
│   ├── feature_engineer.py  # Anonimización (hashes), lógica de egreso e instantáneas temporales
│   └── eda_visualizer.py    # Generación de gráficos e histogramas limpios por carrera
│
├── notebooks/               # Entorno experimentalizado
│   └── eda_exploration.ipynb # Notebook orquestador optimizado para análisis visual
│
├── scripts/                 # Automatizaciones fuera del entorno de cuadernos
│   └── run_profiling.py     # Script optimizado para la ejecución aislada de fg-data-profiling
│
├── .env                     # Archivo local de credenciales de base de datos (ignorado en Git)
├── .gitignore               # Exclusiones de Git (entornos virtuales, backups, .env, etc.)
└── requirements.txt         # Dependencias oficiales del proyecto (Python 3.13)
```

---

## Requisitos e Instalación

### 1. Preparar el entorno virtual (venv)
El proyecto utiliza Python versión 3.13 de manera nativa. Sigue estos comandos en tu terminal para inicializar y activar tu entorno virtual aislado:

# En Windows (PowerShell / CMD)
```bash
python -m venv venv
.\venv\Scripts\activate
```

# En Mac / Linux / Git Bash
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias desde el archivo requirements.txt
Una vez activado el entorno virtual, puedes instalar de golpe todas las dependencias del proyecto (incluyendo los motores de Machine Learning avanzados como XGBoost, LightGBM, y el motor de perfilado fg-data-profiling):
```bash
pip install -r requirements.txt
```
---

## Configuración de Infraestructura y Entorno

### 1. Base de Datos Local (Docker)
El proyecto incluye un entorno aislado para hospedar tu base de datos mediante contenedores. Para levantar la instancia de PostgreSQL, navega a la carpeta correspondiente y arranca el daemon:
```bash
cd database
docker-compose up -d
```

### 2. Variables de Enorno (.env)
Crea un archivo llamado .env en la raíz principal del proyecto para mapear de manera segura tus credenciales locales:
```bash
DB_HOST=localhost
DB_NAME=nes            # Nota: Mantener en minúsculas por sensibilidad de PostgreSQL
DB_USER=postgres
DB_PASSWORD=tu_contraseña_secreta
DB_PORT=5432
```
---

## Modos de Ejecución

### Opción A: Perfilado Masivo Automatizado con fg-data-profiling
Para analizar la distribución completa de los 133,882 registros históricos sin saturar la memoria gráfica de tu navegador ni congelar Jupyter, ejecuta el script automatizado en consola. Este utilizará fg-data-profiling para generar un reporte estático interactivo en la raíz:

```bash
python scripts/run_profiling.py
```

Si tu hardware cuenta con limitaciones de memoria RAM, puedes abrir el script y activar la bandera de muestreo (use_sample=True) para procesar una sección controlada de 25,000 registros.

### Opción B: Experimentación Visual (Jupyter Notebook)
Para continuar refinando análisis bivariables o ajustar modelos predictivos avanzados, abre el orquestador:

```bash
jupyter notebook notebooks/eda_exploration.ipynb
```
---

## Estándar de Código y Formato

Para garantizar la mantenibilidad y calidad de software requerida a nivel de ingeniería, el repositorio utiliza herramientas estrictas de análisis estático coordinadas bajo una regla extendida de 100 caracteres por línea:

* Black Formatter: Encargado del autoformateo estricto del código al guardar.
* Flake8: Validador de estilo PEP 8 (configurado para omitir falsos positivos como el espacio E203 en slices complejos).

La configuración ideal compartida se encuentra automatizada dentro de la carpeta .vscode/settings.json.