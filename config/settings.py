import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Setup Base Directory Path
# Resolves the absolute path to the root folder of your project
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Load Environment Variables from .env file
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# 3. Database Credentials Dictionary
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", "nes"),
    "user": os.getenv("DB_USER", "mi_usuario"),
    "password": os.getenv("DB_PASSWORD", "mi_password_seguro"),
    "port": os.getenv("DB_PORT", "5432"),
}

# 4. Academic Timeline Constants
# Centralized to avoid repeating them across multiple scripts
PERIOD_ORDER = [
    35,
    36,
    40,
    39,
    42,
    43,
    41,
    46,
    47,
    44,
    49,
    50,
    48,
    53,
    54,
    51,
    52,
    56,
    55,
    58,
    59,
    57,
    61,
]

PERIOD_NAMES = [
    "2017-2018A",
    "2017-2018B",
    "2017-2018V",
    "2018-2019A",
    "2018-2019B",
    "2018-2019V",
    "2019-2020A",
    "2019-2020B",
    "2019-2020V",
    "2020-2021A",
    "2020-2021B",
    "2020-2021V",
    "2021-2022A",
    "2021-2022B",
    "2021-2022V",
    "2022-2023A",
    "2022-2023B",
    "2022-2023V",
    "2023-2024A",
    "2023-2024B",
    "2023-2024V",
    "2024-2025A",
    "2024-2025B",
]
