import logging
import os

from dotenv import load_dotenv
from rich.logging import RichHandler

load_dotenv(verbose=True)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
SQL_MIGRATIONS_PATH = os.path.join(PACKAGE_PATH, "sql_migrations")

AGENTS_PACKAGE_PATH = os.path.join(PACKAGE_PATH, "agents")
AGENTS_MIGRATIONS_PATH = os.path.join(SQL_MIGRATIONS_PATH, "app")

BACKOFFICE_PACKAGE_PATH = os.path.join(PACKAGE_PATH, "backoffice")
BACKOFFICE_MIGRATIONS_PATH = os.path.join(SQL_MIGRATIONS_PATH, "backoffice")
