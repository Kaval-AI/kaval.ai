"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
