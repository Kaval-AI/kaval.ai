import logging

from dotenv import load_dotenv
from rich.logging import RichHandler

load_dotenv(verbose=True)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]", handlers=[RichHandler()])

