import unicodedata
from typing import Any


def clean_text(text: Any) -> Any:
    """Normalizes string to NFC and removes null characters and other problematic non-printable chars.

    If input is not a string, returns it as is.
    """
    if not isinstance(text, str):
        return text

    # Normalize to NFC
    text = unicodedata.normalize("NFC", text)

    # Remove null characters which PostgreSQL does not support in TEXT/VARCHAR/JSONB
    text = text.replace("\u0000", "")

    # Remove other problematic non-printable characters, but keep common whitespace
    # (newline, carriage return, tab)
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t"
    )

