"""Sphinx extension: an in-browser Kaval.AI playground for the HTML docs.

It does three things:

* registers the playground CSS/JS (``_static/pyodide/playground.{css,js}``),
* stages the most recently built ``dist/kavalai-*.whl`` into
  ``_static/pyodide/`` so Pyodide's ``micropip`` can install it client-side, and
* emits ``_static/pyodide/playground-config.js`` telling the front-end which
  wheel filename and Pyodide build to use.

The wheel is copied (not committed) on every build; if ``dist/`` has no wheel,
any previously staged one is reused, and failing that the playground falls back
to plain-Python (Pyodide stdlib) mode. Build a wheel with ``uv build --wheel``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

DEFAULT_PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/pyodide.js"


def _newest_wheel(directory: Path) -> Path | None:
    """Return the most recently modified ``kavalai-*.whl`` in *directory*."""
    wheels = sorted(directory.glob("kavalai-*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def _stage_wheel(app: Sphinx) -> None:
    """Copy the freshest built wheel into ``_static/pyodide`` and write config."""
    static_dir = Path(app.srcdir) / "_static" / "pyodide"
    static_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(app.srcdir).parent
    built = _newest_wheel(repo_root / "dist") if (repo_root / "dist").is_dir() else None

    if built is not None:
        # Replace any stale staged wheels with the freshly built one.
        for stale in static_dir.glob("kavalai-*.whl"):
            if stale.name != built.name:
                stale.unlink()
        target = static_dir / built.name
        shutil.copy2(built, target)
        wheel_name: str | None = built.name
        logger.info("[kaval-playground] staged wheel %s", built.name)
    else:
        # No fresh build — reuse a previously staged wheel if one exists.
        existing = _newest_wheel(static_dir)
        wheel_name = existing.name if existing else None
        if wheel_name:
            logger.info("[kaval-playground] reusing staged wheel %s", wheel_name)
        else:
            logger.warning(
                "[kaval-playground] no kavalai wheel in dist/ or _static/pyodide; "
                "the playground will run plain Python only. Build one with "
                "`uv build --wheel`."
            )

    config = {
        "wheelName": wheel_name,
        "pyodideUrl": app.config.kaval_pyodide_url,
    }
    config_js = static_dir / "playground-config.js"
    config_js.write_text(
        "window.KAVAL_PLAYGROUND_CONFIG = " + json.dumps(config, indent=2) + ";\n",
        encoding="utf-8",
    )


def setup(app: Sphinx) -> dict:
    app.add_config_value("kaval_pyodide_url", DEFAULT_PYODIDE_URL, "html")

    # Generate the staged wheel + config before static files are copied.
    app.connect("builder-inited", _stage_wheel)

    # playground-config.js must load before playground.js (registration order).
    app.add_css_file("pyodide/playground.css")
    app.add_js_file("pyodide/playground-config.js")
    app.add_js_file("pyodide/playground.js")

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
