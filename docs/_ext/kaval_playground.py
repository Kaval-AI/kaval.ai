"""Sphinx extension: an in-browser Kaval.AI playground for the HTML docs.

The playground itself lives in ``webwidget/`` (the single source of truth, also
used by the standalone ``webwidget/python-playground.html`` and embeddable on any
site). This extension wires it into the Sphinx build so the docs don't duplicate
any of that code. On every build it:

* copies ``webwidget/kaval-playground.{css,js}`` and ``webwidget/kaval-chat.{css,js}``
  into ``_static/pyodide/`` and registers them (the chat widget powers the
  interactive workflow demo on the landing page),
* stages the most recently built ``dist/kavalai-*.whl`` into ``_static/pyodide/``
  so Pyodide's ``micropip`` can install it client-side, and
* emits ``_static/pyodide/playground-config.js`` telling the widget which wheel
  filename and Pyodide build to use.

Copied/staged files are not committed (see ``.gitignore``). If ``dist/`` has no
wheel, any previously staged one is reused, and failing that the playground
falls back to plain-Python (Pyodide stdlib) mode. Build a wheel with
``uv build --wheel``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

DEFAULT_PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/pyodide.js"

# Widget assets copied from webwidget/ into the docs' _static/pyodide on build.
WIDGET_FILES = (
    "kaval-playground.css",
    "kaval-playground.js",
    "kaval-chat.css",
    "kaval-chat.js",
)


def _newest_wheel(directory: Path) -> Path | None:
    """Return the most recently modified ``kavalai-*.whl`` in *directory*."""
    wheels = sorted(directory.glob("kavalai-*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def _copy_widget(app: Sphinx, static_dir: Path) -> None:
    """Copy the shared widget from ``webwidget/`` into ``_static/pyodide``."""
    webwidget = Path(app.srcdir).parent / "webwidget"
    for name in WIDGET_FILES:
        src = webwidget / name
        if src.is_file():
            shutil.copy2(src, static_dir / name)
            logger.info("[kaval-playground] copied widget asset %s", name)
        else:
            logger.warning(
                "[kaval-playground] missing widget asset %s — the playground "
                "will not work. Expected it under webwidget/.",
                src,
            )


def _stage_assets(app: Sphinx) -> None:
    """Copy the widget + freshest wheel into ``_static/pyodide`` and write config."""
    static_dir = Path(app.srcdir) / "_static" / "pyodide"
    static_dir.mkdir(parents=True, exist_ok=True)

    _copy_widget(app, static_dir)

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

    # Copy the widget + stage the wheel + config before static files are copied.
    app.connect("builder-inited", _stage_assets)

    # playground-config.js must load before kaval-playground.js (it reads the
    # window.KAVAL_PLAYGROUND_CONFIG it emits).
    app.add_css_file("pyodide/kaval-playground.css")
    app.add_css_file("pyodide/kaval-chat.css")
    app.add_js_file("pyodide/playground-config.js")
    app.add_js_file("pyodide/kaval-playground.js")
    app.add_js_file("pyodide/kaval-chat.js")

    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
