# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# Local Sphinx extensions (docs/_ext), e.g. the in-browser playground.
sys.path.insert(0, os.path.abspath("_ext"))

# Set dummy environment variables for modules that require them during doc build
# (autodoc imports the real modules, some of which read DB config at import time).
os.environ.setdefault("KAVALAI_BO_DB_SCHEMA", "public")
os.environ.setdefault("KAVALAI_BO_DB_URI", "postgresql://user:pass@localhost:5432/db")
os.environ.setdefault("KAVALAI_DB_SCHEMA", "public")
os.environ.setdefault("KAVALAI_DB_URI", "postgresql://user:pass@localhost:5432/db")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Kaval.AI"
copyright = "2026, Kaval.AI Team"
author = "Kaval.AI Team"
# Derive the version from the installed package so the docs never drift from
# pyproject.toml. Falls back gracefully if the package isn't installed.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("kavalai")
except Exception:  # pragma: no cover - build-time best effort
    release = "1.0.0"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "myst_nb",
    "sphinx_immaterial",
    "kaval_playground",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render "Attributes:" sections inline as :ivar: fields instead of separate
# object descriptions — avoids duplicate-object warnings for Pydantic models.
napoleon_use_ivar = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# MyST / myst-nb settings. Notebooks are rendered from their stored outputs;
# they are NOT re-executed at build time (which would require API keys).
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3
suppress_warnings = ["mystnb.unknown_mime_type"]

# Cross-project references.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_includes"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://kaval.ai/",
    "repo_url": "https://github.com/kavalai/kaval.ai/",
    "repo_name": "Kaval.AI",
}

html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
