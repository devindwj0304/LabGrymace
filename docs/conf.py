# Configuration file for the Sphinx documentation builder.
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'LabGrymace'
author = 'Wenjin Dong and Bing Ye'
copyright = '2026, Bing Ye Lab, University of Michigan'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# myst_parser lets Sphinx read Markdown (.md) sources, so the pages use the same
# syntax as the README. sphinx_copybutton adds a copy button to every code block.
extensions = [
    'myst_parser',
    'sphinx_copybutton',
]

# MyST extensions. colon_fence allows directives written with ::: fences.
myst_enable_extensions = [
    'colon_fence',
]

# LABGYM_CHANGES.md and images/README.md are GitHub-only files that support the
# repository README. They are not part of the documentation site, so exclude them
# from the Sphinx build to avoid orphan-document warnings.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'LABGYM_CHANGES.md',
    'images/README.md',
]

# -- HTML output -------------------------------------------------------------
# Furo is the same theme used by the upstream LabGym documentation.
html_theme = 'furo'
html_title = 'LabGrymace'
