# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'CancerStFormer'
copyright = 'MIT'
author = 'Ivy Strope'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex'
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    '.ipynb':'myst-nb'
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Bibliography settings
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
