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
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'myst_nb'
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    '.ipynb':'myst-nb'
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto", "ftp")
myst_heading_anchors = 3
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True

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
