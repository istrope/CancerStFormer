# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'CancerStFormer'
copyright = 'MIT'
author = 'Ivy Strope'

release = '0.1'
version = '0.1.0'

repository_url='https://github.com/istrope/CancerStFormer'


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.coverage',
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
nb_execution_mode = "off"

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
