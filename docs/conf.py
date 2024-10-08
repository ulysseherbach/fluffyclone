# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------

project = 'FluffyClone'
copyright = '2024, Ulysse Herbach'
author = 'Ulysse Herbach'

# -- General configuration ---------------------------------------------------

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']


# -- Dynamic information -----------------------------------------------------

# Get current version
try:
    version = get_version('fluffyclone')
except ImportError:
    raise RuntimeError('fluffyclone must be installed for autodoc to work.')

release = version
