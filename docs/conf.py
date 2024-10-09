# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    # 'sphinx_multiversion',
    # 'sphinx_sitemap',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README*',
    'api'
]
source_suffix = {'.md': 'markdown', '.rst': 'restructuredtext'}

# -- Options for HTML output -------------------------------------------------

html_title = 'FluffyClone documentation'
html_baseurl = 'https://ulysseherbach.github.io/fluffyclone/'
html_static_path = ['_static']
html_css_files = ['custom.css']

# html_sidebars = {
#     '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
# }

# html_theme = 'sphinx_book_theme'

html_theme = 'alabaster'
html_theme_options = {
    'fixed_sidebar': True,
    # 'logo': 'logo.png',
    # 'github_user': 'ulysseherbach',
    # 'github_repo': 'fluffyclone',
}

# html_theme = 'pydata_sphinx_theme'
# html_theme_options = {
#     'icon_links': [
#         {
#             'name': 'GitHub',
#             'url': 'https://github.com/ulysseherbach/fluffyclone',
#             'icon': 'fa-brands fa-github',
#             'type': 'fontawesome',
#         }
#    ],
#    'pygments_light_style': 'default',
#    'pygments_dark_style': 'material',
# }

# -- Options for sphinx.ext.autodoc ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method
autodoc_typehints = 'description'

# Don't show class signature with the class' name
autodoc_class_signature = 'separated'

# -- Options for sphinx.ext.napoleon -----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for sphinx.ext.autosummary --------------------------------------

autosummary_generate = True
autosummary_ignore_module_all = False
# autosummary_imported_members = True

# -- Options for myst_parser -------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    # 'amsmath',
    # 'dollarmath',
    # 'strikethrough',
    # 'tasklist',
]

# -- Options for sphinx-copybutton ------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest

copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = r'[$!]\s*'
copybutton_prompt_is_regexp = True

# -- Project information -----------------------------------------------------

# project = 'FluffyClone'
copyright = '2024, Ulysse Herbach'
author = 'Ulysse Herbach'

# -- Dynamic settings --------------------------------------------------------

# Get current version
try:
    version = get_version('fluffyclone')
except ImportError:
    raise RuntimeError('fluffyclone must be installed for autodoc to work.')

release = version
project = f'FluffyClone {version}'
# html_title = f'FluffyClone {version} documentation'
