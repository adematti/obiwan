# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Obiwan'
copyright = '2020, Hui Kong, Kaylan Burleigh, John Moustakas'
author = 'Hui Kong, Kaylan Burleigh, John Moustakas'

# The full version, including alpha/beta/rc tags
release = 'DR9'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

import sphinx_glpi_theme
html_theme = "glpi"
html_theme_path = sphinx_glpi_theme.get_html_themes_path()
"""
import sphinx_bootstrap_theme
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {
    'navbar_title': 'Obiwan',
    'navbar_site_name': 'Pages',
    'navbar_pagenav_name': 'This Page',
    'navbar_fixed_top': 'false',
    'source_link_position': 'none',
    #'bootswatch_theme': 'cosmo',
    #'bootswatch_theme': 'lumen',
    #'bootswatch_theme': 'sandstone',
    'bootswatch_theme': 'spacelab',
    'navbar_links': [
    ("API", "api"),
    ],
}
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
