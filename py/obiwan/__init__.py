"""
**Obiwan** is a Monte Carlo method for adding fake galaxies to Legacy Survey images,
and re-processing the modified images with the legacypipe.

Contains:
- runbrick.py : **Obiwan** main executable, equivalent of ``legacypipe.runbrick``.
- kenobi.py : Classes to extend legacypipe.
- catalog.py : Convenient classes to handle catalogs and bricks.
- utils.py : Convenient functions to handle Obiwan inputs/outputs.
"""

from .version import __version__

__all__ = ['LegacySurveySim','BaseCatalog','SimCatalog','BrickCatalog','scripts','utils','setup_logging']

from .kenobi import LegacySurveySim
from .catalog import BaseCatalog,SimCatalog,BrickCatalog
from .utils import setup_logging
