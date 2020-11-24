"""
**Obiwan** is a Monte Carlo method for adding fake galaxies to Legacy Survey images and re-processing the modified images with the legacypipe.

Contains:
- runbrick.py : **Obiwan** main executable, equivalent of ``legacypipe.runbrick``.
- kenobi.py : Classes to extend legacypipe.
- catalog.py : Convenient classes to handle catalogs and bricks.
- utils.py : Convenient functions to handle Obiwan inputs/outputs.
"""

from .version import __version__

__all__ = ['LegacySurveySim','get_randoms_id','find_file','find_survey_file','find_output_file']
__all__ += ['BaseCatalog','SimCatalog','BrickCatalog','utils','setup_logging','scripts','batch']

from .kenobi import LegacySurveySim,get_randoms_id,find_file,find_survey_file,find_output_file
from .catalog import BaseCatalog,SimCatalog,BrickCatalog
from .utils import setup_logging
