"""Convenient classes to handle catalogs and bricks."""

import os
import logging
import copy
import numpy as np
from astrometry.util import fits
from legacypipe.survey import LegacySurveyData,wcs_for_brick
from . import utils

logger = logging.getLogger('obiwan.catalog')

class BaseCatalog(fits.tabledata):

    """Extend ``astrometry.util.fits.tabledata``, with convenient methods."""

    def __init__(self, *args, **kwargs):
        """
        Call ``astrometry.util.fits.fits_table()``.

        For an empty catalog, one provide size or length, to define ``len(self)``,
        used in methods to build ``numpy`` arrays (e.g. ``self.zeros()``).
        Otherwise, ``len(self)`` is set when the first column is added to ``self``.
        """
        length = kwargs.pop('length',kwargs.pop('size',None))
        table = fits.fits_table(*args,**kwargs)
        self.__dict__.update(table.__dict__)
        if length is not None: self._length = length

    def __getitem__(self, item):
        """Redefine ``__getitem__`` to avoid calling ``__init__()`` each time."""
        toret = self.copy()
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                toret.set(name, copy.deepcopy(val))
                continue
            try:
                if name in self and len(self) < 2:
                    val = np.atleast_1d(val)
                col = fits.cut_array(val, item, name)
            except:
                raise ValueError('Error in cut_array() via __getitem__, name %s' % name)
            toret.set(name, col)
            if np.isscalar(item):
                toret._length = 1
            else:
                toret._length = len(getattr(toret, name))
        return toret

    @classmethod
    def from_dict(self, d):
        """Construct catalog from dictionary of field, array."""
        self = cls()
        for key,val in d.items():
            self.set(key,val)
        return self

    @property
    def size(self):
        """Return ``size = len(self)``."""
        return len(self)

    @property
    def fields(self):
        """Return ``fields = self.get_columns(internal=False)``."""
        return self.get_columns(internal=False)

    def __contains__(self, name):
        """Return whether ``self`` contains column ``name``."""
        return (name in self.fields)

    def zeros(self, dtype=np.float64):
        """Return array of size ``len(self)`` filled with zeros."""
        return np.zeros(len(self),dtype=dtype)

    def ones(self, dtype=np.float64):
        """Return array of size ``len(self)`` filled with zeros."""
        return np.ones(len(self),dtype=dtype)

    def falses(self):
        """Return array of size ``len(self)`` filled with ``False``."""
        return self.zeros(dtype=np.bool_)

    def trues(self):
        """Return array of size ``len(self)`` filled with ``True``."""
        return self.ones(dtype=np.bool_)

    def nans(self):
        """Return array of size ``len(self)`` filled with ``np.nan``."""
        return self.ones()*np.nan

    def full(self,*args,**kwargs):
        """Call ``np.full`` with ``shape = len(self)``."""
        return np.full(self.size,*args,**kwargs)

    def index(self):
        """Return zero-starting index."""
        return np.arange(self.size)

    def delete_columns(self, *fields):
        """Delete columns."""
        for field in fields:
            self.delete_column(field)

    def keep_columns(self, *fields):
        """Keep only columns in argument."""
        for field in self.fields:
            if field not in fields:
                self.delete_column(field)

    def fill(self, other, index_self=None, index_other=None, fields_other=None, fill_nan=True):
        """
        Fill (or create) columns of ``self`` with those of ``other`` catalog.

        If a column of name ``field`` already exists in ``self`` (with a shape compatible with that of ``other``),
        the values with indices ``index_self`` are replaced by those of ``other`` with indices ``index_other``.
        Else, a new column is added to ``self``, with default value given by ``np.zeros()`` (or ``np.nan`` if ``fill_nan``).
        Again, the values with indices ``index_self`` are replaced by those of ``other`` with indices ``index_other``.

        Parameters
        ----------
        other : BaseCatalog
            Catalog to be merged to ``self``.

        index_self : slice, ndarray, string, default=None
            Indices (or bool mask) of ``self`` to fill with ``other``.
            If ``None``, all indices of ``self`` are considered.
            If `before`, ``other[index_other]`` rows are added at the beginning of ``self``.
            If `after`, ``other[index_other]`` rows are added at the end of ``self``.

        index_other : slice, ndarray, default=None
            Indices (or bool mask) of ``other`` to fill ``self``.
            If ``None``, all indices of ``other`` are considered.

        fields_other : list, default=None
            Fields of ``other`` to be added to ``self``.
            If ``None``, all fields of other are considered.

        fill_nan : bool, default=True
            If ``True``, default value for added column is ``np.nan`` instead of ``np.zeros()``.
        """
        if index_self is None:
            index_self = self.trues()
        if index_other is None:
            index_other = other.trues()
        if fields_other is None:
            fields_other = other.fields

        other_size = other.get(fields_other[0])[index_other].size
        def isfloating(dtype):
            return isinstance(np.zeros(1,dtype=dtype)[0],np.floating)

        if isinstance(index_self,str):
            index_self_ = index_self
            totalsize = self.size + other_size
            if index_self_ == 'before':
                index_self = slice(0,other_size)
            elif index_self_ == 'after':
                index_self = slice(self.size,totalsize)
            else:
                raise ValueError('If string, index_self should be either "before" or "after"')
            for field in self.fields:
                col_self = self.get(field)
                col_new = np.zeros(shape=(other_size,)+col_self.shape[1:],dtype=col_self.dtype)
                if fill_nan and isfloating(col_self.dtype):
                    col_new[...] = np.nan
                if index_self_  == 'before':
                    col_new = np.concatenate([col_new,col_self])
                elif index_self_  == 'after':
                    col_new = np.concatenate([col_self,col_new])
                self.set(field,col_new)
            self._length = totalsize

        for field in fields_other:
            col_other = other.get(field)
            if field in self.fields:
                col_self = self.get(field)
                if col_self.shape[1:] != col_other.shape[1:]: # other column overrides
                    self.delete_column(field)
            if field not in self.fields:
                col_self = np.zeros(shape=(self.size,)+col_other.shape[1:],dtype=col_other.dtype)
                if fill_nan and isfloating(col_self.dtype):
                    col_self[...] = np.nan
            col_self[index_self] = col_other[index_other]

            self.set(field,col_self)

    def copy(self):
        """Return a (deep) copy of ``self``."""
        # ugly hack, because of tabledata.copy() definition...
        toret = object.__new__(self.__class__)
        toret.__dict__.update(super(BaseCatalog,self).copy().__dict__)
        return toret

    def __eq__(self, other):
        """Return whether ``self == other`` (same type, length and colums)."""
        if not isinstance(other,self.__class__):
            return False
        if not set(self.fields) == set(other.fields):
            return False
        if not self.size == other.size:
            return False
        if self.size == 0:
            return True
        for field in self.fields:
            if isinstance(self.get(field).flat[0],np.floating):
                mask = ~np.isnan(self.get(field))
                if not np.all(self.get(field)[mask] == other.get(field)[mask]):
                    return False
            else:
                if not np.all(self.get(field) == other.get(field)):
                    return False
        return True

    def __radd__(self, other):
        """Right add: ``self + other``."""
        if other == 0:
            return self.copy()
        return self.__add__(other)

    def __ladd__(self, other):
        """Left add: ``other + self``."""
        if other == 0:
            return self.copy()
        return self.__add__(other)

    def __add__(self, other):
        """
        Append ``self`` and ``other`` catalogs and return a new instance ``toret``.

        ``len(toret) == len(self) + len(other)``
        """
        toret = self.copy()
        if other == 0:
            return toret
        if not other.fields: #empty catalog
            return toret
        toret.append(other)
        return toret

    def writeto(self, fn, *args, **kwargs):
        """Save to fn."""
        logger.info('Saving catalog to %s.' % fn)
        utils.mkdir(os.path.dirname(fn))
        super(BaseCatalog,self).writeto(fn,*args,**kwargs)

class SimCatalog(BaseCatalog):

    """Extend ``BaseCatalog`` with convenient methods for **Obiwan** randoms."""

    def fill_obiwan(self, survey=None):
        """
        Fill ``self`` with columns required for **Obiwan** (if not already in ``self``):
            ``id`` : defaults to ``np.arange(len(self))``
            ``brickname``, (brick) ``bx``, ``by``: based on ``survey`` and ``self.ra``, ``self.dec``.

        Columns mandatory for **Obiwan** are:
            ``ra``, ``dec`` : coordinates (degree)
            ``flux_g``, ``flux_r``, ``flux_z`` : including galactic extinction (nanomaggies)
            ``sersic`` : Sersic index
            ``shape_r`` : half light radius (arcsecond)
            ``shape_e1``, ``shape_e2`` : ellipticities.

        Parameters
        ----------
        survey : LegacySurveyData, string, default=None
            ``survey_dir`` or survey. Used to determine brick-related quantities (``brickname``, ``bx``, ``by``).
            If ``None``, brick-related quantities are inferred from ``desiutil.brick.Bricks`` (see ``BrickCatalog``).

        References
        ----------
        https://en.wikipedia.org/wiki/Sersic_profile
        https://www.legacysurvey.org/dr8/catalogs/
        """
        bricks = None
        if 'brickname' not in self:
            bricks = BrickCatalog(survey=survey)
            self.brickname = bricks.get_by_radec(self.ra,self.dec).brickname

        if 'id' not in self:
            self.id = self.index()

        if 'bx' not in self:
            if bricks is None: bricks = BrickCatalog(survey=survey)
            self.bx,self.by = bricks.get_xy_from_radec(self.ra,self.dec,brickname=self.brickname)

    def mask_collisions(self, radius_in_degree=5./3600.):
        """
        Return mask of collided objects.

        Calls ``utils.mask_collisions()``.

        Parameters
        ----------
        radius_in_degree : float, default=5./3600.
            Collision radius (degree).

        Returns
        -------
        mask : bool ndarray
            Mask of collided objects.
        """
        return utils.mask_collisions(self.ra,self.dec,radius_in_degree=radius_in_degree)

    def match_radec(self, other, radius_in_degree=5./3600., **kwargs):
        """
        Match ``self`` and ``other`` SimCatalog ``ra``, ``dec``.

        Calls ``utils.match_radec()``.

        Parameters
        ----------
        other : SimCatalog
            Catalog to be matched against ``self``.

        radius_in_degree : float, default=5./3600.
            Collision radius (degree).

        kwargs : dict, default={}
            Arguments for ``utils.match_radec()``.

        Returns
        -------
        index_self : ndarray
            Indices of matching points in ``self``.

        index_other : ndarray
            Indices of matching points in ``other``.
        """
        return utils.match_radec(self.ra,self.dec,other.ra,other.dec,radius_in_degree,**kwargs)


    def get_extinction(self, band, camera='DES'):
        """
        Return SFD extinction given ``band`` and ``camera``.

        Calls ``utils.get_extinction()``.

        Parameters
        ----------
        band : string
            Photometric band.

        camera : string, default=`DES`
            camera.

        Returns
        -------
        extinction : ndarray
            Extinction (mag).
        """
        return utils.get_extinction(self.ra,self.dec,band=band,camera=camera)

class BrickCatalog(BaseCatalog):

    """Extend ``BaseCatalog`` with convenient methods for bricks."""

    def __init__(self, survey=None):
        """
        Load bricks.

        Parameters
        ----------
        survey : LegacySurveyData, string, default=None
            ``survey_dir`` or ``survey``.
            If ``None``, construct ``BrickCatalog`` from ``desiutil.brick.Bricks``.
        """
        if survey is not None:
            if not isinstance(survey,LegacySurveyData):
                survey = LegacySurveyData(survey_dir=survey)
            bricks_fn = survey.find_file('bricks')
            logger.info('Reading bricks from %s',bricks_fn)
            super(BrickCatalog,self).__init__(bricks_fn)
        else:
            logger.info('Building bricks from desiutil.brick')
            from desiutil.brick import Bricks
            table = Bricks().to_table()
            super(BrickCatalog,self).__init__(table,use_fitsio=False)
            self.to_np_arrays()
        assert np.all(self.brickid == 1+self.index())
        self._rowmax = self.brickrow.max()
        self._colmax = self.brickcol.max()
        self._ncols = np.bincount(self.brickrow)
        # should be applied mask as other columns, so no _
        self.hash = self.brickrow * (self._colmax + 1) + self.brickcol

    def get_hash(self):
        """Return ``self.hash`` enfrocing ``dim == 1`` (not the case after e.g. self[0])."""
        return np.atleast_1d(self.hash)

    def get_by_name(self, brickname):
        """
        Return bricks with name ``brickname``.

        Parameters
        ----------
        brickname : string, array-like
            If string, return a single-element ``BrickCatalog``, else a ``BrickCatalog`` with defined ``len()``.

        Returns
        -------
        bricks : BrickCatalog
            ``self`` cut to ``brickname``.
        """
        if np.isscalar(brickname):
            index = np.flatnonzero(self.brickname == brickname)[0]
        else:
            uniques,inverse = np.unique(brickname,return_inverse=True)
            index = np.array([np.flatnonzero(self.brickname == brickname_)[0] for brickname_ in uniques])[inverse]
        return self[index]

    def get_by_radec(self, ra, dec):
        """
        Return bricks containing ``ra``, ``dec``.

        Parameters
        ----------
        ra : float, array-like
            Right ascension (degree).

        dec : float, array-like
            Declination (degree).

        Returns
        -------
        bricks : BrickCatalog
            ``self`` cut to bricks containing ``ra``, ``dec``.
        """
        if not np.isscalar(ra):
            ra, dec = np.array(ra), np.array(dec)
        row = np.int32(np.floor(dec * self._rowmax / 180. + 360. + 0.5))
        row = np.clip(row, 0, self._rowmax)
        ncols = self._ncols[row]
        col = np.int32(np.floor(ra * ncols / 360. ))
        ind = self.get_hash().searchsorted(row * (self._colmax + 1) + col)
        return self[ind]

    def get_radecbox(self, total=False):
        """
        Return ra, dec box (ramin, ramax, decmin, decmax) of individual bricks.

        Parameters
        ----------
        total : bool, default=False
            If ``True``, returns the ra, dec box enclosing all bricks.

        Returns
        -------
        (ramin, ramax, decmin, decmax) : tuple of float, ndarray
            ra, dec box.
        """
        toret = [self.get(key) for key in ['ra1','ra2','dec1','dec2']]
        if total:
            toret = [np.min(toret[0]),np.max(toret[1]),np.min(toret[2]),np.max(toret[3])]
        return toret

    def get_area(self, total=False):
        """
        Return area of individual bricks.

        Calls ``utils.get_radecbox_area``.

        Parameters
        ----------
        total : bool, default=False
            If ``True``, returns total area.

        Returns
        -------
        area : float, ndarray
            Area (degree^2).
        """
        area = utils.get_radecbox_area(self.ra1,self.ra2,self.dec1,self.dec2)
        if total:
            area = np.sum(area)
        return area

    def get_xy_from_radec(self, ra, dec, brickname=None):
        """
        Returns brick ``bx``, ``by`` given ``ra``, ``dec``.

        Parameters
        ----------
        ra : float, array-like
            Right ascension (degree).

        dec : float, array-like
            Declination (degree).

        brickname : string, array-like, default=None
            Brick name. If array-like, should be of the same length as ``ra``, ``dec``.
            If ``None``, will query the correct ``brickname`` given ``ra``, ``dec``.

        Returns
        -------
        bx : float, ndarray
            Brick x coordinate.

        by : float, ndarray
            Brick y coordinate.
        """
        if np.isscalar(brickname):
            bx, by = wcs_for_brick(self.get_by_name(brickname)).radec2pixelxy(ra,dec)[1:]
            return bx-1, by-1 # legacyipe convention: zero-indexed
        elif brickname is None:
            brickname = np.atleast_1d(self.get_by_radec(ra,dec).brickname)
        isscalar = np.isscalar(ra)
        ra,dec = np.atleast_1d(ra),np.atleast_1d(dec)
        brickname = np.asarray(brickname)
        bx,by = np.zeros_like(ra),np.zeros_like(dec)
        for brickname_ in np.unique(brickname):
            mask = brickname == brickname_
            bx[mask],by[mask] = self.get_xy_from_radec(ra[mask],dec[mask],brickname_)
        if isscalar:
            return bx[0],by[0]
        return bx,by

    def write_list(self, fn):
        """
        Write brick names to ``fn``.

        Parameters
        ----------
        fn : string
            Path to brick list.
        """
        utils.mkdir(os.path.dirname(fn))
        with open(fn,'w') as file:
            for brickname in self.brickname:
                file.write('%s\n' % brickname)
