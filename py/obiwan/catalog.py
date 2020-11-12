'''
Convenient classes to handle catalogs and bricks.
'''

import os
import logging
import numpy as np
from astrometry.util import fits
from legacypipe.survey import LegacySurveyData
from . import utils

logger = logging.getLogger('obiwan.catalog')

class BaseCatalog(fits.tabledata):

    '''
    Class that inherits astrometry.util.fits.tabledata, with convenient methods.
    '''
    def __init__(self,*args,**kwargs):
        '''
        Calls astrometry.util.fits.fits_table().
        For an empty catalog, one provide size or length, to define len(self),
        used in methods to build np arrays (e.g. self.zeros()).
        Otherwise, len(self) is set when the first column is added to self.
        '''
        length = kwargs.pop('length',kwargs.pop('size',None))
        table = fits.fits_table(*args,**kwargs)
        self.__dict__.update(table.__dict__)
        if length is not None: self._length = length

    @property
    def size(self):
        '''
        Catalog size = len(self).
        '''
        return len(self)

    @property
    def fields(self):
        '''
        Catalog fields = self.get_columns(internal=False).
        '''
        return self.get_columns(internal=False)

    def __contains__(self,name):
        '''
        Whether self contains column name.
        '''
        return (name in self.fields)

    def zeros(self,dtype=np.float64):
        '''
        Returns array of size len(self) filled with zeros.
        '''
        return np.zeros(len(self),dtype=dtype)

    def ones(self,dtype=np.float64):
        '''
        Returns array of size len(self) filled with zeros.
        '''
        return np.ones(len(self),dtype=dtype)

    def falses(self):
        '''
        Returns array of size len(self) filled with False.
        '''
        return self.zeros(dtype=np.bool_)

    def trues(self):
        '''
        Returns array of size len(self) filled with True.
        '''
        return self.ones(dtype=np.bool_)

    def nans(self):
        '''
        Returns array of size len(self) filled with NaN.
        '''
        return self.ones()*np.nan

    def delete_columns(self,*fields):
        '''
        Delete columns.
        '''
        for field in fields:
            self.delete_column(field)

    def keep_columns(self,*fields):
        '''
        Keep only columns in argument.
        '''
        for field in self.fields:
            if field not in fields:
                self.delete_column(field)

    def merge(self,other,index_self=None,index_other=None,fill_nan=True):
        '''
        Merges `other` catalog to self.

        Parameters
        ----------
        other : BaseCatalog
            Catalog to be merged to self.

        index_self : ndarray, default=None
            Indices (or bool mask) of self to fill with `other`.
            If None, all indices of self are considered.

        index_other : ndarray, default=None
            Indices (or bool mask) of `other` to fill self.
            If None, all indices of other are considered.

        fill_nan : bool, default=True
            Fill empty float arrays with NaN.

        '''
        if index_self is None:
            index_self = self.trues()
        if index_other is None:
            index_other = other.trues()

        for field in other.fields:
            col_other = other.get(field)
            if field in self:
                col_self = self.get(field)
            else:
                col_self = np.zeros(shape=(self.size,)+col_other.shape[1:],dtype=col_other.dtype)
                if fill_nan and isinstance(col_self[0],float):
                    col_self[:] = np.nan
            col_self[index_self] = col_other[index_other]
            self.set(field,col_self)

    def copy(self):
        '''
        Returns a copy of self.
        '''
        # ugly hack, because of tabledata.copy() definition...
        toret = object.__new__(self.__class__)
        toret.__dict__.update(super(BaseCatalog,self).copy().__dict__)
        return toret

    def __eq__(self,other):
        '''
        Tests whether self == other (same type, length colums).
        '''
        if not isinstance(other,self.__class__):
            return False
        if not set(self.fields) == set(other.fields):
            return False
        if not self.size == other.size:
            return False
        for field in self.fields:
            mask = ~np.isnan(self.get(field))
            if not np.all(self.get(field)[mask] == other.get(field)[mask]):
                return False
        return True

    def __radd__(self,other):
        if other == 0:
            return self.copy()
        return self.__add__(other)

    def __ladd__(self,other):
        if other == 0:
            return self.copy()
        return self.__add__(other)

    def __add__(self,other):
        '''
        Appends self and other catalogs (total length = len(self) + len(other))
        and return a new instance.
        '''
        toret = self.copy()
        if other == 0:
            return toret
        if not other.fields: #empty catalog
            return toret
        toret.append(other)
        return toret

    def writeto(self,fn,*args,**kwargs):
        '''
        Saves to fn.
        '''
        logger.info('Wrote %s' % fn)
        utils.mkdir(os.path.dirname(fn))
        super(BaseCatalog,self).writeto(fn,*args,**kwargs)

class SimCatalog(BaseCatalog):

    '''
    Class that inherits BaseCatalog, with convenient methods for Obiwan randoms.
    '''

    def fill_obiwan(self,survey=None):
        '''
        Mandatory columns for Obiwan are:
            ra, dec : coordinates
            flux_g, flux_r, flux_z : in nanomaggies (with galactic extinction)
            sersic : Sersic index
            shape_r : half light radius
            shape_e1, shape_e2 : ellipticities, see https://www.legacysurvey.org/dr8/catalogs/

        Fills self with the required columns (if not already in self):
            id : defaults to np.arange(len(self))
            brickname, (brick) x, y: based on survey and self.ra, self.dec.

        Parameters
        ----------
        survey : LegacySurveyData, string, default=None
            survey_dir or survey. Used to determine brick-related quantities (brickname, x, y).
            If None, constructs BrickCatalog from desiutil.brick.Bricks.

        '''
        bricks = None
        if 'brickname' not in self:
            bricks = BrickCatalog(survey)
            self.brickname = bricks.get_by_radec(self.ra,self.dec).brickname

        if 'id' not in self:
            self.id = np.arange(len(self))

        if 'bx' not in self:
            if bricks is None: bricks = BrickCatalog(survey)
            self.bx,self.by = bricks.get_xy_from_radec(self.ra,self.dec,brickname=self.brickname)

        #if 'added' not in self:
        #    self.added = self.falses()

    def mask_collisions(self,radius_in_degree=5./3600.):
        '''
        Returns mask of collided objects.

        Parameters
        ----------
        radius_in_degree : float, default=5./3600.
            Collision radius (degree).

        Returns
        -------
        mask : bool ndarray
            Mask of collided objects.

        '''
        return utils.mask_collisions(self.ra,self.dec,radius_in_degree=radius_in_degree)

    def match_radec(self,other,radius_in_degree=5./3600.,**kwargs):
        '''
        Match self and other SimCatalog ra, dec.

        Parameters
        ----------
        other : SimCatalog
            Catalog to be matched against self.

        radius_in_degree : float, default=5./3600.
            Collision radius (degree).

        kwargs : dict, default={}
            Arguments to utils.match_radec

        Returns
        -------
        ind_self : ndarray
            Indices into self of matching points.

        ind_other : ndarray
            Indices into other of matching points.
        '''

        return utils.match_radec(self.ra,self.dec,other.ra,other.dec,radius_in_degree,**kwargs)


    def get_extinction(self,band,filter='DES'):
        '''
        Returns SFD extinction given band and filter.

        Parameters
        ----------
        band : string
            Photometric band.

        filter : string, default=`DES`
            Filter.

        Returns
        -------
        extinction : ndarray
            Extinction (mag).
        '''
        return utils.get_extinction(self.ra,self.dec,band=band,filter=filter)

class BrickCatalog(BaseCatalog):
    '''
    Class that inherits BaseCatalog, with convenient methods for bricks.
    '''

    def __init__(self,survey=None):
        '''
        Load bricks from LegacySurveyData or string.

        Parameters
        ----------
        survey : LegacySurveyData, string, default=None
            survey_dir or survey.
            If None, constructs BrickCatalog from desiutil.brick.Bricks.

        '''
        if survey is not None:
            if not isinstance(survey,LegacySurveyData):
                survey = LegacySurveyData(survey_dir=survey)
                super(BrickCatalog,self).__init__(survey.find_file('bricks'))
        else:
            from desiutil.brick import Bricks
            table = Bricks().to_table()
            super(BrickCatalog,self).__init__(table,use_fitsio=False)
            self.to_np_arrays()
        assert np.all(self.brickid == 1+np.arange(self.size))
        self._rowmax = self.brickrow.max()
        self._colmax = self.brickcol.max()
        self._ncols = np.bincount(self.brickrow)
        # should be applied mask as other columns, so no _
        self.hash = self.brickrow * (self._colmax + 1) + self.brickcol

    def get_by_name(self,brickname):
        '''
        Returns bricks with name brickname.

        Parameters
        ----------
        brickname : string, array-like
            If string, returns a single-element BrickCatalog, else a BrickCatalog with defined len().

        Returns
        -------
        bricks : BrickCatalog
            self cut to brickname.
        '''
        if isinstance(brickname,str):
            index = np.flatnonzero(self.brickname == brickname)[0]
        else:
            uniques,inverse = np.unique(brickname,return_inverse=True)
            index = np.array([np.flatnonzero(self.brickname == brickname_)[0] for brickname_ in uniques])[inverse]
        return self[index]

    def get_by_radec(self,ra,dec):
        '''
        Returns bricks containing ra, dec.

        Parameters
        ----------
        ra : float, array-like
            Right ascension (degree).

        dec : float, array-like
            Declination (degree).

        Returns
        -------
        bricks : BrickCatalog
            self cut to bricks containing ra, dec.
        '''
        if not np.isscalar(ra):
            ra, dec = np.array(ra), np.array(dec)
        row = np.int32(np.floor(dec * self._rowmax / 180. + 360. + 0.5))
        row = np.clip(row, 0, self._rowmax)
        ncols = self._ncols[row]
        col = np.int32(np.floor(ra * ncols / 360. ))
        ind = self.hash.searchsorted(row * (self._colmax + 1) + col)
        return self[ind]

    def get_radecbox(self,all=False):
        '''
        Returns ramin, ramax, decmin, decmax of individual bricks.

        Parameters
        ----------
        all : bool, default=False
            If True, returns the ra, dec box enclosing all bricks.

        Returns
        -------
        (ramin, ramax, decmin, decmax) : tuple of float, ndarray
            ra, dec box.
        '''
        toret = [self.get(key) for key in ['ra1','ra2','dec1','dec2']]
        if all:
            toret = [np.min(toret[0]),np.max(toret[1]),np.min(toret[2]),np.max(toret[3])]
        return toret

    def get_area(self,all=False):
        '''
        Returns area of individual bricks.

        Parameters
        ----------
        all : bool, default=False
            If True, returns total area.

        Returns
        -------
        area : float, ndarray
            Area (degree^2).
        '''
        area = utils.get_radecbox_area(self.ra1,self.ra2,self.dec1,self.dec2)
        if all:
            area = np.sum(area)
        return area

    def get_xy_from_radec(self,ra,dec,brickname=None):
        '''
        Returns brick x, y given ra, dec.

        Parameters
        ----------
        ra : float, array-like
            Right ascension (degree).

        dec : float, array-like
            Declination (degree).

        brickname : string, array-like, default=None
            Brick name. If array-like, should be of the same length as ra, dec.
            If None, will query the correct brickname given ra, dec.

        Returns
        -------
        bx : float, ndarray
            Brick corrdinate x.

        by : float, ndarray
            Brick corrdinate y.
        '''
        if isinstance(brickname,str):
            flag, bx, by = utils.wcs_for_brick(self.get_by_name(brickname)).radec2pixelxy(ra,dec)
            return bx, by
        elif brickname is None:
            brickname = np.atleast_1d(self.get_by_radec(ra,dec).brickname)
        if not np.isscalar(ra):
            ra, dec = np.array(ra), np.array(dec)
        bx,by = [],[]
        for brickname_ in np.unique(brickname):
            mask = brickname == brickname_
            x_,y_ = self.get_xy_from_radec(ra[mask],dec[mask],brickname_)
            bx.append(x_),by.append(y_)
        if np.isscalar(ra):
            return bx[0],by[0]
        return np.concatenate(bx),np.concatenate(by)

    def write_list(self,fn):
        '''
        Writes brick names to fn.

        Parameters
        ----------
        fn : string
            Path to brick list.
        '''
        utils.mkdir(os.path.dirname(fn))
        with open(fn,'w') as file:
            for brickname in self.brickname:
                file.write(brickname+'\n')
