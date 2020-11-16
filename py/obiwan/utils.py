"""Convenient functions to handle Obiwan inputs/outputs."""

import os
import sys
import logging
import functools
import numpy as np
from matplotlib import pyplot as plt
from astrometry.libkd import spherematch
from .kenobi import LegacySurveySim

logger = logging.getLogger('obiwan.utils')

def setup_logging(level=logging.INFO):
    """Set up logging, legacypipe style..."""
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stdout)

def get_survey_file(survey_dir,filetype,brickname=None,**kwargs_file):
    """
    Return survey file name.

    survey_dir : string
        Survey directory.

    filetype : string
        Type of file to find.

    brickname : string
        Brick name.

    kwargs_file : dict
        Other arguments to file paths (``fileid``, ``rowstart``, ``skipid``).
    """
    survey = LegacySurveySim(survey_dir=survey_dir,kwargs_file=kwargs_file)
    return survey.find_file(filetype,brick=brickname,output=False)

def get_output_file(output_dir,filetype,brickname=None,**kwargs_file):
    """
    Return Obiwan output file name.

    output_dir : string
        Obiwan output directory.

    filetype : string
        Type of file to find, including:
        `obiwan-randoms` -- Obiwan catalogues
        `obiwan-metadata` -- Obiwan metadata
        `tractor` -- Tractor catalogs
        `depth`   -- PSF depth maps
        `galdepth` -- Canonical galaxy depth maps
        `nexp` -- number-of-exposure maps.

    brickname : string
        Brick name.

    kwargs_file : dict
        Other arguments to file paths (``fileid``, ``rowstart``, ``skipid``).
    """
    survey = LegacySurveySim(output_dir=output_dir,kwargs_file=kwargs_file)
    return survey.find_file(filetype,brick=brickname,output=True)

def saveplot(giveax=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(ax=None,fn=None,kwargs_fig={},**kwargs):
            isax = True
            if giveax:
                if ax is None:
                    isax = False
                    ax = plt.gca()
                func(ax,**kwargs)
            else:
                isax = False
                func(**kwargs)
                if ax is None:
                    ax = plt.gca()
            if fn is not None:
                savefig(fn,**kwargs_fig)
            elif not isax:
                plt.show()
            return ax
        return wrapper
    return decorator

def savefig(fn,bbox_inches='tight',pad_inches=0.1,dpi=200,**kwargs):
    """Save matplotlib figure to ``fn``."""
    mkdir(os.path.dirname(fn))
    logger.info('Saving figure to %s.' % fn)
    plt.savefig(fn,bbox_inches=bbox_inches,pad_inches=pad_inches,dpi=dpi,**kwargs)
    plt.close(plt.gcf())

def mkdir(dirnm):
    """Check if ``dirnm`` exists, if not create it."""
    try: os.makedirs(dirnm) #MPI...
    except OSError: return

def get_git_version(dirnm=None):
    """
    Run ``git describe`` in the current directory (or given ``dirnm``) and return the result as a string.

    Taken from https://github.com/legacysurvey/legacypipe/blob/master/py/legacypipe/survey.py.

    Parameters
    ----------
    dirnm : string
        If non-None, ``cd`` to the given directory before running ``git describe``.

    Returns
    -------
    version : string
        Git version string.
    """
    from astrometry.util.run_command import run_command
    cmd = ''
    if dirnm is None:
        # Get the git version of the legacypipe product
        import obiwan
        dirnm = os.path.dirname(obiwan.__file__)

    cmd = "cd '%s' && git describe" % dirnm
    rtn,version,err = run_command(cmd)
    if rtn:
        raise RuntimeError('Failed to get version string (%s): ' % cmd +
                           version + err)
    version = version.strip()
    return version

def dict_default(self,other):
    """Update other with ``self`` and return the result."""
    toret = {}
    toret.update(other)
    toret.update(self)
    return toret

def get_parser_args(parser,exclude=['help']):
    """
    Return parser list of ``dest``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser.

    exclude : string
        Brick name.

    Returns
    -------
    args : list
        Parser list of ``dest``.
    """
    return [act.dest for act in parser._actions if act.dest not in exclude]

def sample_ra_dec(size=None,radecbox=[0.,360.,-90.,90.],rng=None,seed=None):
    """
    Sample uniform ra, dec coordinates in ``radecbox``.

    Taken from https://github.com/desihub/imaginglss/blob/master/scripts/imglss-mpi-make-random.py.

    Parameters
    ----------
    size : int, default=None
        Number of objects. If ``None``, only a single tuple ra, dec will be sampled.

    radecbox : list, default=[0.,360.,-90.,90.]
        ramin, ramax, decmin, decmax.

    rng : np.random.RandomState, default=None
        Random state; If ``None``, build RandomState from ``seed``.

    seed : int, default=None
        Random seed, only used if ``rng`` is None.

    Returns
    -------
    ra : float, ndarray
        Right ascension (degree).

    dec : float, ndarray
        Declination (degree).
    """
    if rng is None:
        rng = np.random.RandomState(seed=seed)

    ramin,ramax,decmin,decmax = radecbox

    u1,u2 = rng.uniform(size=(2,size) if size is not None else 2)
    #
    cmin = np.sin(np.deg2rad(decmin))
    cmax = np.sin(np.deg2rad(decmax))
    #
    ra = ramin + u1*(ramax-ramin)
    dec = 90.-np.rad2deg(np.arccos(cmin+u2*(cmax-cmin)))

    return ra, dec

def match_radec(ra1,dec1,ra2,dec2,radius_in_degree=None,return_distance=False,notself=False,nearest=True):
    """
    Match ra2,dec2 to ra1,dec1.

    All quantities in degree.
    Uses https://github.com/dstndstn/astrometry.net/blob/master/libkd/spherematch.py.

    Parameters
    ----------
    ra1 : array-like
        Right ascension 1 (degree).

    dec1 : array-like
        Declination 1 (degree).

    ra2 : array-like
        Right ascension 2 (degree).

    dec2 : array-like
        Declination 2 (degree).

    radius_in_degree : float,default=None
        If not None, maximum radius (degree) to match ra, dec pairs.

    return_distance : bool, default=False
        If ``True``, return distance.

    notself : bool, default=False
        If ``True``, avoids returning 'identity' matches, i.e. ra1,dec1 == ra2,dec2.

    nearest : bool, default=True
        If ``True``, returns only the nearest match in (ra2,dec2) for each point in (ra1,dec1).
        Else, returns all matches (index1 and index2 are not made of unique elements).

    Returns
    -------
    index1 : ndarray
        Indices of ra1,dec1 matching points.

    index2 : ndarray
        Indices of ra2,dec2 matching points.

    distance : ndarray
        Distance (degree).
    """
    provided_radius = radius_in_degree is not None
    if not provided_radius: radius_in_degree=360.
    index1,index2,distance = spherematch.match_radec(ra1,dec1,ra2,dec2,radius_in_degree,notself=notself,nearest=nearest)
    if provided_radius:
        mask = distance<radius_in_degree
        index1,index2,distance = index1[mask],index2[mask],distance[mask]

    if return_distance:
        return index1,index2,distance
    return index1,index2

def mask_collisions(ra, dec, radius_in_degree=5./3600.):
    """
    Return mask of collided objects.

    Parameters
    ----------
    ra : array-like
        Right ascension (degree).

    dec : array-like
        Declination (degree).

    radius_in_degree : float, default=5./3600.
        Collision radius (degree).

    Returns
    -------
    mask : bool ndarray
        Mask of collided objects.
    """
    skip = set()
    for ii,(ra_,dec_) in enumerate(zip(ra,dec)):
        if ii in skip: continue
        j = match_radec(ra_,dec_,ra,dec,radius_in_degree,
                            notself=False,nearest=False)[1]
        skip |= set(j[j!=ii]) # removes self-pair
    mask = np.zeros_like(ra,dtype=np.bool_)
    skip = np.array(list(skip))
    if skip.size:
        mask[skip] = True
    return mask

def get_radecbox_area(ramin,ramax,decmin,decmax):
    """
    Return area of ra, dec box.

    Parameters
    ----------
    ramin : float, array-like
        Minimum right ascension (degree).

    ramax : float, array-like
        Maximum right ascension (degree).

    decmin : float, array-like
        Minimum declination (degree).

    decmax : float, array-like
        Maximum declination (degree).

    Returns
    -------
    area : float, ndarray.
        Area (degree^2).
    """
    decfrac = np.diff(np.rad2deg(np.sin(np.deg2rad([decmin,decmax]))),axis=0)
    rafrac = np.diff([ramin,ramax],axis=0)
    area = decfrac*rafrac
    if np.isscalar(ramin):
        return area[0]
    return area

def get_shape_e(ba):
    """
    Return ellipticity ``e`` given minor-to-major axis ratio ``ba``.

    Parameters
    ----------
    ba : float, ndarray
        Minor-to-major axis ratio (b/a).

    Returns
    -------
    e : float, ndarray
        Ellipticity.

    References
    ----------
    https://www.legacysurvey.org/dr8/catalogs/
    """
    return (1.-ba)/(1.+ba)

def get_shape_e1_e2(ba,phi):
    """
    Return ellipticities e1, e2 given minor-to-major axis ratio ``ba`` and angle ``phi``.

    Parameters
    ----------
    ba : float, ndarray
        Minor-to-major axis ratio (b/a).

    phi : float, ndarray
        Angle in radians.

    Returns
    -------
    e1 : float, ndarray
        Ellipticity component 1.

    e2 : float, ndarray
        Ellipticity component 2.

    References
    ----------
    https://www.legacysurvey.org/dr8/catalogs/
    """
    e = get_shape_e(ba)
    return e*np.cos(2*phi), e*np.sin(2*phi)

def get_shape_ba(e):
    """
    Return minor-to-major axis ratio ``ba`` given ellipticity ``e``.

    Parameters
    ----------
    e : float, ndarray
        Ellipticity.

    Returns
    -------
    ba : float, ndarray
        Minor-to-major axis ratio (b/a).

    References
    ----------
    https://www.legacysurvey.org/dr8/catalogs/
    """
    return (1.-np.abs(e))/(1.+np.abs(e))

def get_shape_ba_phi(e1,e2):
    """
    Return minor-to-major axis ratio ``ba`` and angle ``phi`` given ellipticities ``e1``, ``e2``.

    Parameters
    ----------
    e1 : float, ndarray
        Ellipticity component 1.

    e2 : float, ndarray
        Ellipticity component 2.

    Returns
    -------
    ba : float, ndarray
        Minor-to-major axis ratio (b/a).

    phi : float, ndarray
        Angle (radian).

    References
    ----------
    https://www.legacysurvey.org/dr8/catalogs/
    """
    ba = get_shape_ba((e1**2+e2**2)**0.5)
    phi = 0.5*np.arctan2(e2,e1) % (2.*np.pi)
    return ba, phi

def get_extinction(ra,dec,band=None,camera='DES'):
    """
    Return SFD extinction given ``ra``, ``dec``, ``band`` and ``camera``.

    If ``band`` not provided, return EBV.

    Parameters
    ----------
    ra : float, array-like
        Right ascension (degree).

    dec : float, array-like
        Declination (degree).

    band : string, default=None
        Photometric band. If not provided, returns EBV.

    camera : string, default=`DES`
        Camera.

    Returns
    -------
    extinction : float, ndarray
        Extinction.
    """
    from tractor.sfd import SFDMap
    sfd = SFDMap()
    if band is None:
        c = 1
    else:
        c = sfd.extinctions['%s %s' % (camera,band)]
    return c*sfd.ebv(ra, dec)

def mag2nano(mag):
    """Magnitudes to nanomaggies conversion."""
    return 10. ** ((mag - 22.5) / -2.5)

def nano2mag(nano):
    """Nanomaggies to magnitudes conversion."""
    return -2.5 * (np.log10(nano) - 9)