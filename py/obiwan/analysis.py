import os
import logging
import glob
import argparse
import numpy as np
from scipy import stats,special
from matplotlib import pyplot as plt
import fitsio
from .kenobi import find_file,get_randoms_id
from .catalog import BaseCatalog,SimCatalog
from . import utils

logger = logging.getLogger('obiwan.analysis')

class BaseImage(object):

    """Dumb class to read, slice and plot an image."""

    def read_image(self, fn, fmt='jpeg', xmin=0, ymin=0, **kwargs):
        """
        Read image, either in 'jpeg' or 'fits' fmt and add ``img`` to ``self``.

        Parameters
        ----------
        fn : string
            Image filename.

        fmt : string, defaut='jpeg'
            Image format, 'jpeg' or 'fits'.
            If 'jpeg', ``self.img`` will have shape ``(H,W,3)``.
            If 'fits', ``self.img`` will have shape ``(H,W)``.

        xmin : int, float, default=0
            x-coordinate of ``(0,0)`` corner.

        ymin : int, float, default=0
            y-coordinate of ``(0,0)`` corner.

        kwargs : dict, default={}
            Arguments to be passed to ``plt.imread`` or ``fitsio.read``.

        Notes
        -----
        Extracted from https://github.com/legacysurvey/obiwan/blob/master/py/obiwan/qa/visual.py
        """
        self.img_fn = fn
        if fmt in ['jpg','jpeg']:
            #import skimage.io
            #img = skimage.io.imread(fn)
            img = np.array(plt.imread(fn,**kwargs))
            for i in range(3):
                img[:,:,i] = np.rot90(img[:,:,i].T,1)
        elif fmt.startswith('fits'):
            img = fitsio.read(fn,**kwargs)
        else:
            raise ValueError('Unkown image format %s' % fmt)
        self.img = img
        self.xmin = xmin
        self.ymin = ymin

    def set_subimage(self, slicex=slice(0,None), slicey=slice(0,None)):
        """
        Slice image (``self.img``) and update ``self.xmin``, ``self.ymin`` accordingly.

        Slices should be zero-indexed, in the image coordinates
        (i.e. do not account for ``self.xmin`` and ``self.ymin``).

        Parameters
        ----------
        slicex : slice
            Slice along x-axis (second axis of ``self.img``).

        slicey : slice
            Slice along y-axis (first axis of ``self.img``).
        """
        self.xmin = self.xmin + slicex.start
        self.ymin = self.ymin + slicey.start
        self.img = self.img[slicey,slicex]

    @property
    def shape(self):
        """
        Return image shape (H,W).

        If image does not exist, defaults to ``self.baseshape``.
        """
        if not hasattr(self,'img'): return self.baseshape
        return self.img.shape

    def isRGB(self):
        """Whether image is multi-dimensional."""
        return len(self.img.shape) == 3

    @utils.saveplot()
    def plot(self, ax, cmap='gray', vmin=None, vmax=None, kwargs_range={}):
        """
        Plot image.

        Parameters
        ----------
        ax : plt.axes
            Where to plot image.
            One can also provide figure file name ``fn``.
            See ``utils.saveplot``.

        cmap : string, plt.Colormap, default='gray'
            Color map for ``plt.imshow``, ignore in case of RGB(A) image.

        vmin : float, default=None
            Minimum value for ``cmap``.

        vmax : float, default=None
            Maximum value for ``cmap``.

        kwargs_range : dict, default={}
            If provided, provided to ``Binning`` to determine ``vmin`` and ``vmax``.
        """
        if kwargs_range:
            vmin,vmax = Binning(samples=self.img.flatten(),nbins=1,**kwargs_range).range
        ax.imshow(self.img,interpolation='none',origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)

class ImageAnalysis(BaseImage):

    """Extend ``BaseImage`` with **Obiwan**-related convenience functions."""

    baseshape = (3600,3600)

    def __init__(self, base_dir='.', brickname=None, kwargs_file={}):
        """
        Initialize ``ImageAnalysis`` by setting the image path in **Obiwan** file structure.

        Parameters
        ----------
        base_dir : string, default='.'
            **Obiwan** root file directory.

        brickname : string, default=None
            Brick name.

        kwargs_file : dict, default={}
            Other arguments to file paths (e.g. ``get_randoms_id.keys()``).
        """
        self.base_dir = base_dir
        self.brickname = brickname
        self.kwargs_file = kwargs_file

    def read_image(self, filetype='image-jpeg', band=['g','r','z'], xmin=0, ymin=0):
        """
        Read **Obiwan** image and add ``img`` to ``self``.

        Parameters
        ----------
        filetype : string, default='image-jpeg'
            Image filetype. See ``legacypipe.survey.find_file``.

        band : list, string, default=['g','r','z']
            Image band(s). Only used in case the image is in 'fits' format.
            In this case, if band is a list, should be of length 3 to build up a RGB image
            with ``legacypipe.survey.get_rgb``.
            Else, only the image in the corresponding band is read.

        xmin : int, float, default=0
            Brick x-coordinate of ``(0,0)`` corner.

        ymin : int, float, default=0
            Brick y-coordinate of ``(0,0)`` corner.
        """
        fmt = 'jpeg' if 'jpeg' in filetype else 'fits'
        if fmt == 'fits' and not np.isscalar(band):
            assert len(band) == 3
            img = []
            for b in band:
                fn = find_file(self.base_dir,filetype,brickname=self.brickname,source='obiwan',band=b,**self.kwargs_file)
                super(ImageAnalysis,self).read_image(fn=fn,fmt=fmt,xmin=xmin,ymin=ymin)
                img.append(self.img)
            self.img = np.moveaxis(img,0,-1)
            from legacypipe.survey import get_rgb
            self.img = get_rgb(self.img,bands=band)
        else:
            fn = find_file(self.base_dir,filetype,brickname=self.brickname,source='obiwan',band=band,**self.kwargs_file)
            super(ImageAnalysis,self).read_image(fn=fn,fmt=fmt,xmin=xmin,ymin=ymin)

    def read_sources(self, filetype='randoms'):
        """
        Read sources in the image add ``sources`` to ``self``.

        Parameters
        ----------
        filetype : string, default='randoms'
            Source filetype. See ``kenobi.find_file``.
        """
        self.sources_fn = find_file(self.base_dir,filetype,brickname=self.brickname,source='obiwan',**self.kwargs_file)
        self.sources = SimCatalog(self.sources_fn)
        if hasattr(self,'collided'):
            # Remove sources that were not injected
            self.sources = self.sources[~self.sources.collided]

    def suggest_zooms(self, boxsize_in_pixels=None, match_in_degree=0.1/3600, range_observed_injected_in_degree=[5./3600,30./3600]):
        """
        Suggest image cutouts around injected sources and within some distance to true sources.

        Parameters
        ----------
        boxsize_in_pixels : int, default=None
            Box size (pixel) around the injected source.
            If ``None``, defaults to image smaller side divided by 36.

        match_in_degree : float, default=0.1/3600
            Radius (degree) to match injected to output sources.

        range_observed_injected_in_degree : list, default=[5./3600,30./3600]
            Range (degree) around the injected source where to find a true source.

        Returns
        -------
        slices : list
            List of slice ``(slicex,slicey)``, to be passed to ``self.set_subimage.``
        """
        fn = find_file(self.base_dir,'tractor',brickname=self.brickname,source='obiwan',**self.kwargs_file)
        tractor = SimCatalog(fn)
        index_sources,index_tractor,distance = self.sources.match_radec(tractor,radius_in_degree=range_observed_injected_in_degree[-1],nearest=False,return_distance=True)
        matched_sources = index_sources[distance<match_in_degree]
        mask_matched = np.in1d(index_sources,matched_sources)
        if not mask_matched.any():
            raise ValueError('No match found between random and Tractor catalogs.')
        mask_inrange = mask_matched & (distance > range_observed_injected_in_degree[0]) #& (distance < range_observed_injected_in_degree[-1])
        if not mask_inrange.any():
            raise ValueError('No random/tractor pair found within range = %s, you should try larger range' % range_observed_injected_in_degree)
        index_sources = np.unique(index_sources[mask_inrange])
        if boxsize_in_pixels is None: boxsize_in_pixels = np.min(self.shape[:2])//36
        bx,by = np.rint(self.sources.bx[index_sources]).astype(int)-self.xmin,np.rint(self.sources.by[index_sources]).astype(int)-self.ymin
        halfsize = boxsize_in_pixels//2
        rangex = bx-halfsize,bx+boxsize_in_pixels-halfsize+1
        rangey = by-halfsize,by+boxsize_in_pixels-halfsize+1
        mask_boxsize = (rangex[0]>=0) & (rangex[-1]<=self.shape[1]) & (rangey[0]>=0) & (rangey[-1]<=self.shape[0])
        if not mask_boxsize.any():
            raise ValueError('Boxsize too large')
        rangex = tuple(r[mask_boxsize] for r in rangex)
        rangey = tuple(r[mask_boxsize] for r in rangey)
        slices = [(slice(rangex[0][i],rangex[1][i]),slice(rangey[0][i],rangey[1][i])) for i in range(mask_boxsize.sum())]
        return slices

    def plot_sources(self, ax, radius_in_pixel=3./0.262, dr=None, color='r'):
        """
        Plot circles around injected sources.

        Parameters
        ----------
        ax : plt.axes
            Where to plot sources.

        radius_in_pixel : float, default=3./0.262
            Radius (pixel) around sources.

        dr : float, default=None
            Circle width (pixel).
            If ``None``, defaults to ``radius_in_pixel/20``.

        color : string, default='r'
            Circle color.
        """
        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection
        if dr is None: dr = radius_in_pixel/20.
        patches = [Wedge((x-self.xmin,y-self.ymin),r=radius_in_pixel,theta1=0,theta2=360,width=dr) for x,y in zip(self.sources.bx,self.sources.by)]
        coll = PatchCollection(patches,color=color) #,alpha=1)
        ax.add_collection(coll)

class RunCatalog(BaseCatalog):

    """
    Extend ``BaseCatalog`` with convenient methods for run (i.e. brick x randoms id) lists.

    This is useful to navigate through the **Obiwan** or **legacypipe** file structure.
    """

    @staticmethod
    def read_bricklist(bricklist):
        """
        Read brick list and return list of unique brick names.

        Parameters
        ----------
        bricklist : list, string
            List of strings (or one string) corresponding either brick names or ASCII files containing a column of brick names.

        Returns
        -------
        bricknames : list
            List of (unique) brick names.
        """
        bricknames = []
        if np.isscalar(bricklist): bricklist = [bricklist]
        for brickname in bricklist:
            if os.path.isfile(brickname):
                logger.info('Reading brick list %s' % brickname)
                with open(brickname,'r') as file:
                    for line in file:
                        brickname = line.replace('\n','')
                        if brickname not in bricknames:
                            bricknames.append(brickname)
            else:
                bricknames.append(brickname)
        return bricknames

    @staticmethod
    def get_input_parser(parser=None):
        """
        Add parser arguments to define runs.

        Parameters
        ----------
        parser : argparse.ArgumentParser, default=None
            Parser to add arguments to. If ``None``, a new one is created.

        Returns
        -------
        parser : argparse.ArgumentParser
            Parser with arguments to define runs.
        """
        if parser is None:
            parser = argparse.ArgumentParser()
        for key,default in zip(get_randoms_id.keys(),get_randoms_id.defs()):
            parser.add_argument('--%s' % key, nargs='*', type=int, default=[default], help='Use these %ss.' % key)
        parser.add_argument('--brick', nargs='*', type=str, default=None, help='Use these bricknames. Can be a brick list file.')
        parser.add_argument('--list', nargs='*', type=str, default=None, help='Use these run lists. Overrides all other run arguments')
        return parser

    @staticmethod
    def get_output_parser(parser=None):
        """
        Add parser arguments to select runs among the **Obiwan** files.

        Parameters
        ----------
        parser : argparse.ArgumentParser, default=None
            Parser to add arguments to. If ``None``, a new one is created.

        Returns
        -------
        parser : argparse.ArgumentParser
            Parser with arguments to select runs among the **Obiwan** files.
        """
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--outdir', dest='output_dir', help='Output base directory, default "."')
        #parser.add_argument('--survey-dir', type=str, default=None, help='Override the $LEGACY_SURVEY_DIR environment variable')
        for key in get_randoms_id.keys():
            parser.add_argument('--%s' % key, nargs='*', type=int, default=None, help='If provided, restrict to these %ss.' % key)
        parser.add_argument('--brick', nargs='*', type=str, default=None, help='If provided, restrict to these bricknames. Can be a brick list file.')
        parser.add_argument('--list', nargs='*', type=str, default=None, help='Restrict to these run lists. Overrides all other run arguments')
        return parser

    @classmethod
    def from_input_cmdline(cls, opt):
        """
        Initialize ``RunCatalog`` from command-line options of ``RunCatalog.get_input_parser()``.

        Parameters
        ----------
        opt : argparse.Namespace, dict
            Holds options of ``RunCatalog.get_input_parser()``.

        Returns
        -------
        self : ``RunCatalog``
            ``RunCatalog`` object.
        """
        if not isinstance(opt,dict):
            opt = vars(opt)
        for key in ['list','brick']:
            if key not in opt: opt[key] = None
        if opt['list'] is not None:
            return cls.from_list(opt['list'])
        bricknames = cls.read_bricklist(opt['brick'])
        kwargs_files = []
        args = np.array([opt[key] for key in get_randoms_id.keys()]).T
        for arg in args:
            kwargs_files.append({key:val for key,val in zip(get_randoms_id.keys(),arg)})
        return cls.from_brick_ranid(bricknames=bricknames, kwargs_files=kwargs_files)

    @classmethod
    def from_output_cmdline(cls, opt, force_from_disk=False, filetype='tractor', source='obiwan'):
        """
        Initialize ``RunCatalog`` from command-line options of ``RunCatalog.get_output_parser()``.

        Parameters
        ----------
        opt : argparse.Namespace, dict
            Holds options of ``RunCatalog.get_output_parser()``.

        force_from_disk : bool, default=False
            If, ``True`` limit run list to files that exist on disk.

        filetype : string, default='Tractor'
            File type to look for on disk.

        source : string, default=`obiwan`
            If 'obiwan', search in **Obiwan** file structure, else in **legacypipe** file structure.

        Returns
        -------
        self : ``RunCatalog``
            ``RunCatalog`` object.
        """
        if not isinstance(opt,dict):
            opt = vars(opt)
        for key in ['list','brick'] + get_randoms_id.keys():
            if key not in opt: opt[key] = None
        if (not force_from_disk)\
            and ((opt['list'] is not None)\
                or (opt['brick'] is not None) and all([opt[key] is not None for key in get_randoms_id.keys()])):
            return cls.from_input_cmdline(opt)

        bricknames = None
        if opt['brick'] is not None:
            bricknames = cls.read_bricklist(opt['brick'])

        def decode_output_obiwan_fn(dirname):
            kwargs_file = get_randoms_id.split(os.path.basename(dirname))
            dirname = os.path.dirname(dirname)
            brickname = os.path.basename(dirname)
            return {**{self.fields[0]:brickname},**kwargs_file}

        def decode_output_legacypipe_fn(fullname):
            # not well-tested
            kwargs_file = get_randoms_id.as_dict()
            basename = os.path.basename(fullname)
            brickname = basename.split('-')[1].split('.')[0]
            return {**{self.fields[0]:brickname},**kwargs_file}

        if bricknames is None:
            if source == 'obiwan':
                pattern = os.path.join(opt['output_dir'],filetype,'*','*','*')
            else:
                pattern = os.path.join(opt['output_dir'],filetype,'*','*')
            fns = glob.iglob(pattern)
        else:
            fns = []
            for brickname in bricknames:
                if source == 'obiwan':
                    pattern = os.path.join(opt['output_dir'],filetype,brickname[:3],brickname,'*')
                else:
                    pattern = os.path.join(opt['output_dir'],filetype,brickname[:3],'*')
                fns.append(glob.iglob(pattern))
            import itertools
            fns = itertools.chain(*fns) # chain iterators

        self = cls()
        for field in self.fields:
            self.set(field,[])

        for fn in fns:
            if source == 'obiwan':
                decode = decode_output_obiwan_fn(fn)
            else:
                decode = decode_output_legacypipe_fn(fn)
            for field in self.fields:
                self.get(field).append(decode[field])
        self.to_np_arrays()

        # restrictions
        mask = self.trues()
        if opt['list'] is not None:
            other = cls.from_dict(opt['list'])
            mask &= self.in1d(other)
        if bricknames is not None:
            mask &= np.in1d(self.brickname,bricknames)
        for key in get_randoms_id.keys():
            if opt[key] is not None:
                mask &= self.get(key) == opt[key]

        self = self[mask]
        return self.prune()

    @classmethod
    def from_brick_ranid(cls, bricknames=[], kwargs_files={}):
        """
        Initialize ``RunCatalog`` from a list of bricks and randoms file ids.

        Parameters
        ----------
        bricknames : list, string, default=[]
            List of brick names (or a single brick name).

        kwargs_files : list, dict, default=None
            List of ``get_randoms_id`` dictionaries (or a single dictionary).

        Returns
        -------
        self : ``RunCatalog``
            ``RunCatalog`` object.
        """
        if np.isscalar(bricknames): bricknames = [bricknames]

        if isinstance(kwargs_files,dict): kwargs_files = [kwargs_files]
        self = cls()
        for field in self.fields: self.set(field,[])
        for brickname in bricknames:
            for kwargs_file in kwargs_files:
                tmp = {**{self.fields[0]:brickname},**get_randoms_id.as_dict(**kwargs_file)}
                for field in self.fields:
                    self.get(field).append(tmp[field])
        self.to_np_arrays()
        return self

    @classmethod
    def from_catalog(cls, cat):
        """
        Initialize ``RunCatalog`` from a catalog.

        Parameters
        ----------
        cat : BaseCatalog
            Catalog containing ``brickname`` and ``get_randoms_id`` arguments.

        Returns
        -------
        self : ``RunCatalog``
            ``RunCatalog`` object.
        """
        self = cls()
        for field in self.fields: self.set(field,np.array(cat.get(field)))
        return self.prune()

    @property
    def fields(self):
        """Return fields."""
        return ['brickname'] + get_randoms_id.keys()

    def unique(self, field):
        """Return unique values of column ``field``."""
        return np.unique(self.get(field))

    def uniqid(self):
        """Return unique run identifier."""
        uniqid = []
        for run in self:
            uniqid.append('-'.join([str(run.get(field)) for field in self.fields]))
        return np.array(uniqid)

    def prune(self):
        """Remove duplocate runs in ``self``."""
        uniqid = self.uniqid()
        indices = np.unique(uniqid,return_index=True)[1]
        return self[indices]

    def in1d(self,other):
        """Return mask selecting runs that are in ``other``."""
        selfid,otherid = self.uniqid(),other.uniqid()
        return np.in1d(selfid,otherid)

    def __iter__(self):
        """Iterate through the different runs."""
        for run in super(RunCatalog,self).__iter__():
            run.kwargs_file = {key:run.get(key) for key in get_randoms_id.keys()}
            yield run

    def iter_mask(self, cat):
        """Yield boolean mask for the different runs in input catalog ``cat``."""
        for run in self:
            yield np.all([cat.get(field) == run.get(field) for field in self.fields],axis=0)

    def iter_index(self, cat):
        """Yield indices for the different runs in input catalog ``cat``."""
        for mask in self.iter_mask(cat):
            yield np.flatnonzero(mask)

    def count_runs(self, cat):
        """Return the number of runs in input catalog ``cat``."""
        return sum(mask.any() for mask in self.iter_mask(cat))

    def write_list(self, fn):
        """
        Write run list to ``fn``.

        Parameters
        ----------
        fn : string
            Path to run list.
        """
        utils.mkdir(os.path.dirname(fn))
        with open(fn,'w') as file:
            for run in self:
                file.write(run.brickname + ' ' + get_randoms_id(**vars(run)))

    @classmethod
    def from_list(cls, fns):
        """
        Build RunCatalog from brick list(s) in ``fns``.

        Parameters
        ----------
        fns : list, string
            Path to run list. If multiple paths are provided, they are concatenated.

        Returns
        -------
        self : RunCatalog
            New instance corresponding to the input run list(s).
        """
        self = cls()
        for field in self.fields: self.set(field,[])
        if np.isscalar(fns):
            fns = [fns]
        for fn in fns:
            with open(fn,'r') as file:
                for line in file:
                    brickname,tmp = line.split()
                    self.get('brickname').append(brickname)
                    tmp = get_randoms_id.split(tmp)
                    for key in tmp:
                        self.get(key).append(tmp[key])
        self.to_np_arrays()
        return self

class BaseAnalysis(object):

    """Class to load, merge and save **Obiwan** products."""

    def __init__(self, base_dir='.', runcat=None, bricknames=[], kwargs_files={}, cats_dir=None, save_fn=None):
        """
        Initialize ``BaseAnalysis`` by setting **Obiwan** file structure.

        Parameters
        ----------
        base_dir : string, default='.'
            **Obiwan** (if ``source == 'obiwan'``) or legacypipe (if ``source == 'legacypipe'``) root file directory.

        runcat : RunCatalog, defaut=None
            Run catalog used to select files from **Obiwan** or **legacypipe** data structure.
            If provided, superseeds ``bricknames`` and ``kwargs_files``.

        bricknames : list, default=[]
            List of bricknames.

        kwargs_files : list, default={}
            List of other arguments to file paths (e.g. ``get_randoms_id.keys()``).

        cats_dir : string, default=None
            Directory where to save merged catalogs.

        save_fn : string, default=None
            File name where to save ``self``.
        """
        self.base_dir = base_dir
        if runcat is None:
            self.runcat = RunCatalog.from_brick_ranid(bricknames=bricknames,kwargs_files=kwargs_files)
        else:
            self.runcat = runcat
        self.cats = {}
        self.cats_fn = {}
        self.cats_dir = cats_dir
        self.save_fn = save_fn

    @classmethod
    def get_key(cls, filetype='tractor', source='obiwan'):
        """Return key to catalog in internal dictionary ``self.cats``."""
        return '%s_%s' % (source,filetype.replace('-','_'))

    def merge_catalogs(self, filetype='tractor', base_dir=None, source='obiwan', keep_columns=None, add=False, write=False, **kwargs_write):
        """
        Merge catalogs, return the result and add to ``self`` (``add == True``) and/or directly write on disk (``write == True``).

        Parameters
        ----------
        filetype : string, default='tractor'
            Type of file to merge.

        base_dir : string, default=None
            **Obiwan** (if ``source == 'obiwan'``) or legacypipe (if ``source == 'legacypipe'``) root file directory.
            Superseeds ``self.base_dir``.

        source : string, default='obiwan'
            If 'obiwan', search for an **Obiwan** output file name, else a **legacypipe** file name.

        keep_columns : list, default=None
            Keep only these columns.

        add : bool, default=False
            Add merged catalog to ``self``.

        write : bool, default=False
            Write merged catalog to disk.

        kwargs_write : bool, default={}
            If ``write``, arguments to pick up catalog file name. See ``self.set_cat_fn()``

        Returns
        -------
        cat : SimCatalog
            Merged catalog.
        """
        if base_dir is None: base_dir = self.base_dir
        if (not write) and (not add):
            logger.warning('Nothing to be done with loaded files. Escaping.')
            return
        cat = 0
        def read_catalog(fn,**kwargs):
            if os.path.isfile(fn):
                return SimCatalog(fn,**kwargs)
            logger.warning('File %s not found.' % fn)
            return None
        for run in self.runcat:
            if filetype in ['ps','ps-events']:
                fn = find_file(base_dir=base_dir,filetype='ps',brickname=run.brickname,source='obiwan',**run.kwargs_file)
                tmp = read_catalog(fn,ext=1)
                if tmp is None: continue
                tf = tmp.unixtime.max()
                events = read_catalog(fn,ext=2)
                ti = events.unixtime.min()
                if filetype == 'ps-events':
                    tmp = events
                else:
                    tmp.mid = tmp.full(tmp._header['PPID'])
                tmp.unixti = tmp.full(ti)
                tmp.unixtf = tmp.full(tf)
                tmp.brickname = tmp.full(run.brickname)
            else:
                fn = find_file(base_dir=base_dir,filetype=filetype,brickname=run.brickname,source=source,**run.kwargs_file)
                tmp = read_catalog(fn)
                if tmp is None: continue
            for key,val in run.kwargs_file.items():
                tmp.set(key,tmp.full(val))
            if filetype == 'randoms':
                tmp.cut(~tmp.collided)
            if keep_columns is not None:
                tmp.keep_columns(*keep_columns)
            cat += tmp
        key = self.get_key(filetype=filetype,source=source)
        if write:
            self.write_catalog(cat=cat,**kwargs_write)
        if add:
            self.cats[key] = cat
        return cat

    def set_cat_fn(self, cat_base=None, cat_fn=None, **kwargs_key):
        """
        Set catalog file name.

        Parameters
        ----------
        cat_base : string, default=None
            Catalog base name, added to `self.cats_dir``.

        cat_fn : string, default=None
            Catalog full name. If provided, superseeds ``cat_base``.

        kwargs_key : bool, default={}
            Arguments to ``self.get_key()``.

        Returns
        -------
        key : string
            Key to catalog file name in internal ``self.cats_fn`` dictionary.
        """
        key = self.get_key(**kwargs_key)
        if cat_fn is None:
            if cat_base is not None:
                cat_fn = os.path.join(self.cats_dir,cat_base)
        if cat_fn is not None:
            self.cats_fn[key] = cat_fn
        return key

    def write_catalog(self, cat=None, **kwargs):
        """
        Write catalog to disk.

        Parameters
        ----------
        cat : SimCatalog, default=None
            Catalog to save. If ``None``, is got from internal dictionary ``self.cats``.

        kwargs : dict
            Arguments for ``self.set_cat_fn()``, to set catalog file name.
        """
        key = self.set_cat_fn(**kwargs)
        if cat is None: cat = self.cats[key]
        cat.writeto(self.cats_fn[key])

    def read_catalog(self, add=False, **kwargs):
        """
        Read catalog from disk.

        Parameters
        ----------
        add : bool, default=False
            Add catalog to ``self``.

        kwargs : dict
            Arguments for ``self.set_cat_fn()``, to set catalog file name.
        """
        key = self.set_cat_fn(**kwargs)
        cat = SimCatalog(self.cats_fn[key])
        if add: self.cats[key] = cat
        return cat

    def get(self, *args, **kwargs):
        """Return ``self`` attribute."""
        return getattr(self,*args,**kwargs)

    def set(self, *args, **kwargs):
        """Set ``self`` attribute."""
        return setattr(self,*args,**kwargs)

    def has(self, *args, **kwargs):
        """Has ``self`` this attribute?"""
        return hasattr(self,*args,**kwargs)

    def getstate(self):
        """Export ``self`` to ``dict``."""
        state = {}
        for key in ['base_dir','cats_fn','cats_dir']:
            state[key] = self.get(key)
        state['runcat'] = self.runcat.to_dict()
        return state

    def setstate(self, state):
        """Add ``state`` to ``self``."""
        self.cats = {}
        for key in state:
            self.set(key,state[key])
        self.runcat = RunCatalog.from_dict(self.runcat)

    def save(self, save_fn=None):
        """
        Save ``self`` to disk.

        Parameters
        ----------
        save_fn : string, default=None
            File name where to save ``self``.
            If not ``None``, superseeds ``self.save_fn``.
        """
        if save_fn is not None: self.save_fn = save_fn
        logger.info('Saving %s to %s.' % (self.__class__.__name__,self.save_fn))
        utils.mkdir(os.path.dirname(self.save_fn))
        np.save(self.save_fn,self.getstate())

    @classmethod
    def load(cls, save_fn):
        """
        Load ``self`` from disk.

        Parameters
        ----------
        save_fn : string
            File name where ``self`` is saved.
        """
        state = np.load(save_fn,allow_pickle=True)[()]
        self = object.__new__(cls)
        self.setstate(state)
        return self

    def set_catalog(self, name, filetype=None, source=None, **kwargs_merge):
        """
        Convenience function to add catalog to ``self`` with name ``name``.

        First try to get it from internal dictionary ``self.cats``.
        Else if the file name is specified in ``self.cats_fn``, read the merged catalog from disk.
        Else merge catalogs.

        Parameters
        ----------
        name : string
            Name of the catalog to be added to ``self``.

        filetype : string, default=None
            Type of file to search for.

        source : string, default='obiwan'
            If 'obiwan', search for an **Obiwan** output file name, else a **legacypipe** file name.

        kwargs_merge : dict
            Extra arguments for ``self.merge_catalogs()``.
        """
        key = self.get_key(filetype=filetype,source=source)
        if key in self.cats:
            pass
        elif key in self.cats_fn:
            self.read_catalog(filetype=filetype,source=source,add=True)
        else:
            self.merge_catalogs(filetype=filetype,source=source,add=True,**kwargs_merge)
        self.set(name,self.cats[key])

class RessourceEventAnalysis(BaseAnalysis):

    """Class that analyses computing time based on events saved in ``ps`` files."""

    sorted_events = ['start', 'stage_tims: starting', 'stage_tims: starting calibs', 'stage_tims: starting read_tims', 'stage_tims: done read_tims',
        'stage_refs: starting', 'stage_outliers: starting', 'stage_halos: starting', 'stage_srcs: starting', 'stage_srcs: detection maps', 'stage_srcs: sources', 'stage_srcs: SED-matched',
        'stage_fitblobs: starting', 'stage_coadds: starting', 'stage_coadds: model images', 'stage_coadds: coadds', 'stage_coadds: extras', 'stage_writecat: starting']

    def process_events(self, events=None, time='reltime', statistic='mean'):
        """
        Return statistics about event durations.
        Assumes event are recorded at the beginning of the corresponding step.

        Parameters
        ----------
        events : array-like, string, default=None
            List of events to get statistics from, e.g. ['stage_halos: starting'].
            See ``self.sorted_events``.
            If ``None``, consider all events.
            If ``stage``, consider only starting stage events ('stage_xxx: starting').

        time : string, default='reltime'
            Time to consider for calculation.
            If 'reltime', consider the time at which events happens relative to the beginning of the run.
            Else, if 'steptime', consider the time duration of each event, i.e. the time difference between events,
            starting from the last time step registered for the run.

        statistic : string, callable, default='mean'
            Statistic to compute from event time, passed on to ``scipy.stats.binned_statistic``.

        Returns
        -------
        events, values : ndarray
            Events and corresponding statistic
        """
        self.set_catalog(name='events',filetype='ps-events')

        def get_sorted_events():
            sorted_events = np.array(self.sorted_events)
            return sorted_events[np.in1d(sorted_events,self.events.event)]

        if events is None:
            events = get_sorted_events()
        elif isinstance(events,str) and events == 'stage':
            uniques = get_sorted_events()
            base = uniques[0].split(':')[0]
            events_sel = [uniques[0]]
            for event in uniques:
                if event.startswith(base): continue
                base = event.split(':')[0]
                events_sel.append(event)
            events = np.asarray(events_sel)
        if time == 'reltime':
            indices = self.events.event
            values = self.events.unixtime - self.events.unixti
        elif time == 'steptime':
            indices = self.events.event
            values = self.events.nans()
            for mask in self.runcat.iter_mask(self.events):
                events_run = self.events[mask]
                dt = np.zeros(events_run.size,dtype='f4')
                tf = events_run.unixtf[0]
                for event in events[::-1]:
                    mask_event = events_run.event == event
                    if mask_event.any():
                        tmp = events_run.unixtime[mask_event]
                        dt[mask_event] = tf - tmp
                        tf = tmp[0]
                values[mask] = dt
        else:
            raise ValueError('Unknown requested time %s' % time)

        uniques,indices,inverses = np.unique(indices,return_index=True,return_inverse=True)
        uniqid = np.arange(len(uniques))
        edges = np.concatenate([uniqid,[uniqid[-1]+1]])
        values = stats.binned_statistic(uniqid[inverses],values,statistic=statistic,bins=edges)[0]
        values = values[np.searchsorted(uniques,events)]

        return events,values

    def plot_bar(self, ax, events='stage', label_entries=True, kwargs_bar={}):
        """
        Plot event mean and standard deviation in the form of a bar graph.

        Parameters
        ----------
        ax : plt.axes
            Where to plot image.
            One can also provide figure file name ``fn``.
            See ``utils.saveplot``.

        events : array-like, string, default=None
            Passed on to ``self.process_events()``.

        label_entries : bool, default=True
            If ``True``, add the number of entries to the plot.

        kwargs_bar : dict, default={}
            Extra arguments for ``plt.bar()``.
        """
        kwargs_bar = {**{'align':'center','alpha':0.5,'ecolor':'black','capsize':2},**kwargs_bar}
        names,mean = self.process_events(events=events,time='steptime',statistic='mean')
        std = self.process_events(events=names,time='steptime',statistic='std')[1]
        labels = []
        for name in names:
            label = name
            if events in ['start']: label = name.split(':')[0]
            labels.append(label.replace('stage_',''))
        labels = np.array(labels)
        ax.yaxis.grid(True)
        ax.bar(labels,mean,yerr=std,**kwargs_bar)
        ax.set_xticklabels(labels=labels,rotation=40,ha='right')
        ax.set_ylabel('Average wall time [s]')
        if label_entries:
            label = '%d entries' % self.runcat.count_runs(self.events)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')

class RessourceAnalysis(RessourceEventAnalysis):

    """Class that extends events-only analyses of ``RessourceEventAnalysis`` with analysis of full time series saved in ``ps`` files."""

    @staticmethod
    def process_one_series(series, quantities=['proc_icpu','vsz']):
        """
        Compute ressources for a single time series, split between the following categories:
        'ps' (the process running `ps`), 'main' (the master process),
        'workers' (non-zero in case of multithreading), 'others' (other processes).

        Parameters
        ----------
        series : SimCatalog
            Time series to analyze.

        quantities : list, default=['proc_icpu','vsz']
            Quantities to analyze.

        Returns
        -------
        toret : dict
            A dictionary holding time series for each quantity and process category.
        """
        steps,index_steps = np.unique(series.step, return_index=True)
        map_steps = np.zeros(steps.max()+1,dtype='i4')
        map_steps[steps] = np.arange(len(steps))
        reltime = series.unixtime[index_steps] - series.unixti[index_steps]
        mid = series.mid[0]

        toret = {'time':reltime}
        for q in quantities:
            toret[q] = {}
            toret[q]['ps'] = np.zeros(len(steps),dtype='f4')
            toret[q]['others'] = np.zeros(len(steps),dtype='f4')
            toret[q]['main'] = np.zeros(len(steps),dtype='f4')
            toret[q]['workers'] = []

        mask_main = series.pid == mid
        mask_mine = mask_main | (series.ppid == mid)
        pids = np.unique(series.pid[mask_mine])

        for pid in pids:
            index_pid = np.flatnonzero(series.pid == pid)
            step_pid = map_steps[series.step[index_pid]]
            cmds = np.unique(series.command[index_pid])
            if len(cmds) == 1 and cmds[0].startswith('ps ax'):
                for q in quantities:
                    toret[q]['ps'][step_pid] += series.get(q)[index_pid]
            else:
                for q in quantities:
                    tmp = np.zeros(len(steps),dtype='f4')
                    tmp[step_pid] = series.get(q)[index_pid]
                    if pid == mid: toret[q]['main'] = tmp
                    else: toret[q]['workers'].append(tmp)
        for q in quantities: toret[q]['workers'] = np.array(toret[q]['workers'])

        pids = np.unique(series.pid[~mask_mine])
        for pid in pids:
            index_pid = np.flatnonzero(series.pid == pid)
            step_pid = map_steps[series.step[index_pid]]
            for q in quantities:
                toret[q]['others'][step_pid] += series.get(q)[index_pid]

        return toret

    def process_all_series(self, quantities=['proc_icpu','vsz']):
        """
        Compute average ressources for all time series, split between the following categories:
        'ps' (the process running `ps`), 'main' (the master process),
        'workers' (non-zero in case of multithreading), 'others' (other processes).

        Parameters
        ----------
        quantities : list, default=['proc_icpu','vsz']
            Quantities to analyze.

        Returns
        -------
        toret : dict
            A dictionary holding time series for each quantity and process category.
        """
        self.set_catalog(name='series',filetype='ps')
        toret = {q:{} for q in quantities}
        toret['time'] = []
        for mask in self.runcat.iter_mask(self.series):
            series = self.series[mask]
            qseries = self.process_one_series(series,quantities=quantities)
            for q in quantities:
                for key,val in qseries[q].items():
                    if key not in toret[q]: toret[q][key] = []
                    if key == 'workers':
                        val = val.max(axis=0)
                    toret[q][key].append(val)
            toret['time'].append(qseries['time'])
        time_range = (0.,max(time.max() for time in toret['time']))
        num = max(len(time) for time in toret['time'])
        time = np.linspace(*time_range,num=num)
        for q in quantities:
            for key,val in toret[q].items():
                toret[q][key] = np.mean([np.interp(time,t,v) for t,v in zip(toret['time'],val)],axis=0)
        toret['time'] = time
        return toret

    def add_event_vlines(self, ax, events=None):
        """
        Plot vertical lines to indicate start of events.

        Parameters
        ----------
        ax : plt.axes
            Where to plot events.

        events : array-like, string, default=None
            Passed on to ``self.process_events()``.
        """
        names,times = self.process_events(events=events)
        for ievent,(name,time) in enumerate(zip(names,times)):
            ax.axvline(time,color='k',alpha=0.1)
            label = name
            if events == 'stage': label = name.split(':')[0]
            label = label.replace('stage_','')
            ycoord = [0.05,0.35,0.65][ievent % 3]
            ax.text(time,ycoord,label,rotation='vertical',horizontalalignment='left',
                    verticalalignment='bottom',transform=ax.get_xaxis_transform(),color='k')

    @utils.saveplot()
    def plot_one_series(self, ax, series=None, events='stage', processes=['main','workers'], kwargs_fig={'figsize':(10,5)}, kwargs_plot={}):
        """
        Plot ressources for a single time series.

        Parameters
        ----------
        ax : plt.axes
            Where to plot the time series.
            One can also provide figure file name ``fn``.
            See ``utils.saveplot``.

        series : SimCatalog, default=None
            Time series to analyze.
            If ``None``, defaults to ``self.series``.

        events : array-like, string, default='stage'
            Passed on to ``self.process_events()``.

        processes : list, default=['main','workers']
            Processes to plot.

        kwargs_fig : dict, default={}
            Extra arguments for ``plt.savefig()``.
            See ``utils.saveplot()``.

        kwargs_plot : dict, default={}
            Extra arguments for ``plt.plot()``.
        """
        if series is None:
            self.set_catalog(name='series',filetype='ps')
            series = self.series
        series = self.process_one_series(series=series,quantities=['proc_icpu','vsz'])
        if processes is None: processes = list(series['proc_icpu'].keys())
        colors = kwargs_plot.pop('colors',plt.rcParams['axes.prop_cycle'].by_key()['color'])
        alpha = kwargs_plot.pop('alpha',1.)
        ax2 = ax.twinx()
        for key,color in zip(processes,colors):
            if key in ['workers']:
                alpha_ = 0.25*alpha
                cpu,vsz = series['proc_icpu'][key],series['vsz'][key]
            else:
                alpha_ = alpha
                cpu,vsz = [series['proc_icpu'][key]],[series['vsz'][key]]
            for iwork,(cpu_,vsz_) in enumerate(zip(cpu,vsz)):
                ax.plot(series['time'],cpu_,label=key if iwork==0 else None,color=color,alpha=alpha_,**kwargs_plot)
                ax2.plot(series['time'],vsz_/1e6,color=color,alpha=alpha_,linestyle='--',**kwargs_plot)
        ax.set_xlabel('Wall time [s]')
        ax.set_ylabel('Proc [%]')
        ax2.set_ylabel('VSS [GB]')
        self.add_event_vlines(ax,events=events)
        ax.legend()

    @utils.saveplot()
    def plot_all_series(self, ax, events='stage', processes=['main','workers'], label_entries=True, kwargs_plot={}):
        """
        Plot summary ressources for all time series.

        Parameters
        ----------
        ax : plt.axes
            Where to plot the time series.
            One can also provide figure file name ``fn``.
            See ``utils.saveplot``.

        events : array-like, string, default='stage'
            Passed on to ``self.process_events()``.

        processes : list, default=['main','workers']
            Processes to plot.

        label_entries : bool, default=True
            If ``True``, add number of entries to plot.

        kwargs_plot : dict, default={}
            Extra arguments for ``plt.plot()``.
        """
        series = self.process_all_series(quantities=['proc_icpu','vsz'])
        if processes is None: processes = list(series['proc_icpu'].keys())
        colors = kwargs_plot.pop('colors',plt.rcParams['axes.prop_cycle'].by_key()['color'])
        ax2 = ax.twinx()
        for key,color in zip(processes,colors):
            ax.plot(series['time'],series['proc_icpu'][key],label=key,color=color,**kwargs_plot)
            ax2.plot(series['time'],series['vsz'][key]/1e6,color=color,linestyle='--',**kwargs_plot)
        ax.set_xlabel('Wall time [s]')
        ax.set_ylabel('Proc [%]')
        ax2.set_ylabel('VSS [GB]')
        self.add_event_vlines(ax,events=events)
        ax.legend()
        if label_entries:
            label = '%d entries' % self.runcat.count_runs(self.events)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')

class MatchAnalysis(BaseAnalysis):

    def getstate(self):
        """Export ``self`` to ``dict``."""
        state = super(MatchAnalysis,self).getstate()
        for key in ['add_input_tractor','injected','observed','distance']:
            if self.has(key):
                state[key] = self.get(key)
        for put in ['input','output']:
            for template in ['inter_%s','extra_%s','inter_%s_injected']:
                key = template % put
                if self.has(key):
                    state[key] = self.get(key)
        return state

    def setstate(self, state):
        """Add ``state`` to ``self``."""
        super(MatchAnalysis,self).setstate(state)
        if self.has('add_input_tractor'):
            self.setup(add_input_tractor=state['add_input_tractor'])

    def setup(self, add_input_tractor=False):
        """
        Add ``input`` and ``output`` **Tractor** catalog to ``self``.

        By default, the injected sources of **Obiwan** randoms only are considered for ``input``.
        These can be merged to the sources fitted by **legacypipe** by setting ``add_input_tractor``.

        Parameters
        ----------
        add_input_tractor : bool, string, default=False
            If ``True``, **legacypipe** **Tractor** catalogs are added to ``input``.
            In this case ``self.base_dir`` is considered as the **legacyipe** root directory.
            If ``string``, specifies the path to the **legacyipe** root directory.

        """
        self.add_input_tractor = add_input_tractor
        self.set_catalog(name='input',filetype='randoms',source='obiwan')
        self.set_catalog(name='output',filetype='tractor',source='obiwan')
        if add_input_tractor:
            kwargs = {}
            if isinstance(add_input_tractor,str): kwargs = {'base_dir':add_input_tractor}
            self.set_catalog(name='input_tractor',filetype='tractor',source='legacypipe',**kwargs)
        self.injected = self.input.index()
        self.observed = np.array([],dtype=int)
        if add_input_tractor:
            self.observed = np.arange(self.input.size,self.input.size+self.input_tractor.size)
            self.input.fill(self.input_tractor,index_self='after')

    def match(self, radius_in_degree=1.5/3600., add_input_tractor=False):
        """
        Match ``self.input`` to ``self.output``.

        Parameters
        ----------
        radius_in_degree : float, default=1.5/3600.
            Radius (degree) for input - output matching.

        add_input_tractor : bool, string, default=False
            Passed on to ``self.setup()``.
        """
        self.setup(add_input_tractor=add_input_tractor)
        self.inter_input,self.inter_output,self.distance = [],[],[]
        index_input,index_output = self.input.index(),self.output.index()
        for mask_input,mask_output in zip(self.runcat.iter_mask(self.input),self.runcat.iter_mask(self.output)):
            mask_input[self.observed] = self.input.brickname[self.observed] == self.input.brickname[self.injected][0]
            inter_input,inter_output,distance = self.input[mask_input].match_radec(self.output[mask_output],nearest=True,radius_in_degree=radius_in_degree,return_distance=True)
            self.inter_input.append(index_input[mask_input][inter_input])
            self.inter_output.append(index_output[mask_output][inter_output])
            self.distance.append(distance)
        for key in ['inter_input','inter_output','distance']:
            self.set(key,np.concatenate(self.get(key)))

        logger.info('Matching %d objects / %d in input, %d in output' % (self.inter_input.size,self.input.size,self.output.size))
        mask_injected = np.in1d(self.inter_input,self.injected)
        for key in ['input','output']:
            self.set('extra_%s' % key,np.setdiff1d(self.get(key).index(),self.get('inter_%s' % key)))
            self.set('inter_%s_injected' % key,self.get('inter_%s' % key)[mask_injected])
        self.distance_injected = self.distance[mask_injected]
        logger.info('Matching %d injected objects / %d in input, %d in output' % (self.inter_input_injected.size,self.injected.size,self.output.size))

    def export(self, base='input', key_input='input', key_output=None, key_distance='distance', key_matched='matched', key_injected='injected', injected=False, write=False, **kwargs_write):
        """
        Export the matched catalog obtained with ``self.match()``.

        Parameters
        ----------
        base : string, default='input'
            Base for matched catalog. Can be:
            - 'input': output columns are added to input
            - 'output': input columns are added to output
            - 'inter': as 'input', but matched sources only are kept
            - 'extra': keep only sources which could not be matched, in either input or output
            - 'all': stack all sources in both input and output

        key_input : string, default='input'
            If not ``None``, prefix to add to input fields.

        key_output : string, default=None
            If not ``None``, prefix to add to output fields.

        key_distance : string, default='distance'
            If not ``None``, field to save angular distance between matched input and output sources.

        key_matched : string, default='matched'
            If not ``None``, field to save boolean mask identifying matched input and output sources.

        key_injected : string, default='injected'
            If not ``None``, field to save boolean mask identifying injected sources.

        injected : bool, default=False
            If ``True``, restrict to injected sources.

        write : bool, default=False
            Write catalog to disk.

        kwargs_write : dict
            Arguments for ``self.write_catalog()``.

        Returns
        -------
        cat : SimCatalog
            Catalog of input - output sources.
        """
        full_input = self.input.copy()
        if key_input:
            for field in full_input.fields: full_input.rename(field,'%s_%s' % (key_input,field))
        output = self.output.copy()
        if key_output:
            for field in output.fields: output.rename(field,'%s_%s' % (key_output,field))
        if key_distance is not None:
            for key,cat in zip(['input','output'],[full_input,output]):
                distance = cat.nans()
                distance[self.get('inter_%s' % key)] = self.distance
                cat.set(key_distance,distance)
        if key_matched is not None:
            for key,cat in zip(['input','output'],[full_input,output]):
                match = cat.falses()
                match[self.get('inter_%s' % key)] = True
                cat.set(key_matched,match)

        inter_input,inter_output = self.inter_input,self.inter_output
        extra_input,extra_output = self.extra_input,self.extra_output
        if injected:
            inter_input,inter_output = self.inter_input_injected,self.inter_output_injected
        if key_injected:
            mask_injected = full_input.falses()
            mask_injected[self.injected] = True
            full_input.set(key_injected,mask_injected)
        if base == 'input':
            cat = full_input
            cat.fill(output,index_self=inter_input,index_other=inter_output)
        elif base == 'output':
            cat = output
            cat.fill(full_input,index_self=inter_output,index_other=inter_input)
        elif base == 'inter':
            cat = full_input[inter_input]
            cat.fill(output,index_self=None,index_other=inter_output)
        elif base == 'extra':
            cat = full_input[extra_input]
            cat.fill(output,index_self='after',index_other=extra_output)
        elif base == 'all':
            cat = full_input[self.injected] if injected else full_input
            cat.fill(output,index_self='after')
        if write:
            self.write_catalog(cat=cat,filetype='match_%s' % base,source='obiwan',**kwargs_write)
        return cat

    @utils.saveplot()
    def plot_scatter(self, ax, field, injected=True, xlabel=None, ylabel=None,
                    square=False, regression=False, diagonal=False, label_entries=True,
                    kwargs_xlim={}, kwargs_ylim={}, kwargs_scatter={}, kwargs_regression={}, kwargs_diagonal={}):
        """
        Scatter plot output v.s. input.

        Parameters
        ----------
        ax : plt.axes
            Where to plot.

        field : string
            Name of catalog column to plot.

        injected : bool, default=True
            If ``True``, restrict to injected sources. Else all (actual and injected) sources are considered.

        xlabel : string, default=None
            x-label, if ``None``, defaults to ``'input_%s' % field``.

        ylabel : string, default=None
            y-label, if ``None``, defaults to ``'output_%s' % field``.

        square : bool, default=False
            If ``True``, enforce square plot.

        regression : bool, default=False
            If ``False``, plot regression line.

        diagonal : bool, default=False
            If ``False``, plot diagonal line.

        label_entries : bool, default=True
            If ``True``, add the number of entries to the plot.

        kwargs_xlim : dict, default={}
            Arguments for ``Binning``, to define the x-range.

        kwargs_ylim : dict, default={}
            Arguments for ``Binning``, to define the y-range.

        kwargs_scatter : dict, default={}
            Extra arguments for ``plt.scatter()``.

        kwargs_regression : dict, default={}
            Extra arguments for ``plt.plot()`` regression line.

        kwargs_diagonal : dict, default={}
            Extra arguments for ``plt.plot()`` diagonal line.
        """
        kwargs_scatter = {**{'s':10,'marker':'.','alpha':1,'edgecolors':'none'},**kwargs_scatter}
        kwargs_diagonal = {**{'linestyle':'--','linewidth':2,'color':'k'},**kwargs_diagonal}
        kwargs_regression = {**{'linestyle':'--','linewidth':2,'color':'r','label':''},**kwargs_regression}
        label_regression = kwargs_regression.pop('label',None)

        if injected:
            inter_input,inter_output = self.inter_input_injected,self.inter_output_injected
        else:
            inter_input,inter_output = self.inter_input,self.inter_output

        values1 = self.input.get(field)[inter_input]
        values2 = self.output.get(field)[inter_output]

        if xlabel is None: xlabel = 'input_%s' % field
        if ylabel is None: ylabel = 'output_%s' % field

        if kwargs_xlim:
            xlim = Binning(samples=values1,nbins=1,**kwargs_xlim).range
            ax.set_xlim(xlim)

        if kwargs_ylim:
            ylim = Binning(samples=values2,nbins=1,**kwargs_ylim).range
            ax.set_ylim(ylim)

        ax.scatter(values1,values2,**kwargs_scatter)

        if square:
            xlim,ylim = ax.get_xlim(),ax.get_ylim()
            xylim = min(xlim[0],ylim[0]),max(xlim[1],ylim[1])
            ax.axis('square')
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)

        xlim,ylim = [np.array(tmp) for tmp in [ax.get_xlim(),ax.get_ylim()]]
        try:
            a,b = np.polyfit(values1,values2,1)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError('Regression failed, %s x-range = %.3f - %.3f, y-range = %.3f - %.3f'
                                        % (field,values1.min(),values1.max(),values2.min(),values2.max()))
        y = a*xlim + b
        r = np.corrcoef(values1,values2)[0,1]

        label_regression_ = label_regression
        if label_regression_ is not None:
            label_regression = '$\\rho = %.3f$' % r
            if label_regression_:
                label_regression = '%s %s' % (label_regression_,label_regression)
        else: label_regression = None

        if regression:
            ax.plot(xlim,y,label=label_regression_,**kwargs_regression)
            ax.set_ylim(ylim)
        elif label_regression:
            ax.text(0.95,0.05,label_regression,horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes,color='k')
        if diagonal:
            ax.plot(xlim,xlim,**kwargs_diagonal)
            ax.set_ylim(ylim)
        if label_entries:
            label = '%d entries' % len(values1)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @utils.saveplot()
    def plot_hist(self, ax, field, injected=True, xlabel=None, ylabel=None, divide_uncer=True,
                label_entries=True, label_mean_std=True, kwargs_xedges={}, kwargs_hist={}):
        """
        Histogram of output v.s. input differences.

        Parameters
        ----------
        ax : plt.axes
            Where to plot.

        field : string
            Name of catalog column to plot.

        injected : bool, default=True
            If ``True``, restrict to injected sources. Else all (actual and injected) sources are considered.

        xlabel : string, default=None
            x-label, if ``None``, defaults to ``'input_%s' % field``.

        ylabel : string, default=None
            y-label, if ``None``, defaults to ``'output_%s' % field``.

        divide_uncer : bool, default=True
            If ``True``, divide the difference ``output - input`` by the estimated uncertainty.

        label_entries : bool, default=True
            If ``True``, add the number of entries to the plot.

        label_mean_std : bool, default=True
            If ``True``, add the estimated mean, median, standard deviation to the plot.

        kwargs_xedges : dict, default={}
            Arguments for ``Binning``, to define the x-edges.

        kwargs_hist : dict, default={}
            Extra arguments for ``plt.hist()``.
        """
        if injected:
            inter_input,inter_output = self.inter_input_injected,self.inter_output_injected
        else:
            inter_input,inter_output = self.inter_input,self.inter_output
        values = self.output.get(field)[inter_output]-self.input.get(field)[inter_input]
        if divide_uncer:
            if 'flux' in field: field_ivar = field.replace('flux','flux_ivar')
            else: field_ivar = '%s_ivar' % field
            ivar = self.output.get(field_ivar)[inter_output]
            values = values*np.sqrt(ivar)
        if xlabel is None:
            xlabel = '\\Delta \\mathrm{%s}' % field.replace('_','\_')
            if divide_uncer: xlabel = '$%s \\times \\sqrt{\\mathrm{%s}}$' % (xlabel,field_ivar.replace('_','\_'))
            else: xlabel = '$%s$' % xlabel
        kwargs_hist = {**{'histtype':'step','color':'k'},**kwargs_hist}
        edges = Binning(samples=values,**kwargs_xedges).edges
        ax.hist(values,bins=edges,**kwargs_hist)
        ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if label_entries:
            label = '%d entries' % len(values)
            ax.text(0.05,0.95,label,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k')
        if label_mean_std:
            label = '$\mathrm{median} = %.2g$\n' % np.median(values)
            label += '$\mathrm{mean} = %.2g$\n' % np.mean(values)
            label +='$\mathrm{std} = %.2g$\n' % np.std(values,ddof=1)
            label +='$\mathrm{std(med)} = %.2g$' % estimate_std_outliers(values)
            ax.text(0.95,0.95,label,horizontalalignment='right',verticalalignment='top',transform=ax.transAxes,color='k')

"""Utilities."""

class Binning(object):

    """Class that creates edges to bin samples."""

    def __init__(self, samples=None, weights=None, edges=None, nbins=10, range=None, quantiles=None, scale='linear'):
        """
        Initialise ``Binning``.

        First set the total range, then the number of bins with ``np.histogram_bin_edges()``.

        Parameters
        ----------
        samples : array-like
            Samples to be binned.

        weights : array-like, default=None
            Weights associated to ``samples``, used by ``np.histogram_bin_edges()``.
            If ``None``, defaults to 1.

        edges : array-like, default=None
            If edges already provided, nothing to do!

        nbins : int, string, default=None
            Number of bins.
            If ``int``, used to define ``edges`` using ``scale``.
            Else, passed on to ``np.histogram_bin_edges()``.

        range : array-like, default=None
            Range, i.e. ``edges`` minimum and maximum boundaries. If not ``None``, superseeds ``quantiles``.

        quantiles : array-like, default=None
            Quantiles (low,up) to define ``range``.
            If ``None``, set ``range`` to ``samples`` minimum and maximum.

        scale : string, default='linear'
            Used if ``nbins`` is ``int``.
            If 'linear', use a linear binning.
            If 'log', use a logarithmic binning.
        """
        self.edges = edges
        if edges is None:
            if range is None:
                if quantiles is None:
                    range = [samples.min(axis=-1),samples.max(axis=-1)]
                    range[-1] = range[-1]+(range[-1]-range[0])*1e-5
                else:
                    range = np.percentile(samples,q=np.array(quantiles)*100.,axis=-1).T
            if range[0] is None: range[0] = samples.min(-1)
            if range[-1] is None:
                range[-1] = samples.max(-1)
                range[-1] = range[-1]+(range[-1]-range[0])*1e-5
            if isinstance(nbins,np.integer):
                if scale == 'linear':
                    self.edges = np.linspace(range[0],range[-1],nbins+1)
                elif scale == 'log':
                    self.edges = np.logspace(np.log10(range[0]),np.log10(range[-1]),nbins+1,base=10)
                else:
                    raise ValueError('Scale %s is unkown.' % scale)
            else:
                self.edges = np.histogram_bin_edges(samples,bins=nbins,range=range,weights=weights)

    @property
    def range(self):
        """Return ``edges`` minimum and maximum boundaries."""
        return (self.edges[0],self.edges[-1])

    @property
    def nbins(self):
        """Return number of bins."""
        return len(self.edges)-1

    @property
    def centers(self):
        """Return bin centers."""
        return (self.edges[:-1]+self.edges[1:])/2.


def estimate_std_outliers(samples):
    """
    Estimate the standard deviation of ``samples`` from the median of the folded distribution.

    Less sensitive to outliers than standard deviation.
    """
    return np.median(np.abs(samples-np.median(samples)))/(2.**0.5*special.erfinv(1./2.))
