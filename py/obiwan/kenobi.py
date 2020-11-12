# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
'''
Gathers classes to overload legacypipe.
'''

import os
import sys
import logging
import numpy as np
from legacypipe.decam import DecamImage
from legacypipe.bok import BokImage
from legacypipe.mosaic import MosaicImage
from legacypipe.survey import LegacySurveyData
from legacypipe.runs import DecamSurvey, NinetyPrimeMosaic
from legacypipe.runcosmos import DecamImagePlusNoise, CosmosSurvey
from astrometry.util.ttime import Time
import tractor
import galsim

logger = logging.getLogger('obiwan.kenobi')

class LegacySurveySim(LegacySurveyData):
    '''
    Same behavior as legacypipe.survey.LegacySurveyData,
    but this class also stores all the relevant obiwan objects.
    '''

    def __init__(self, *args, simcat=None, sim_stamp='tractor', add_sim_noise=False,
                 image_eq_model=False, seed=0, kwargs_file={}, **kwargs):
        '''
        Creates a LegacySurveySim object. kwargs are to be passed on to LegacySurveyData,
        other arguments are specific to LegacySurveySim.
        Note that only `survey_dir` must be specified to obtain bricks through self.get_brick_by_name(brickname).

        Parameters
        ----------
        simcat : SimCatalog, default=None
            Simulated source catalog for a given brick (not CCD).

        sim_stamp : string, default='tractor'
            Method to simulate objects, either 'tractor' (TractorSimStamp) or 'galsim' (GalSimStamp).

        add_sim_noise : bool, default=False
            Add Poisson noise from the simulated source to the image.

        image_eq_model : bool, default=False
            Wherever add a simulated source, replace both image and invvar of the image
            with that of the simulated source only.

        seed : int, default=0
            For random number generators.

        kwargs_file : dict, default={}
            Used to specify output paths, see LegacySurveySim.find_file().

        kwargs : dict, default={}
            Arguments for legacypipe.survey.LegacySurveyData.
        '''
        super(LegacySurveySim, self).__init__(*args,**kwargs)
        self.update_sim(simcat=simcat,sim_stamp=sim_stamp,add_sim_noise=add_sim_noise,
                        image_eq_model=image_eq_model,seed=seed,kwargs_file=kwargs_file)

    def update_sim(self,**kwargs):
        self.image_typemap = {
            'decam' : DecamSimImage,
            'decam+noise' : DecamSimImagePlusNoise,
            'mosaic' : MosaicSimImage,
            '90prime' : BokSimImage,
            }
        for key,val in kwargs.items():
            setattr(self,key,val)
        self.rng = np.random.RandomState(self.seed)
        #for key in kwargs:
        #    logger.info('%s = %s' % (key,getattr(self,key)))

    @classmethod
    def from_data(cls, data, **kwargs):
        self = object.__new__(cls)
        self.__dict__.update(data.__dict__)
        self.update_sim(**kwargs)


    def find_file(self, filetype, brick=None, output=False, **kwargs):
        '''
        Returns the filename of a Legacy Survey file.

        Parameters
        ----------
        filetype : string
            Type of file to find, including:
            `obiwan-randoms` -- Obiwan catalogues
            `obiwan-metadata` -- Obiwan metadata
            `tractor` -- Tractor catalogs
            `depth`   -- PSF depth maps
            `galdepth` -- Canonical galaxy depth maps
            `nexp` -- number-of-exposure maps.

        output : bool
            Whether we are about to write this file; will use self.output_dir as
            the base directory rather than self.survey_dir.

        brick : string, defaut=None
            Brick name.

        kwargs : dict, default={}
            Arguments for LegacySurveyData.find_file().

        Returns
        -------
        fn : string
            Path to the specified file (whether or not it exists).
        '''
        if filetype == 'obiwan-randoms':
            fn = super(LegacySurveySim,self).find_file('tractor',brick=brick,output=output,**kwargs)
            assert 'tractor' in fn # make sure not to overwrite tractor files
            dirname = os.path.dirname(fn).replace('tractor','obiwan')
            basename = os.path.basename(fn).replace('tractor','randoms')
            fn = os.path.join(dirname,basename)
        else:
            fn = super(LegacySurveySim,self).find_file(filetype,brick=brick,output=output,**kwargs)

        if brick is None:
            brick = '%(brick)s'

        def rsdir(fileid=0,rowstart=0,skipid=0):
            return 'file%d_rs%d_skip%d' % (fileid,rowstart,skipid)

        def wrap(fn):
            basename = os.path.basename(fn)
            dirname = os.path.dirname(fn)
            if 'coadd/' in dirname: return os.path.join(dirname,rsdir(**self.kwargs_file),basename)
            if 'obiwan/' in dirname or 'metrics/' in dirname or 'tractor/' in dirname or 'tractor-i/' in dirname:
                return os.path.join(dirname,brick,rsdir(**self.kwargs_file),basename)
            return fn

        if isinstance(fn,list):
            fn = list(map(wrap,fn))
        elif fn is not None:
            fn = wrap(fn)

        return fn


class CosmosSim(LegacySurveySim,CosmosSurvey):
    '''
    Filters the CCDs to just those in the cosmos survey.
    Call with LegacySurveySim arguments plus additional CosmosSurvey argument 'subset'.
    '''

    def filter_ccd_kd_files(self, fns):
        return [fn for fn in fns if 'decam+noise' in fn]
    def filter_ccds_files(self, fns):
        return [fn for fn in fns if 'decam+noise' in fn]
    def get_default_release(self):
        return 9007

class DecamSim(LegacySurveySim,DecamSurvey):
    '''
    Filters the CCDs to just those in the decam survey.
    '''
    pass

class NinetyPrimeMosaicSim(LegacySurveySim,NinetyPrimeMosaic):
    '''
    Filters the CCDs to just those in the mosaic or 90prime surveys.
    '''
    pass

runs = {
    'decam': DecamSim,
    '90prime-mosaic': NinetyPrimeMosaicSim,
    'south': DecamSim,
    'north': NinetyPrimeMosaicSim,
    'cosmos': CosmosSim,
    None: LegacySurveySim,
}

def get_survey(name, **kwargs):
    survey_class = runs[name]
    if name != 'cosmos':
        kwargs.pop('subset',None)
    survey = survey_class(**kwargs)
    return survey

class GSImage(galsim.Image):
    '''
    A wrapper around galsim.Image, to bypass array privacy.
    '''

    def __setitem__(self,*args):
        if (len(args) == 2) and isinstance(args[0],np.ndarray):
            self._array[args[0]] = args[1]
        else:
            super(GSImage,self).__setitem__(*args)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self,array):
        self._array = array

def _Image(array, bounds, wcs):
    '''
    This function of galsim.image must be redefined to have all methods of galsim.Image consistent
    within GSImage (e.g. copy()).
    Equivalent to ``GSImage(array, bounds, wcs)``, but without the overhead of sanity checks,
    and the other options for how to provide the arguments.
    '''
    ret = GSImage.__new__(GSImage)
    ret.wcs = wcs
    ret._dtype = array.dtype.type
    if ret._dtype in GSImage._alias_dtypes:
        ret._dtype = GSImage._alias_dtypes[ret._dtype]
        array = array.astype(ret._dtype)
    ret._array = array
    ret._bounds = bounds
    return ret

galsim.image._Image = _Image

def sum_invar(stamp_invar,tim_invar):
    '''
    Returns tim_invar when stamp_invar == 0, the harmonic sum of stamp_invar and tim_invar otherwise.

    Parameters
    ----------
    stamp_invar : GSImage

    tim_invar : GSImage

    Returns
    -------
    obj_invar : GSImage

    '''
    # return
    obj_invar = tim_invar.copy()
    obj_invar[stamp_invar.array>0] = (stamp_invar.array[stamp_invar.array>0]**(-1) + tim_invar.array[stamp_invar.array>0]**(-1))**(-1)
    return obj_invar


class BaseSimImage(object):

    '''
    Dumb class that overloads get_tractor_image for future multiple inheritance.
    '''

    def get_tractor_image(self, **kwargs):

        get_dq = kwargs.get('dq',True)
        kwargs['dq'] = True # dq required in the following
        tim = super(BaseSimImage,self).get_tractor_image(**kwargs)

        if tim is None: # this can be None when the edge of a CCD overlaps
            return tim

        #shot the image to 0
        #tim.data = np.zeros(tim.data.shape)
        t1 = Time()
        if self.survey.sim_stamp == 'tractor':
            objstamp = TractorSimStamp(self,tim)
        else:
            objstamp = GalSimStamp(self,tim)

        # Grab the data and inverse variance images [nanomaggies!]
        tim_image = GSImage(tim.getImage())
        tim_invar = GSImage(tim.getInvvar())
        tim_dq = GSImage(tim.dq)
        if not get_dq: del tim.dq
        # Also store galaxy sims and sims invvar
        sims_image = tim_image.copy()
        sims_image.fill(0.)
        sims_invar = sims_image.copy()

        if self.survey.simcat is None:
            return tim
        # Store simulated galaxy images in tim object
        # Loop on each object.
        #f = open(self.stamp_stat_fn,'a')
        for ii, obj in enumerate(self.survey.simcat):
            # Print timing
            t0 = Time()
            logger.info('Drawing object id=%d: sersic=%.2f, shape_r=%.2f, shape_e1=%.2f, shape_e2=%.2f' %
                 (obj.id,obj.sersic,obj.shape_r,obj.shape_e1,obj.shape_e2))
            stamp = objstamp.draw(obj)
            t0 = logger.info('%s finished drawing object id=%d: band=%s dbflux=%f addedflux=%f in %s' %
                (objstamp.__class__.__name__,obj.id,objstamp.band,obj.get('flux_%s' % objstamp.band),stamp.array.sum(),Time()-t0))
            stamp_nonoise = stamp.copy()
            if self.survey.add_sim_noise:
                stamp += objstamp.get_noise_gaussian(stamp)
            stamp_invar = objstamp.get_invar_poisson(stamp)
            # Add source if EVEN 1 pix falls on the CCD
            overlap = stamp.bounds & tim_image.bounds
            if overlap.area() > 0:
                logger.info('Stamp overlaps tim: id=%d band=%s' % (obj.id,objstamp.band))
                #self.survey.simcat.added[ii] = True
                stamp = stamp[overlap]
                stamp_invar = stamp_invar[overlap]
                stamp_nonoise = stamp_nonoise[overlap]
                # Zero out invar where bad pixel mask is flagged (> 0)
                stamp_invar[tim_dq[overlap].array > 0] = 0.
                # Add stamp to image
                tim_image[overlap] += stamp
                # Add variances
                tim_invar[overlap] = sum_invar(stamp_invar, tim_invar[overlap])
                #Extra
                sims_image[overlap] += stamp
                sims_invar[overlap] += stamp_invar

                if np.min(sims_invar.array) < 0:
                    logger.warning('Negative invvar!')
                    import pdb ; pdb.set_trace()
        #f.close()
        tim.sims_image = sims_image.array
        sims_invar = np.zeros_like(sims_image.array)
        sims_invar[sims_image.array > 0.] = 1./sims_image.array[sims_image.array > 0.]
        tim.sims_inverr = 1./np.sqrt(sims_invar)
        # Can set image=model, invar=1/model for testing
        if self.survey.image_eq_model:
            tim.data = tim.sims_image
            tim.inverr = tim.sims_inverr
        else:
            tim.data = tim_image.array
            tim.setInvvar(tim_invar.array)
        sys.stdout.flush()
        return tim


#class BaseSimImage(object):
#    pass


class DecamSimImage(BaseSimImage,DecamImage):

    pass

class BokSimImage(BaseSimImage,BokImage):

    pass

class MosaicSimImage(BaseSimImage,MosaicImage):

    pass

class DecamSimImagePlusNoise(BaseSimImage,DecamImagePlusNoise):

    pass

class BaseSimStamp(object):

    """
    Base class to draw model galaxies for a single image.

    """

    def __init__(self,image,tim,**attrs):

        '''
        Parameters
        ----------
        image : LegacySurveyImage
            Current LegacySurveyImage.

        tim : tractor.Image
            Current tractor.Image.

        attrs : dict, default={}
            Other attributes useful to define patches (e.g. nx, ny).

        '''
        self.band = tim.band.strip()
        self.zpscale = tim.zpscale
        self.attrs = attrs

        # nanomaggies-->ADU (decam) or e-/sec (bass,mzls)
        if image.camera == 'decam':
            gain = np.average([tim.hdr['GAINA'],tim.hdr['GAINB']])
            self.nano2e = self.zpscale*gain
        else:
            self.nano2e = self.zpscale
        self.tim = tim
        self.rng = image.survey.rng

    def set_local(self,ra,dec):
        '''
        Sets local x,y coordinates.

        Parameters
        ----------
        ra : float
            Right ascension (degree).

        dec : float
            Declination (degree).
        '''
        # x,y coordinates start at +1
        flag, self.xcen, self.ycen = self.tim.subwcs.radec2pixelxy(ra,dec)

    def get_noise_gaussian(self,gal):
        '''
        Generates a GSImage random Gaussian noise corresponding to gal.

        Parameters
        ----------
        gal : GSImage
            Input image.

        Returns
        -------
        noise : GSImage
            Noise.
        '''
        # Noise model + no negative image vals when compute noise
        noise = gal.copy()
        one_std_per_pix = noise.array
        one_std_per_pix[one_std_per_pix < 0] = 0
        one_std_per_pix = np.sqrt(one_std_per_pix * self.nano2e) # e-
        num_stds = self.rng.randn(*one_std_per_pix.shape)
        noise.array = one_std_per_pix * num_stds / self.nano2e #nanomaggies
        return noise

    def get_invar_poisson(self,gal):
        '''
        Returns a GSImage Poisson noise corresponding to gal.

        Parameters
        ----------
        gal : GSImage
            Input image.

        Returns
        -------
        invar : GSImage
            Inverse variance.
        '''
        invar = gal.copy() #nanomaggies
        #invar.array[:] = 0
        #invar.array[gal.array!=0] = self.nano2e**2/np.abs(gal.array[gal.array!=0]*self.nano2e)
        invar.array = self.nano2e**2/np.abs(gal.array*self.nano2e)
        return invar


class TractorSimStamp(BaseSimStamp):

    """
    Class to draw stamps with Tractor objects for a single image.

    """

    def get_subimage(self):
        '''
        Returns a subimage around xcen,ycen.

        Parameters
        ----------
        obj : SimCatalog row
            An object with attributes ra, dec.

        Returns
        -------
        tim : tractor.Image
            Subimage.
        '''
        nx = self.attrs.get('nx',64)
        ny = self.attrs.get('ny',64)
        xlow,ylow = nx//2+1,ny//2+1
        xhigh,yhigh = nx-xlow-1,ny-ylow-1
        # size of the stamp is 64*64, so get an image of the same size, unless it's around the edge
        xcen_int,ycen_int = round(self.xcen),round(self.ycen)
        # note: boundary need to be checked to see if it's consistent !TODO
        self.sx0,self.sx1,self.sy0,self.sy1 = xcen_int-xlow,xcen_int+xhigh,ycen_int-ylow,ycen_int+yhigh
        h,w = self.tim.shape
        self.sx0 = np.clip(self.sx0, 0, w-1)
        self.sx1 = np.clip(self.sx1, 0, w-1) + 1
        self.sy0 = np.clip(self.sy0, 0, h-1)
        self.sy1 = np.clip(self.sy1, 0, h-1) + 1
        return self.tim.subimage(self.sx0,self.sx1,self.sy0,self.sy1)

    def draw(self,obj):
        '''
        Returns a GSImage with obj in the center.
        If either obj.sersic or obj.shape_r is 0, a point source is drawn.
        Else a Sersic profile is used.

        Parameters
        ----------
        obj : SimCatalog row
            An object with attributes ra, dec, sersic, shape_r, shape_e1, shape_e2, flux_+self.band.

        Returns
        -------
        gim : GSImage
            Image with obj.
        '''
        ra,dec,flux = obj.ra,obj.dec,obj.get('flux_%s' % self.band)
        sersic,shape_r,shape_e1,shape_e2 = obj.sersic,obj.shape_r,obj.shape_e1,obj.shape_e2
        pos = tractor.RaDecPos(ra,dec)
        brightness = tractor.NanoMaggies(**{self.band:flux,'order':[self.band]})
        sersicindex = tractor.sersic.SersicIndex(sersic)
        if (shape_r==0.) or (sersic==0):
            src = tractor.PointSource(pos,brightness)
        else:
            shape = tractor.EllipseESoft(logre=np.log(shape_r),ee1=shape_e1,ee2=shape_e2)
            src = tractor.SersicGalaxy(pos=pos,brightness=brightness,
                                       shape=shape,sersicindex=sersicindex)
            #if sersic==1: src = tractor.ExpGalaxy(pos=pos,brightness=brightness,shape=shape)
            #elif sersic==4: src = tractor.DevGalaxy(pos=pos,brightness=brightness,shape=shape)
            #else: raise ValueError('sersic = %s should be 1 or 4' % sersic)
        self.set_local(ra,dec)
        new = tractor.Tractor([self.get_subimage()], [src])
        mod0 = new.getModelImage(0)
        gim = GSImage(mod0,xmin=self.sx0+1,ymin=self.sy0+1)
        return gim


class GalSimStamp(BaseSimStamp):

    """
    Class to draw stamps with GalSim objects for a single image.

    """
    def set_local(self,ra,dec):
        '''
        Sets local x,y coordinates, psf and scale.

        Parameters
        ----------
        ra : float
            Right ascension (degree).

        dec : float
            Declination (degree).
        '''
        # x,y coordinates start at +1
        super(GalSimStamp,self).set_local(ra,dec)
        #self.scale = self.tim.subwcs.pixscale_at(self.xcen,self.ycen)
        self.scale = self.tim.subwcs.pixel_scale()
        self.psf = GSImage(self.tim.psf.getPointSourcePatch(self.xcen,self.ycen).patch,scale=self.scale)
        self.psf = galsim.InterpolatedImage(self.psf)
        def frac(x):
            return abs(x-int(x))
        self.offsetint = galsim.PositionI(int(self.xcen),int(self.ycen))
        self.offsetfrac = galsim.PositionD(frac(self.xcen),frac(self.ycen))

    def draw(self,obj):
        '''
        Returns a GSImage with obj in the center.
        If either obj.sersic or obj.shape_r is 0, a point source is drawn.
        Else a Sersic profile is used.

        Parameters
        ----------
        obj : SimCatalog row
            An object with attributes ra, dec, sersic, shape_r, shape_e1, shape_e2, flux_+self.band.

        Returns
        -------
        gim : GSImage
            Image with obj.
        '''
        ra,dec,flux = obj.ra,obj.dec,obj.get('flux_%s' % self.band)
        sersic,shape_r,shape_e1,shape_e2 = obj.sersic,obj.shape_r,obj.shape_e1,obj.shape_e2
        gsparams = galsim.GSParams(maximum_fft_size=256**2)
        if (shape_r==0.) or (sersic==0):
            src = galsim.DeltaFunction(flux=flux,gsparams=gsparams)
        else:
            src = galsim.Sersic(sersic,half_light_radius=shape_r,flux=flux,gsparams=gsparams)
            src = src.shear(e1=shape_e1,e2=shape_e2)
        self.set_local(ra,dec)
        src = galsim.Convolve([src,self.psf])
        gim = src.drawImage(method='auto',scale=self.scale,
                            use_true_center=False,offset=self.offsetfrac)
        gim.setCenter(self.offsetint)
        return gim
