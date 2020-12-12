import os
import logging
import tempfile
import numpy as np
import fitsio
from legacypipe.catalog import read_fits_catalog
from tractor.basics import PointSource
from tractor.galaxy import DevGalaxy
from tractor.sersic import SersicGalaxy
from obiwan import LegacySurveySim, SimCatalog, BrickCatalog, runbrick, utils, setup_logging
from legacypipe.survey import wcs_for_brick

logger = logging.getLogger('obiwan.test_runbrick')

setup_logging(logging.DEBUG)

def generate_randoms(brickname, zoom=[0,3600,0,3600], zoom_margin=5, mag_range=[19.,20.], shape_r_range=[0.,1.], size=2, seed=42):

    brick = BrickCatalog().get_by_name(brickname)
    wcs = wcs_for_brick(brick)
    (x0,x1,y0,y1) = zoom
    W = x1-x0-2*zoom_margin
    H = y1-y0-2*zoom_margin
    assert (W>0) and (H>0)
    targetwcs = wcs.get_subimage(x0+zoom_margin, y0+zoom_margin, W, H)
    radecbox = np.ravel([targetwcs.pixelxy2radec(x,y) for x,y in [(1,1),(W,H)]],order='F')
    radecbox = np.concatenate([np.sort(radecbox[:2]),np.sort(radecbox[2:])])
    randoms = SimCatalog(size=size)
    rng = np.random.RandomState(seed=seed)
    randoms.ra,randoms.dec = utils.sample_ra_dec(radecbox=radecbox,size=randoms.size,rng=rng)
    randoms.bx,randoms.by = brick.get_xy_from_radec(randoms.ra,randoms.dec)
    flux_range = utils.mag2nano(mag_range)
    for b in ['g','r','z']:
        randoms.set('flux_%s' % b,rng.uniform(*flux_range,size=randoms.size))
    randoms.sersic = randoms.full(4)
    ba = rng.uniform(0.2,1.,size=randoms.size)
    phi = rng.uniform(0,np.pi,size=randoms.size)
    randoms.shape_e1,randoms.shape_e2 = utils.get_shape_e1_e2(ba,phi)
    randoms.shape_r = rng.uniform(*shape_r_range,size=randoms.size)
    randoms.brickname = randoms.full(brickname)

    return randoms

def test_case3():

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    outdir = 'out-testcase3-obiwan'
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
    checkpoint_fn = os.path.join(outdir, 'checkpoint.pickle')
    if os.path.exists(checkpoint_fn):
        os.unlink(checkpoint_fn)
    randoms_fn = os.path.join(outdir,'input_randoms.fits')
    brickname = '2447p120'
    zoom = [1020,1070,2775,2815]
    randoms = generate_randoms(brickname,zoom=[1020,1070,2785,2815],mag_range=[19.,20.],shape_r_range=[0.,0.])
    randoms.writeto(randoms_fn)

    for extra_args in [['--plots','--plot-base',os.path.join(outdir,'brick-%(brick)s')],
                    ['--sim-stamp','tractor'],['--sim-stamp','galsim'],
                    ['--sim-stamp','tractor','--add-sim-noise','gaussian'],
                    ['--sim-stamp','tractor','--add-sim-noise','poisson'],
                    ['--sim-stamp','galsim','--add-sim-noise','gaussian'],
                    ['--sim-stamp','galsim','--add-sim-noise','poisson'],
                    ['--sim-stamp','galsim','--add-sim-noise','gaussian','--nobj',1],
                    ['--sim-stamp','galsim','--add-sim-noise','gaussian','--rowstart',1,'--nobj',1],
                    ]:

        runbrick.main(args=['--brick', brickname, '--zoom', *map(str,zoom),
                            '--no-wise', '--force-all', '--no-write',
                            '--survey-dir', surveydir,
                            '--ran-fn', randoms_fn,
                            '--outdir', outdir,
                            '--seed', 42,
                            '--force-all',
                            '--threads', 1] + extra_args)

        # build-up truth
        origin_ra = [244.77973,244.77828]
        origin_dec = [12.07234,12.07250]
        origin_type = [(DevGalaxy,SersicGalaxy),PointSource]
        randoms = SimCatalog(randoms_fn)
        rowstart,nobj = 0,len(randoms)
        if '--rowstart' in extra_args: rowstart = extra_args[extra_args.index('--rowstart')+1]
        if '--nobj' in extra_args: nobj = extra_args[extra_args.index('--nobj')+1]
        randoms = randoms[rowstart:rowstart+nobj]
        ra,dec = np.concatenate([origin_ra,randoms.ra]),np.concatenate([origin_dec,randoms.dec])

        nsigmas = 30 # max tolerance
        # Read catalog into Tractor sources to test read_fits_catalog
        survey = LegacySurveySim(output_dir=outdir, kwargs_file={'rowstart':rowstart})
        fn = survey.find_file('tractor', brick=brickname, output=True)
        logger.info('Reading %s' % fn)
        tractor = SimCatalog(fn)

        # first match ra,dec
        assert len(tractor) == len(origin_ra) + len(randoms), 'Found %d objects, injected %d sources' % (len(tractor),len(origin_ra) + len(randoms))
        # first match ra,dec
        indin,indout,distance = utils.match_radec(ra,dec,tractor.ra,tractor.dec,radius_in_degree=0.05/3600.,nearest=True,return_distance=True)
        assert len(indin) == len(tractor), 'Matched %d objects among %d sources' % (len(indin),len(tractor)) # all matches
        indout = indout[np.argsort(indin)]
        tractor_all = tractor[indout] # reorder such that first -len(randoms): are injected sources
        # ra,dec tolerance
        sigma = np.sqrt(((tractor_all.ra-ra)**2*tractor_all.ra_ivar + (tractor_all.dec-dec)**2*tractor_all.dec_ivar)/2.)
        logger.info('Max angular distance is %.4f arcsec, %.4f sigmas' % (distance.max()*3600.,sigma.max()))
        assert np.all(sigma < nsigmas)
        # flux tolerance
        tractor = tractor_all[-len(randoms):]
        for b in ['g','r','z']:
            diff = np.abs(tractor.get('flux_%s' % b) - randoms.get('flux_%s' % b))
            sigma = diff*np.sqrt(tractor.get('flux_ivar_%s' % b))
            logger.info('Max flux diff in %s band is %.4f, %.4f sigmas' % (b,diff.max(),sigma.max()))
            assert np.all(sigma < nsigmas)

        cat = read_fits_catalog(tractor_all)
        logger.info('Read catalog: %s' % cat)
        assert len(cat) == len(tractor_all)
        # check origin sources are of the correct type
        for isrc,src in enumerate(cat[:len(randoms)]):
            assert isinstance(src,origin_type[isrc])

        # check injected sources are of the correct type
        for isrc,src in enumerate(cat[-len(randoms):]):
            assert isinstance(src,(PointSource))

def test_case3_shape():

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    outdir = 'out-testcase3-obiwan-shape'
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
    checkpoint_fn = os.path.join(outdir, 'checkpoint.pickle')
    if os.path.exists(checkpoint_fn):
        os.unlink(checkpoint_fn)
    randoms_fn = os.path.join(outdir,'input_randoms.fits')
    log_fn = os.path.join(outdir,'log.out')
    brickname = '2447p120'
    zoom = [1020,1070,2775,2815]
    randoms = generate_randoms(brickname,zoom=[1020,1040,2785,2815], zoom_margin=5, mag_range=[19.,20.], size=1)
    randoms.shape_r = randoms.full(2.)
    randoms.writeto(randoms_fn)

    for extra_args in [['--plots','--plot-base',os.path.join(outdir,'brick-%(brick)s')],
                    ['--sim-stamp','tractor'],['--sim-stamp','galsim'],
                    ['--sim-stamp','tractor','--add-sim-noise','gaussian'],
                    ['--sim-stamp','galsim','--add-sim-noise','poisson']
                    ]:

        runbrick.main(args=['--brick', brickname, '--zoom', *map(str,zoom),
                            '--no-wise', '--force-all', '--no-write',
                            '--survey-dir', surveydir,
                            '--ran-fn', randoms_fn,
                            '--outdir', outdir,
                            '--seed', 42,
                            '--force-all',
                            '--threads', 2,
                            '--verbose', '--log-fn', log_fn] + extra_args)

        setup_logging(logging.INFO)
        # build-up truth
        origin_ra = [244.77973,244.77828]
        origin_dec = [12.07234,12.07250]
        origin_type = [(DevGalaxy,SersicGalaxy),PointSource]
        ra,dec = np.concatenate([origin_ra,randoms.ra]),np.concatenate([origin_dec,randoms.dec])

        nsigmas = 80 # max tolerance
        # Read catalog into Tractor sources to test read_fits_catalog
        survey = LegacySurveySim(output_dir=outdir)
        fn = survey.find_file('tractor', brick=brickname, output=True)
        logger.info('Reading %s' % fn)
        tractor = SimCatalog(fn)

        assert len(tractor) == len(origin_ra) + len(randoms), 'Found %d objects, injected %d sources' % (len(tractor),len(origin_ra) + len(randoms))
        # first match ra,dec
        indin,indout,distance = utils.match_radec(ra,dec,tractor.ra,tractor.dec,radius_in_degree=0.05/3600.,nearest=True,return_distance=True)
        assert len(indin) == len(tractor), 'Matched %d objects among %d sources' % (len(indin),len(tractor)) # all matches
        indout = indout[np.argsort(indin)]
        tractor_all = tractor[indout] # reorder such that first -len(randoms): are injected sources
        # ra,dec tolerance
        sigma = np.sqrt(((tractor_all.ra-ra)**2*tractor_all.ra_ivar + (tractor_all.dec-dec)**2*tractor_all.dec_ivar)/2.)
        logger.info('Max angular distance is %.4f arcsec, %.4f sigmas' % (distance.max()*3600.,sigma.max()))
        assert np.all(sigma < nsigmas)
        # flux tolerance
        tractor = tractor_all[-len(randoms):]
        for b in ['g','r','z']:
            diff = np.abs(tractor.get('flux_%s' % b) - randoms.get('flux_%s' % b))
            sigma = diff*np.sqrt(tractor.get('flux_ivar_%s' % b))
            logger.info('Max flux diff in %s band is %.4f, %.4f sigmas' % (b,diff.max(),sigma.max()))
            assert np.all(sigma < nsigmas)

        for field in ['shape_e1','shape_e2','shape_r']:
            diff = np.abs(tractor.get(field) - randoms.get(field))
            sigma = diff*np.sqrt(tractor.get('%s_ivar' % field))
            logger.info('Max %s diff is %.4f, %.4f sigmas' % (field,diff.max(),sigma.max()))
            assert np.all(sigma < nsigmas)

        cat = read_fits_catalog(tractor_all)
        logger.info('Read catalog: %s' % cat)
        assert len(cat) == len(tractor_all)
        # check origin sources are of the correct type
        for isrc,src in enumerate(cat[:len(randoms)]):
            assert isinstance(src,origin_type[isrc])

        # check injected sources are of the correct type
        for isrc,src in enumerate(cat[-len(randoms):]):
            assert isinstance(src,(DevGalaxy,SersicGalaxy))


def test_mzlsbass2():

    surveydir = os.path.join(os.path.dirname(__file__), 'mzlsbass2')
    outdir = 'out-mzlsbass2-obiwan'
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'

    randoms_fn = os.path.join(outdir,'input_randoms.fits')
    log_fn = os.path.join(outdir,'log.out')
    brickname = '1773p595'
    zoom = [1300,1500,700,900]
    #randoms = generate_randoms(brickname,zoom=zoom,zoom_margin=10)
    randoms = generate_randoms(brickname,zoom=[1300,1400,700,800],zoom_margin=10)
    randoms.writeto(randoms_fn)

    for extra_args in [['--plots','--plot-base',os.path.join(outdir,'brick-%(brick)s')],
                    ['--sim-stamp','tractor','--add-sim-noise','gaussian'],
                    ['--sim-stamp','galsim','--add-sim-noise','poisson']
                    ]:

        runbrick.main(args=['--brick', brickname, '--zoom', *map(str,zoom),
                            '--no-wise', '--force-all', '--no-write',
                            '--survey-dir', surveydir,
                            '--ran-fn', randoms_fn,
                            '--outdir', outdir,
                            '--sim-blobs',
                            '--seed', 42,
                            '--verbose','--log-fn', log_fn] + extra_args)

        setup_logging(logging.INFO)
        nsigmas = 30 # max tolerance
        # Read catalog into Tractor sources to test read_fits_catalog
        survey = LegacySurveySim(output_dir=outdir)
        fn = survey.find_file('tractor', brick=brickname, output=True)
        logger.info('Reading %s' % fn)
        tractor = SimCatalog(fn)

        # first match ra,dec
        indin,indout,distance = randoms.match_radec(tractor,radius_in_degree=0.1/3600.,nearest=True,return_distance=True)
        assert len(indin) == len(randoms), 'Matched %d objects among %d injected sources' % (len(indin),len(randoms))
        indout = indout[np.argsort(indin)]
        tractor = tractor[indout] # reorder such that first -len(randoms): are injected sources
        # ra,dec tolerance
        sigma = np.sqrt(((tractor.ra-randoms.ra)**2*tractor.ra_ivar + (tractor.dec-randoms.dec)**2*tractor.dec_ivar)/2.)
        logger.info('Max angular distance is %.4f arcsec, %.4f sigmas' % (distance.max()*3600.,sigma.max()))
        assert np.all(sigma < nsigmas)
        # flux tolerance
        for b in ['g','r','z']:
            diff = np.abs(tractor.get('flux_%s' % b) - randoms.get('flux_%s' % b))
            sigma = diff*np.sqrt(tractor.get('flux_ivar_%s' % b))
            logger.info('Max flux diff in %s band is %.4f, %.4f sigmas' % (b,diff.max(),sigma.max()))
            assert np.all(sigma < nsigmas)


def test_plot_case3():

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    outdir = 'out-testcase3-obiwan'
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
    checkpoint_fn = os.path.join(outdir,'checkpoint.pickle')
    if os.path.exists(checkpoint_fn):
        os.unlink(checkpoint_fn)
    randoms_fn = os.path.join(outdir,'input_randoms.fits')
    brickname = '2447p120'
    zoom = [1020,1070,2775,2815]
    randoms = generate_randoms(brickname,zoom=[1020,1070,2785,2815],mag_range=[19.,20.],shape_r_range=[0.,0.])
    randoms.writeto(randoms_fn)

    for extra_args in [['--plots','--plot-base',os.path.join(outdir,'brick-%(brick)s')],
                    ]:

        runbrick.main(args=['--brick', brickname, '--zoom', *map(str,zoom),
                            '--no-wise', '--force-all', '--no-write',
                            '--survey-dir', surveydir,
                            '--ran-fn', randoms_fn,
                            '--outdir', outdir,
                            '--seed', 42,
                            '--force-all',
                            '--threads', 1] + extra_args)

    from matplotlib import pyplot as plt
    from obiwan.analysis import ImageAnalysis

    image = ImageAnalysis(outdir,brickname=brickname)
    image.read_sources(filetype='randoms')
    filetypes = ['image-jpeg','model-jpeg','resid-jpeg']
    image.read_image(filetype=filetypes[0],xmin=zoom[0],ymin=zoom[2])
    slicex,slicey = image.suggest_zooms(boxsize_in_pixels=20,range_observed_injected_in_degree=[0.,1.])[0]
    #filetypes = ['image','model']
    fig,lax = plt.subplots(ncols=len(filetypes),sharex=False,sharey=False,figsize=(4*len(filetypes),4),squeeze=False)
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    lax = lax.flatten()
    for ax,filetype in zip(lax,filetypes):
        image.read_image(filetype=filetype,xmin=zoom[0],ymin=zoom[2])
        #print(image.shape)
        image.set_subimage(slicex,slicey)
        image.plot(ax)
        image.plot_sources(ax)
    utils.savefig(fn=os.path.join(outdir,'injected_sources.png'))
