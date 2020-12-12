import os
import tempfile
import logging
import argparse
import numpy as np
from obiwan.utils import *

setup_logging(logging.DEBUG)

def test_paths():
    from obiwan.kenobi import find_file,find_legacypipe_file,find_obiwan_file
    fn = find_legacypipe_file('tests','bricks',brickname='2599p187',fileid=0,rowstart=0,skipid=0)
    assert os.path.normpath(fn) == os.path.normpath('tests/survey-bricks.fits.gz')
    fn2 = find_file('tests','bricks',brickname='2599p187',source='legacypipe',fileid=0,rowstart=0,skipid=0)
    assert fn2==fn
    fn = find_obiwan_file('.','randoms',brickname='2599p187',fileid=1,rowstart=2,skipid=3)
    assert os.path.normpath(fn) == os.path.normpath('./obiwan/259/2599p187/file1_rs2_skip3/randoms-2599p187.fits')
    fn2 = find_file('.','randoms',brickname='2599p187',source='obiwan',fileid=1,rowstart=2,skipid=3)
    assert fn2==fn

def test_plots():
    with tempfile.TemporaryDirectory() as tmp_dir:
        @saveplot()
        def plot(self, ax,label='label'):
            ax.plot(np.linspace(0.,1.,10))
        plot(0,fn=os.path.join(tmp_dir,'plot.png'))

def test_misc():

    parent = argparse.ArgumentParser(add_help=True)
    parent.add_argument('--apodize', default=False, action='store_true',
    					help='Apodize image edges for prettier pictures?')
    parser = argparse.ArgumentParser(add_help=False,parents=[parent])
    parser.add_argument('-r', '--run', default=None,
    					help='Set the run type to execute')
    group = parser.add_argument_group('Obiwan', 'Obiwan-specific arguments')
    group.add_argument(
    	'-f', '--force-stage', dest='force', action='append', default=[],
    	help="Force re-running the given stage(s) -- don't read from pickle.")
    group.add_argument('--sim-blobs', action='store_true',
    					help='Process only the blobs that contain simulated sources.')
    assert get_parser_dests(group) == ['apodize','run','force','sim_blobs']
    """
    # Works but file not created in pytest...
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir,'log.out')
        setup_logging(logging.INFO,filename=tmp_file)
        words = 'TESTOBIWAN'
        logger.info(words)
        ok = False
        with open(tmp_file,'r') as tmp:
            for line in tmp.readlines():
                if words in line:
                    ok = True
                    break
        assert ok
    """
    truth = ['--a','1','--b','2']
    args = '--a 1 --b 2'
    assert get_parser_args(args) == truth
    args = ['--a',1,'--b',2]
    assert get_parser_args(args) == truth
    args = {'a':1,'b':2}
    assert get_parser_args(args) == truth

    id1 = np.arange(4)
    id2 = np.array([1,3,2,4,6])
    ind1,ind2 = match_id(id1,id2)
    assert len(ind1) == len(ind2) and len(ind1) == 3
    assert (id1[ind1] == id2[ind2]).all()

def test_radec():
    ramin,ramax,decmin,decmax = 259.9,260.2,18.7,18.8
    ra, dec = sample_ra_dec(size=None,radecbox=[ramin,ramax,decmin,decmax],seed=20)
    assert np.isscalar(ra) and np.isscalar(dec)
    ra, dec = sample_ra_dec(size=20,radecbox=[ramin,ramax,decmin,decmax],seed=20)
    assert len(ra) == 20 and len(dec) == 20
    assert np.all((ra>=ramin) & (ra<=ramax) & (dec>=decmin) & (dec<=decmax))
    ind1,ind2 = match_radec(ra,dec,ra[::-1],dec[::-1])
    assert (ind1 == ind2[::-1]).all()
    mask = mask_collisions(ra,dec,radius_in_degree=1.)
    assert mask[1:].all()
    ra[:] = dec[:] = 0
    ra = np.linspace(0.,20.,ra.size)
    ra[:10] = 0.1
    mask = mask_collisions(ra,dec,radius_in_degree=0.5)
    assert mask[1:10].all() and not mask[10:].any()
    area = get_radecbox_area(ramin,ramax,decmin,decmax)
    assert np.isscalar(area)
    decfrac = np.diff(np.rad2deg(np.sin(np.deg2rad([decmin,decmax]))),axis=0)
    rafrac = np.diff([ramin,ramax],axis=0)
    assert np.allclose(area,decfrac*rafrac)

def test_quantities():
    ba,phi = 0.42,0.69
    e1,e2 = get_shape_e1_e2(ba,phi)
    ba_,phi_ = get_shape_ba_phi(e1,e2)
    assert np.allclose(ba,ba_) and np.allclose(phi,phi_)
    mag = 24.
    nano = mag2nano(mag)
    mag_ = nano2mag(nano)
    assert np.allclose(mag,mag_)
    ra,dec = 12,4
    ebvref = get_extinction(ra,dec)
    assert np.isscalar(ebvref) and np.allclose(ebvref,0.01896500348082133)
    ebv = get_extinction([ra]*4,[dec]*4)
    assert (ebv.size==4) and np.allclose(ebv,ebvref)
    trans_g = get_extinction(ra,dec,band='g',camera='DES')
    assert (ebv.size==4) and np.allclose(trans_g,3.214*ebv)
