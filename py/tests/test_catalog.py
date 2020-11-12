import tempfile
import logging
import numpy as np
from obiwan import *

setup_logging(logging.DEBUG)

def test_base():
    cat = BaseCatalog(size=100)
    cat.ra,cat.dec = cat.zeros(),cat.ones()
    assert 'ra' in cat
    assert cat.ra.size == 100
    assert np.all(cat.dec==cat.ra+1)
    assert cat.trues().all()
    assert not cat.falses().any()
    assert np.isnan(cat.nans()).all()
    cat2 = cat.copy()
    cat2.ra[:] += 1
    assert np.all(cat.ra == 0)
    mask = cat.falses()
    mask[:50] = True
    cat2 = cat2[mask]
    assert len(cat2) == 50
    with tempfile.TemporaryDirectory() as dir:
        fn = dir + 'tmp.fits'
        cat.writeto(fn)
        cat2 = BaseCatalog(fn)
    assert cat2 == cat
    cat2 = 0
    cat2 += cat + 0
    cat2 = cat2 + cat
    assert cat2.size==200
    cat.flux = np.linspace(0.,1.,cat.size)
    cat2.merge(cat,index_self=100+np.arange(cat.size),index_other=np.arange(cat.size))
    assert 'flux' in cat2
    assert np.isnan(cat2.flux[:100]).all()
    cat2.keep_columns('ra','dec')
    assert cat2.fields == ['ra','dec']
    cat2.delete_columns('ra','dec')
    assert cat2.fields == []

def test_sim():
    cat = SimCatalog(size=100)
    cat.ra, cat.dec = utils.sample_ra_dec(size=cat.size,radecbox=[259.9,260.1,18.7,18.8],seed=20)
    cat.fill_obiwan()
    assert np.all(cat.id == np.arange(len(cat)))
    assert np.all(cat.brickname == '2599p187')
    mask = cat.mask_collisions(radius_in_degree=1.)
    assert mask[1:].all()
    ind1,ind2 = cat.match_radec(cat[::-1],radius_in_degree=1e-6)
    assert (ind1 == ind2[::-1]).all()

def test_brick():
    bricks = BrickCatalog()
    brick = bricks.get_by_name('2599p187')
    assert np.isscalar(brick.get('brickid'))
    brick = bricks.get_by_name(['2599p187'])
    assert len(brick.get('brickid')) == 1
    brick = bricks.get_by_radec([259.91]*2,[18.71]*2)
    assert len(brick) == 2 and np.all(brick.brickname == '2599p187')
    radecbox = bricks.get_radecbox(all=True)
    assert np.allclose(radecbox,(0.,360.,-90.,90.))
    area = bricks.get_area(all=True)
    assert np.allclose(area,4.*np.pi*(180./np.pi)**2)
    x,y = bricks.get_xy_from_radec([259.91]*2,[18.71]*2)
    assert (x>=0).all() and (x<=3600).all() and (y>=0).all() and (y<=3600).all()
    with tempfile.TemporaryDirectory() as dir:
        fn = dir + '/bricklist.txt'
        brick = bricks.get_by_name(['2599p187']).write_list(fn)
        with open(fn,'r') as file:
            assert file.read().replace('\n','') == '2599p187'
