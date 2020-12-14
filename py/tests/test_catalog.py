import os
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
    assert np.all(cat.dec == cat.ra+1)
    assert cat.trues().all()
    assert not cat.falses().any()
    assert np.isnan(cat.nans()).all()
    assert np.all(cat.full(4) == 4)
    cat2 = cat.copy()
    cat2.ra[:] += 1
    assert np.all(cat.ra == 0)
    mask = cat.falses()
    mask[:50] = True
    cat2 = cat2[mask]
    assert len(cat2) == 50
    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir,'tmp.fits')
        cat.writeto(fn)
        cat2 = BaseCatalog(fn)
    assert cat2 == cat
    cat2 = 0
    cat2 += cat + 0
    cat2 = cat2 + cat
    assert cat2.size == 200
    cat.flux = np.linspace(0.,1.,cat.size)
    cat.shape_r = np.linspace(0.,1.,cat.size).astype('f4')
    cat.type = np.full(cat.size,'DEV')
    cat2.fill(cat,index_self=100+np.arange(cat.size),index_other=np.arange(cat.size),fields_other=['ra','dec'])
    assert cat2.fields == ['ra','dec']
    cat2.fill(cat,index_self=100+np.arange(cat.size),index_other=np.arange(cat.size))
    assert ('flux' in cat2) and ('shape_r' in cat2)
    assert np.isnan(cat2.flux[:100]).all() and np.isnan(cat2.shape_r[:100]).all()
    cat3 = cat2[slice(0,0)]
    cat3.fill(cat,index_self='after',index_other=np.arange(cat.size),fields_other=['ra','dec'])
    assert cat3.size == cat.size
    assert np.isnan(cat3.flux).all() and np.isnan(cat3.shape_r).all()
    assert np.all(cat3.ra == cat.ra) and np.all(cat3.dec == cat.dec)
    cat4 = cat3.copy()
    cat3.fill(cat,index_self='after',index_other=np.arange(0),fields_other=['ra','dec'])
    assert cat3 == cat4
    cat3.fill(cat,index_self='before')
    assert np.all(cat3.flux[:cat.size] == cat.flux) and np.all(cat3.shape_r[:cat.size] == cat.shape_r)
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
    brick = bricks.get_by_name(['2599p187'])
    assert len(brick.get('brickid')) == 1
    brick = bricks.get_by_name('2599p187')
    assert np.isscalar(brick.get('brickid'))
    brick = bricks.get_by_radec([259.91]*2,[18.71]*2)
    assert len(brick) == 2 and np.all(brick.brickname == '2599p187')
    brick2 = bricks[bricks.ra1>240].get_by_radec([259.91]*2,[18.71]*2)
    assert len(brick2) == 2 and np.all(brick2.brickname == '2599p187')
    radecbox = brick.get_radecbox(total=False)
    radecbox = brick.get_radecbox(total=True)
    assert np.all([np.isscalar(x) for x in radecbox])
    radecbox = bricks.get_radecbox(total=True)
    assert np.allclose(radecbox,(0.,360.,-90.,90.))
    area = brick.get_area(total=True)
    assert np.isscalar(area)
    area = bricks.get_area(total=True)
    assert np.allclose(area,4.*np.pi*(180./np.pi)**2)
    x,y = bricks.get_xy_from_radec(259.91,18.71)
    assert np.isscalar(x) and np.isscalar(y)
    x,y = bricks.get_xy_from_radec([259.91]*2,[18.71]*2,brickname=['2599p187']*2)
    assert (x>=0).all() and (x<=3600).all() and (y>=0).all() and (y<=3600).all()
    x2,y2 = brick.get_xy_from_radec([259.91]*2,[18.71]*2)
    assert np.allclose(x2,x) and np.allclose(y2,y)
    x3,y3 = bricks.get_xy_from_radec([13.42]+[259.91]*2+[97.81],[21.12]+[18.71]*2+[48.51])
    assert np.allclose(x3[1:-1],x) and np.allclose(y3[1:-1],y)
    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = tmp_dir + '/bricklist.txt'
        brick = bricks.get_by_name(['2599p187']).write_list(fn)
        with open(fn,'r') as file:
            assert file.read().replace('\n','') == '2599p187'
