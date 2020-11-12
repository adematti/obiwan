import tempfile
import logging
import numpy as np
from obiwan import *
from obiwan.scripts import *

setup_logging(logging.DEBUG)

def test_match():
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir,brickname = output_dir,'2599p187'
        cat = SimCatalog(size=100)
        cat.id = np.arange(cat.size)
        cat.ra, cat.dec = utils.sample_ra_dec(size=cat.size,radecbox=[259.9,260.2,18.7,18.8],seed=20)
        cat.writeto(utils.get_output_file(output_dir,'obiwan-randoms',brickname=brickname,fileid=0,rowstart=0,skipid=0))
        cat[10:].writeto(utils.get_output_file(output_dir,'tractor',brickname=brickname,fileid=0,rowstart=0,skipid=0))
        cat = match(output_dir,brickname,base='input',radius_in_degree=0.001/3600.)
        assert cat.fields == ['sim_id','sim_ra','sim_dec','id','ra','dec']
        assert np.isnan(cat.ra[:10]).all() and np.isnan(cat.dec[:10]).all()
        assert np.all((cat.ra==cat.sim_ra)[10:]) and np.all((cat.dec==cat.sim_dec)[10:])
        cat = match(output_dir,brickname,base='output',radius_in_degree=1.5/3600.)
        assert np.allclose(cat.ra,cat.sim_ra) and np.allclose(cat.dec,cat.sim_dec)
        scatter_match(fn=output_dir+'scatter.png',match=cat,field='ra')
