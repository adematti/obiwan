import argparse
import logging
import numpy as np
from obiwan import scripts,SimCatalog,utils,setup_logging
import settings

logger = logging.getLogger('postprocessing')

if __name__ == '__main__':

    setup_logging()

    parser = argparse.ArgumentParser(description='Obiwan preprocessing')
    parser.add_argument('-d','--do',nargs='*',type=str,choices=['match','plot'],default=[],required=False,help='What should I do')
    opt = parser.parse_args()

    if 'match' in opt.do:
        match = 0
        for brickname in settings.bricknames:
            match += scripts.match(settings.output_dir,brickname,base='inter',radius_in_degree=0.2/3600,rowstart=0,fileid=0,skipid=0)
        print(np.abs(match.sim_dec-match.dec).max()*3600.)
        logger.info('Matched catalog of size %d' % match.size)
        match.writeto(settings.randoms_matched_fn)

    if 'plot' in opt.do:
        from matplotlib import pyplot as plt
        match = SimCatalog(settings.randoms_matched_fn)
        fields = ['ra','dec','flux_g','flux_r','flux_z','sersic','shape_r','shape_e1','shape_e2']
        print(match.sim_id)
        for field in fields:
            print(field,match.get('sim_%s' % field),match.get(field))
        #for b in ['g','r','z']:
        #    print(b,match.get('apflux_ivar_%s' % b)**(-0.5))
        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=True,sharey=True,figsize=(16,6))
        fig.subplots_adjust(hspace=0.4,wspace=0.4)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            scripts.scatter_match(ax,match=match,diagonal=True)
        utils.savefig(fn=settings.dir_plot+'match.png')
