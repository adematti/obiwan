import argparse
import logging
from obiwan import SimCatalog,utils,setup_logging,scripts
import settings

logger = logging.getLogger('postprocessing')

if __name__ == '__main__':

    setup_logging()

    parser = argparse.ArgumentParser(description='Obiwan preprocessing')
    parser.add_argument('-d','--do',nargs='*',type=str,choices=['match','plot'],default=[],required=False,help='What should I do')
    opt = parser.parse_args()

    if 'match' in opt.do:
        match = 0
        for brickname in settings.get_bricknames():
            match += scripts.match(settings.output_dir,brickname,base='inter',radius_in_degree=0.2/3600,
                                rowstart=settings.rowstart,fileid=settings.fileid,skipid=settings.skipid)
        match.writeto(settings.randoms_matched_fn)

    if 'plot' in opt.do:
        from matplotlib import pyplot as plt
        match = SimCatalog(settings.randoms_matched_fn)
        fields = ['ra','dec','flux_g','flux_r','flux_z','sersic','shape_r','shape_e1','shape_e2']
        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=False,sharey=False,figsize=(16,6))
        fig.subplots_adjust(hspace=0.4,wspace=0.4)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            scripts.scatter_match(ax,match=match,field=field,diagonal=True)
        utils.savefig(fn=settings.dir_plot+'match.png')
