import argparse
import logging
from matplotlib import pyplot as plt
import numpy as np
from obiwan import SimCatalog,utils,setup_logging
from obiwan.analysis import ImageAnalysis,MatchAnalysis,RessourceAnalysis
#import settings
import settings_ebv as settings

logger = logging.getLogger('postprocessing')

if __name__ == '__main__':

    setup_logging()

    parser = argparse.ArgumentParser(description='Obiwan postprocessing')
    parser.add_argument('-d','--do',nargs='*',type=str,choices=['test','match','plotmatch','plottime','plotsource'],default=[],required=False,help='What should I do')
    opt = parser.parse_args()

    match_fn = settings.dir_plot+'match.npy'

    if 'test' in opt.do:
        from obiwan import find_file
        from obiwan import BrickCatalog
        #fn = find_file(settings.output_dir,'randoms',brickname='1011p407',struct='obiwan')
        brickname = '1011p407'
        randoms = SimCatalog(settings.randoms_fn)
        #randoms.fill_obiwan()
        bricks = BrickCatalog()
        #randoms.cut(randoms.brickname == brickname)
        randoms.cut(np.in1d(randoms.brickname,['1011p407','0951p562']))
        randoms.bx,randoms.by = bricks.get_xy_from_radec(randoms.ra,randoms.dec,brickname=randoms.brickname)
        randoms.cut(randoms.brickname == brickname)
        #randoms.fill_obiwan()

        bx,by = bricks.get_xy_from_radec(randoms.ra,randoms.dec,brickname=randoms.brickname)
        print(randoms.bx-bx)
        """
        fn = find_file(settings.output_dir,'randoms',brickname=brickname,struct='obiwan')
        out = SimCatalog(fn)
        print(randoms.bx-out.bx)
        """
        """
        bricks = BrickCatalog()
        bx,by = bricks.get_xy_from_radec(randoms.ra,randoms.dec,brickname=randoms.brickname)
        print(randoms.bx-bx)
        """
        """
        survey = settings.survey_dir
        from obiwan.kenobi import LegacySurveySim
        survey = LegacySurveySim(survey_dir=survey)
        bricks = BrickCatalog(survey)
        bx,by = bricks.get_xy_from_radec(randoms.ra,randoms.dec,brickname=randoms.brickname)
        print(randoms.bx-bx)
        """

    if 'match' in opt.do:
        match = 0
        from obiwan import scripts
        for brickname in np.unique(settings.get_bricknames())[:1]: # 1011p407 appears twice in the bricklist...
            match += scripts.match(settings.output_dir,brickname,base='inter',radius_in_degree=0.2/3600,
                                rowstart=settings.rowstart,fileid=settings.fileid,skipid=settings.skipid)
        #match.writeto(settings.randoms_matched_fn)
        """
        match = MatchAnalysis(base_dir=settings.output_dir, bricknames=settings.get_bricknames(),
                            kwargs_files=settings.kwargs_file,cats_dir=settings.merged_dir)
        match.merge_catalogs('randoms',cat_base='randoms.fits',write=True)
        match.merge_catalogs('tractor',cat_base='tractor.fits',write=True)
        #match.match(radius_in_degree=0.2/3600.,add_input_tractor=settings.legacypipe_output_dir)
        match.match(radius_in_degree=0.05/3600.,add_input_tractor=False)
        #cat = match.export(base='input',key_input='input',key_output=None,cat_base='match.fits')
        match.save(save_fn=match_fn)
        """

    if 'plotmatch' in opt.do:
        match = MatchAnalysis.load(match_fn)
        fields = ['ra','dec','flux_g','flux_r','flux_z']

        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=False,sharey=False,figsize=(12,6))
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            match.plot_scatter(ax,field=field,diagonal=True,kwargs_range={'quantiles':[0.01,0.99]})
        utils.savefig(fn=settings.dir_plot+'scatter_match.png')

        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=False,sharey=False,figsize=(12,6))
        fig.subplots_adjust(hspace=0.3,wspace=0.2)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            match.plot_hist(ax,field=field,kwargs_xedges={'range':[-4.,4.]})
        utils.savefig(fn=settings.dir_plot+'hist_match_rel.png')
        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=False,sharey=False,figsize=(12,6))
        fig.subplots_adjust(hspace=0.3,wspace=0.2)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            match.plot_hist(ax,field=field,divide_uncer=False,kwargs_xedges={'quantiles':[0.01,0.99]})
        utils.savefig(fn=settings.dir_plot+'hist_match_abs.png')

    if 'plottime' in opt.do:

        for brickname in settings.get_bricknames()[:0]:
            time = RessourceAnalysis(base_dir=settings.output_dir,
                    bricknames=[brickname],kwargs_files=settings.kwargs_file,cats_dir=settings.merged_dir)
            time.plot_one_series(fn=settings.dir_plot+'time_series_%s.png' % brickname)

        time = RessourceAnalysis(base_dir=settings.output_dir,bricknames=settings.get_bricknames(),
                                kwargs_files=settings.kwargs_file,cats_dir=settings.merged_dir)
        fig,lax = plt.subplots(ncols=2,sharex=False,sharey=False,figsize=(12,4))
        fig.subplots_adjust(hspace=0.2,wspace=0.2)
        lax = lax.flatten()
        time.plot_bar(ax=lax[0])
        time.plot_all_series(ax=lax[1])
        utils.savefig(fn=settings.dir_plot+'time_series_all.png')

    if 'plotsource' in opt.do:
        for brickname in settings.get_bricknames()[-5:]:
        #for brickname in ['1588p560']:
            image = ImageAnalysis(settings.output_dir,brickname=brickname)
            image.read_sources(filetype='randoms')
            filetypes = ['image-jpeg','model-jpeg','resid-jpeg']
            image.read_image(filetype=filetypes[0],xmin=0,ymin=0)
            slicex,slicey = image.suggest_zoom(boxsize_in_pixels=100)
            #filetypes = ['image','model']
            fig,lax = plt.subplots(ncols=len(filetypes),sharex=False,sharey=False,figsize=(4*len(filetypes),4),squeeze=False)
            fig.subplots_adjust(hspace=0.2,wspace=0.2)
            lax = lax.flatten()
            for ax,filetype in zip(lax,filetypes):
                image.read_image(filetype=filetype,xmin=0,ymin=0)
                image.set_subimage(slicex,slicey)
                image.plot(ax)
                image.plot_sources(ax)
            utils.savefig(fn=settings.dir_plot+'image-%s.png' % brickname)
