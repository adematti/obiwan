import os
import argparse
import logging
from matplotlib import pyplot as plt
from obiwan import SimCatalog,utils,setup_logging
from obiwan.analysis import ImageAnalysis,MatchAnalysis,RessourceAnalysis
import settings

logger = logging.getLogger('postprocessing')

if __name__ == '__main__':

    setup_logging()

    parser = argparse.ArgumentParser(description='Obiwan postprocessing')
    parser.add_argument('-d','--do',nargs='*',type=str,choices=['match','plotmatch','plottime','plotsource'],default=[],required=False,help='What should I do')
    opt = parser.parse_args()

    match_fn = settings.dir_plot+'match.npy'

    if 'match' in opt.do:
        match = MatchAnalysis(base_dir=settings.output_dir, bricknames=settings.get_bricknames(),
                            kwargs_files=settings.kwargs_file,cats_dir=settings.merged_dir)
        match.merge_catalogs('randoms',cat_base='randoms.fits',write=True)
        match.merge_catalogs('tractor',cat_base='tractor.fits',write=True)
        match.match(radius_in_degree=0.2/3600.,add_input_tractor=settings.legacypipe_output_dir)
        cat = match.export(base='input',key_input='input',key_output=None,cat_base='match.fits')
        match.save(save_fn=match_fn)

    if 'plotmatch' in opt.do:
        match = MatchAnalysis.load(match_fn)
        fields = ['ra','dec','flux_g','flux_r','bx','by']
        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=False,sharey=False,figsize=(12,6))
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            match.plot_scatter(ax,field=field,diagonal=True,kwargs_range={'quantiles':[0.01,0.99]})
        utils.savefig(fn=settings.dir_plot+'scatter_match.png')

        """
        fig,lax = plt.subplots(ncols=3,nrows=2,sharex=False,sharey=False,figsize=(12,6))
        fig.subplots_adjust(hspace=0.3,wspace=0.2)
        lax = lax.flatten()
        for ax,field in zip(lax,fields):
            match.plot_hist(ax,field=field,kwargs_xedges={'range':[-4.,4.]})
        utils.savefig(fn=settings.dir_plot+'hist_match_rel.png')
        """
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
        #for brickname in settings.get_bricknames():
        for brickname in settings.get_bricknames():
            """
            image = ImageAnalysis(settings.output_dir,brickname=brickname,kwargs_file=settings.kwargs_file)
            image.read_sources(filetype='randoms')
            slicex,slicey = image.suggest_zoom(boxsize_in_pixels=100)
            filetypes = ['image-jpeg']
            fig,lax = plt.subplots(ncols=len(filetypes),sharex=False,sharey=False,figsize=(4*len(filetypes),4),squeeze=False)
            fig.subplots_adjust(hspace=0.2,wspace=0.2)
            lax = lax.flatten()
            for ax,filetype in zip(lax,filetypes):
                image.read_image(filetype=filetype)
                image.set_subimage(slicex,slicey)
                image.plot(ax)
                image.plot_sources(ax)
            utils.savefig(fn=settings.dir_plot+os.path.basename(image.img_fn))
            """
            image = ImageAnalysis(settings.output_dir,brickname=brickname)
            image.read_sources(filetype='randoms')
            filetypes = ['image-jpeg','model-jpeg','resid-jpeg']
            #filetypes = ['image','model']
            image.read_image(filetype=filetypes[0],xmin=0,ymin=0)
            slicex,slicey = image.suggest_zoom(boxsize_in_pixels=100)
            #slicex,slicey = slice(0,500),slice(0,500)
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
