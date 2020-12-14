"""
Script to produce image cutouts.

For details, run::

    python cutout.py --help

"""

import os
import argparse
import logging
from matplotlib import pyplot as plt
from obiwan.analysis import ImageAnalysis,RunCatalog
from obiwan import find_file,utils,setup_logging

logger = logging.getLogger('cutout')

def main(args=None):

    parser = argparse.ArgumentParser(description='Cutout')
    parser.add_argument('--ncuts', type=int, default=1,
                        help='Maximum number of cutouts for each brick run. -1 to ignore')
    parser.add_argument('--xmin', type=int, default=0,
                        help='x-coordinate of (0,0) corner')
    parser.add_argument('--ymin', type=int, default=0,
                        help='y-coordinate of (0,0) corner')
    plot_base_template = 'cutout-%(brickname)s-%(icut)d.png'
    parser.add_argument('--plot-fn', type=str, default=None, help='Plot file name; '\
                        'defaults to coadd-dir/%s' % plot_base_template)
    RunCatalog.get_output_parser(parser=parser)
    opt = parser.parse_args(args=utils.get_parser_args(args))
    runcat = RunCatalog.from_output_cmdline(opt)

    for run in runcat:
        image = ImageAnalysis(opt.output_dir,brickname=run.brickname,kwargs_file=run.kwargs_file)
        image.read_sources(filetype='randoms')
        filetypes = ['image-jpeg','model-jpeg','resid-jpeg']
        image.read_image(filetype=filetypes[0],xmin=opt.xmin,ymin=opt.ymin)
        slices = image.suggest_zooms()
        if opt.ncuts >= 0:
            slices = slices[:opt.ncuts]
        for islice,(slicex,slicey) in enumerate(slices):
            fig,lax = plt.subplots(ncols=len(filetypes),sharex=False,sharey=False,figsize=(4*len(filetypes),4),squeeze=False)
            fig.subplots_adjust(hspace=0.2,wspace=0.2)
            for ax,filetype in zip(lax[0],filetypes):
                image.read_image(filetype=filetype,xmin=opt.xmin,ymin=opt.ymin)
                image.set_subimage(slicex,slicey)
                image.plot(ax)
                image.plot_sources(ax)
            plot_fn_kwargs = {'brickname':run.brickname,'icut':islice+1}
            if opt.plot_fn is None:
                image_fn = find_file(base_dir=opt.output_dir,filetype='image-jpeg',brickname=run.brickname,source='obiwan',**run.kwargs_file)
                plot_fn = os.path.join(os.path.dirname(image_fn),plot_base_template % plot_fn_kwargs)
                if plot_fn == image_fn:
                    raise ValueError('Cutout filename is the same as image: %s' % image_fn)
            else:
                plot_fn = opt.plot_fn % plot_fn_kwargs
            utils.savefig(fn=plot_fn)

if __name__ == '__main__':

    setup_logging()
    main()
