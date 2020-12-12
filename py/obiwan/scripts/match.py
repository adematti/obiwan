import os
import argparse
import logging
from matplotlib import pyplot as plt
from obiwan.analysis import MatchAnalysis,RunCatalog,utils
from obiwan import find_file,utils,setup_logging

logger = logging.getLogger('match')

def main(args=None):

    parser = argparse.ArgumentParser(description='Match')
    parser.add_argument('--randoms', type=str, default=None,
                        help='Filename of merged randoms catalog')
    parser.add_argument('--tractor', type=str, default=None,
                        help='Filename of merged Tractor catalog')
    parser.add_argument('--tractor-legacypipe', nargs='?', type=str, default=False, const=None,
                        help='Add legacypipe fitted sources to inject sources in the matching. Load from legacypipe directory or filename if provided.')
    parser.add_argument('--radius', type=float, default=1.5, help='Matching radius in arcseconds')
    parser.add_argument('--base', type=str, default='input', help='Catalog to be used as base for merging')
    parser.add_argument('--cat-dir', type=str, default='.', help='Matched catalog directory')
    cat_matched_base = 'matched_%(base)s.fits'
    parser.add_argument('--cat-fn', type=str, nargs='*', default=None, help='Output filename. '\
                        'If not provided, defaults to cat-dir/%s' % cat_matched_base)
    plot_hist_base_template = 'hist_output_input.png'
    parser.add_argument('--plot-hist', nargs='?', type=str, default=False, const=True,
                        help='Plot histograms of difference (output-input) and residuals. If no filename provided, defaults to cats-dir + %s' % plot_hist_base_template)
    plot_scatter_base_template = 'scatter_output_input.png'
    parser.add_argument('--plot-scatter', nargs='?', type=str, default=False, const=True,
                        help='Scatter plot difference (output-input). If no filename provided, defaults to cats-dir/%s' % plot_scatter_base_template)
    parser.add_argument('--plot-fields', type=str, nargs='*', default=['ra','dec','flux_g','flux_r','flux_z'], help='Fields to plot')
    RunCatalog.get_output_parser(parser=parser)
    opt = parser.parse_args(args=utils.get_parser_args(args))

    if any([getattr(opt,filetype) is None for filetype in ['randoms','tractor','tractor_legacypipe']]):
        runcat = RunCatalog.from_output_cmdline(opt)
        match = MatchAnalysis(base_dir=opt.output_dir,runcat=runcat)
    else:
        match = MatchAnalysis()

    for filetype in ['randoms','tractor']:
        cat_fn = getattr(opt,filetype)
        if cat_fn is not None:
            cat = match.read_catalog(cat_fn=cat_fn,filetype=filetype,source='obiwan')
            match.runcat = RunCatalog.from_catalog(cat)
    if opt.tractor_legacypipe is None:
        add_input_tractor = True
    elif not opt.tractor_legacypipe:
        add_input_tractor = False
    else:
        if os.path.isfile(opt.tractor_legacypipe):
            add_input_tractor = True
            match.read_catalog(cat_fn=opt.tractor_legacypipe,filetype='tractor',source='legacypipe')
        else: #legacypipe directory
            add_input_tractor = opt.tractor_legacypipe

    match.match(radius_in_degree=opt.radius/3600.,add_input_tractor=add_input_tractor)

    if opt.cat_fn is None:
        opt.cat_fn = os.path.join(opt.cat_dir,cat_matched_base % {'base':opt.base})
    match.export(base=opt.base,key_input='input',key_output=None,cat_fn=opt.cat_fn)

    if opt.plot_hist:
        if not isinstance(opt.plot_hist,str):
            opt.plot_hist = os.path.join(opt.cat_dir,plot_hist_base_template)
        fig,lax = plt.subplots(ncols=len(opt.plot_fields),nrows=2,sharex=False,sharey=False,figsize=(5*len(opt.plot_fields),5))
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        for ax,field in zip(lax[0],opt.plot_fields):
            try:
                match.plot_hist(ax,field=field,divide_uncer=False,kwargs_xedges={'quantiles':[0.01,0.99]})
            except KeyError:
                logger.warning('Could not plot %s bias.' % field)
        for ax,field in zip(lax[1],opt.plot_fields):
            try:
                match.plot_hist(ax,field=field,divide_uncer=True,kwargs_xedges={'quantiles':[0.01,0.99]})
            except KeyError:
                logger.warning('Could not plot %s residual.' % field)
        utils.savefig(fn=opt.plot_hist)

    if opt.plot_scatter:
        if not isinstance(opt.plot_scatter,str):
            opt.plot_scatter = os.path.join(opt.cat_dir,plot_scatter_base_template)
        fig,lax = plt.subplots(ncols=len(opt.plot_fields),nrows=1,sharex=False,sharey=False,figsize=(5*len(opt.plot_fields),5))
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        for ax,field in zip(lax,opt.plot_fields):
            match.plot_scatter(ax,field=field,diagonal=True,kwargs_xlim={'quantiles':[0.01,0.99]},kwargs_ylim={'quantiles':[0.01,0.99]},square=True)
        utils.savefig(fn=opt.plot_scatter)

if __name__ == '__main__':

    setup_logging()
    main()
