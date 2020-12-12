import os
import argparse
import logging
from obiwan.analysis import RunCatalog,BaseAnalysis
from obiwan import utils,setup_logging

logger = logging.getLogger('merge')

def main(args=None):
    parser = argparse.ArgumentParser(description='Merge')
    parser.add_argument('--filetype', type=str, nargs='*', default=[],
                        help='Filetypes to merge.')
    parser.add_argument('--source', type=str, choices=['obiwan','legacypipe'], nargs='*', default=None,
                        help='legacypipe or obiwan file?')
    parser.add_argument('--cat-dir', type=str, default='.', help='Merged catalog directory')
    cat_base_template = 'merged_%(filetype)s.fits'
    cat_legacypipe_base_template = 'merged_%(filetype)s_legacypipe.fits'
    parser.add_argument('--cat-fn', type=str, nargs='*', default=[],
                        help='Output filename. If not provided, defaults to cat-dir/%s if source is legacypipe, else cat-dir/%s.' % (cat_legacypipe_base_template,cat_base_template))
    RunCatalog.get_output_parser(parser=parser)
    opt = parser.parse_args(args=utils.get_parser_args(args))
    runcat = RunCatalog.from_output_cmdline(opt)

    if opt.source is None:
        opt.source = ['obiwan']*len(opt.filetype)
    if not opt.cat_fn:
        for filetype,source in zip(opt.filetype,opt.source):
            opt.cat_fn.append(os.path.join(opt.cat_dir,(cat_legacypipe_base_template if source == 'legacypipe' else cat_base_template) % {'filetype':filetype}))
    merge = BaseAnalysis(base_dir=opt.output_dir,runcat=runcat)
    #for field in runcat.fields: print(field,runcat.get(field))
    for filetype,source,cat_fn in zip(opt.filetype,opt.source,opt.cat_fn):
        merge.merge_catalogs(filetype,source=source,cat_fn=cat_fn,write=True)

if __name__ == '__main__':

    setup_logging()
    main()
