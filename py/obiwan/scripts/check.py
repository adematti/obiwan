"""
Script to check all runs have completed by verifying all files exist on disk (and optionally can be opened).

For details, run::

    python check.py --help

"""

import argparse
import logging
from obiwan.analysis import SimCatalog,RunCatalog
from obiwan import find_file,utils,setup_logging

logger = logging.getLogger('check')

def main(args=None):

    parser = argparse.ArgumentParser(description='Check')
    parser.add_argument('-d', '--outdir', dest='output_dir', help='Output base directory, default "."')
    parser.add_argument('--filetype', type=str, default='tractor',
                        help='Filetype to search for.')
    parser.add_argument('--source', type=str, choices=['obiwan','legacypipe'], default='obiwan',
                        help='legacypipe or obiwan file structure?')
    parser.add_argument('--read', action='store_true', default=False,
                        help='Try read file from disk?')
    runlist_template = 'runlist.txt'
    parser.add_argument('--write-list', nargs='?', type=str, default=False, const=True,
                        help='Write missing run list to file name. If file name is not provided, defaults to %s.' % runlist_template
                         + ' This run list can be used to instantiate RunCatalog through RunCatalog.from_runlist(), in order to iterate easily through the runs.')
    RunCatalog.get_input_parser(parser=parser)
    opt = parser.parse_args(args=utils.get_parser_args(args))
    runinput = RunCatalog.from_input_cmdline(opt)
    runoutput = RunCatalog.from_output_cmdline(opt,force_from_disk=True,filetype=opt.filetype,source=opt.source)
    if opt.read:
        mask = runoutput.trues()
        for irun,run in enumerate(runoutput):
            fn = find_file(opt.output_dir,opt.filetype,brickname=run.brickname,source=opt.source,**run.kwargs_file)
            try:
                SimCatalog(fn)
            except OSError:
                logger.info('File %s exists but yields OSError.' % fn)
                mask[irun] = False
        runoutput = runoutput[mask]

    mask = runinput.in1d(runoutput)
    # remove runs with empty input randoms file.
    if opt.source == 'obiwan':
        for irun,run in enumerate(runinput):
            if not mask[irun]:
                fn = find_file(opt.output_dir,'randoms',brickname=run.brickname,source=opt.source,**run.kwargs_file)
                try:
                    randoms = SimCatalog(fn)
                    if randoms.size == 0: mask[irun] = True
                except OSError:
                    pass

    runinput = runinput[~mask]
    if mask.all():
        logger.info('All runs in output!')
    else:
        logger.info('%d runs missing.' % runinput.size)
        for run in runinput:
            txt = ' '.join(['%s:%s' % (field,run.get(field)) for field in runinput.fields])
            logger.info('Run %s missing.' % txt)
        if opt.write_list:
            if not isinstance(opt.write_list,str):
                opt.write_list = runlist_template
            runinput.write_list(opt.write_list)

    return runinput

if __name__ == '__main__':

    setup_logging()
    main()
