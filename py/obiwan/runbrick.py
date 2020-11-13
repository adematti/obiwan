"""**Obiwan** main executable, equivalent of ``legacypipe.runbrick``."""

import os
import sys
import logging
from legacypipe import runbrick
from legacypipe.utils import RunbrickError, NothingToDoError
from obiwan import SimCatalog,utils
from obiwan.utils import setup_logging

logger = logging.getLogger('obiwan.runbrick')

def get_parser():
    """
    Append **Obiwan** arguments to those of ``legacypipe.runbrick``.

    Returns
    -------
    parser : argarse.ArgumentParser
        Parser.

    args_runbrick : list
        List of **legacypipe**-specific arguments.
    """
    import argparse
    de = ('Main "Obiwan" script for the Legacy Survey (DECaLS, MzLS, Bok) data reductions.')
    ep = """
    e.g., to run a small field containing a cluster:
python -u obiwan/runbrick.py --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle"""
    parser = argparse.ArgumentParser(description=de,epilog=ep,add_help=False,parents=[runbrick.get_parser()])
    args_runbrick = utils.get_parser_args(parser,exclude=['verbose','help'])
    #Obiwan arguments
    group = parser.add_argument_group('Obiwan', 'Obiwan-specific arguments')
    group.add_argument('--subset', type=int, default=0,
                        help='COSMOS subset number [0 to 4, 10 to 12], only used if --run cosmos')
    group.add_argument('--ran-fn', default=None, help='Randoms filename; if not provided, run equivalent to legacypipe.runbrick')
    group.add_argument('--fileid', type=int, default=0, help='Index of ran-fn')
    group.add_argument('--rowstart', type=int, default=0,
                        help='Zero indexed, row of ran-fn, after it is cut to brick, to start on')
    group.add_argument('--nobj', type=int, default=-1,
                        help='Number of objects to inject in the given brick; if -1, all objects in ran-fn are added')
    group.add_argument('--skipid', type=int, default=0, help='Inject collided objects from ran-fn of previous skipid-1 run.\
                       In this case, no cut based on --nobj and --rowstart is applied.')
    group.add_argument('--col-radius', type=float, default=5., help='Collision radius in arcseconds, used to define collided simulated objects.\
                        Ignore if negative')
    group.add_argument('--sim-stamp', type=str, choices=['tractor','galsim'], default='tractor', help='Method to simulate objects')
    group.add_argument('--add-sim-noise', action="store_true", default=False, help='Add noise to simulated sources?')
    group.add_argument('--image-eq-model', action="store_true", default=False, help='Set image ivar by model only (ignore real image ivar)?')
    group.add_argument('--sim-blobs', action='store_true', default=False,
                        help='Process only the blobs that contain simulated sources')
    group.add_argument('--seed', type=int, default=None, help='Random seed to add noise to injected sources of ran-fn.')
    return parser,args_runbrick

def get_runbrick_kwargs(args_runbrick,**opt):
    """
    Convert ``obiwan.runbrick`` command line options into ``survey`` and ``**kwargs`` for ``run_brick()``.

    Wraps ``legacypipe.runbrick.get_runbrick_kwargs()``.

    Parameters
    ----------
    args_runbrick : list
        List of **legacypipe**-specific arguments.

    opt : dict
        Dictionary of the command line options to ``obiwan.runbrick``.

    Returns
    -------
    survey : LegacySurveySim instance
        Survey, without ``simcat``.

    kwargs : dict, default={}
        Arguments for ``legacypipe.runbrick.run_brick()`` following::

            run_brick(brickname, survey, **kwargs)

    """
    opt['kwargs_file'] = {key:opt[key] for key in ['fileid','rowstart','skipid']}

    kwargs_survey = {key:opt[key] for key in \
                          ['sim_stamp','add_sim_noise','image_eq_model','seed','kwargs_file',
                           'survey_dir','output_dir','cache_dir','subset']}
    from obiwan.kenobi import get_survey
    survey = get_survey(opt.get('run',None),**kwargs_survey)
    logger.info(survey)

    survey,kwargs = runbrick.get_runbrick_kwargs(survey,**{key:opt[key] for key in args_runbrick})

    return survey,kwargs

def run_brick(opt, survey, **kwargs):
    """
    Add ``simcat`` to ``survey``, run brick, and saves ``simcat``.

    Wraps ``legacypipe.runbrick.run_brick()``.

    Parameters
    ----------
    opt : Namespace
        Command line options for ``obiwan.runbrick``.

    survey : LegacySurveySim instance
        Survey, without ``simcat``.

    kwargs : dict, default={}
        Arguments for ``legacypipe.runbrick.run_brick()`` following::

            run_brick(brickname, survey, **kwargs)

    """
    # legacypipe-only run if opt.skipid==0 and random filename not provided
    if (not (opt.skipid>0)) and (opt.ran_fn is None):
        survey.simcat = None
        runbrick.run_brick(opt.brick, survey, **kwargs)
        return

    if opt.skipid>0:
        filename = survey.find_file('obiwan-randoms',output=True)
        simcat = SimCatalog(filename)
        simcat.cut(simcat.collided)
    else:
        simcat = SimCatalog(opt.ran_fn)
        simcat.fill_obiwan(survey=survey)
        simcat.cut(simcat.brickname == opt.brick)
        if opt.nobj>=0:
            simcat = simcat[opt.rowstart:opt.rowstart+opt.nobj]
            logger.info('Cutting to nobj = %d' % opt.nobj)
    logger.info('SimCatalog size = %d' % len(simcat))

    if opt.col_radius>0.:
        simcat.collided = simcat.mask_collisions(radius_in_degree=opt.col_radius/3600.)
    else:
        logger.info('Ignore collisions.')
        simcat.collided = simcat.falses()
    ncollided = simcat.collided.sum()
    mask_simcat = ~simcat.collided

    if ncollided>0:
        logger.info('Found %d collisions! You will have to run runbrick.py with --skipid = %d.' % (ncollided,opt.skipid+1))

    survey.simcat = simcat[mask_simcat]

    if len(survey.simcat) == 0:
        raise ValueError('Empty SimCatalog, aborting!')

    if opt.sim_blobs:
        logger.info('Fitting blobs of input catalog.')
        blobxy = zip(survey.simcat.bx,survey.simcat.by)
        kwargs.update(blobxy=blobxy)

    runbrick.run_brick(opt.brick, survey, **kwargs)
    simcat[mask_simcat] = survey.simcat

    ran_fn = survey.find_file('obiwan-randoms',brick=opt.brick,output=True)
    simcat.writeto(ran_fn,header=vars(opt))


def main(args=None):
    """
    Main routine which parses the optional inputs.

    Simple copy-paste from ``legacypipe.runbrick.main()``, simply changing::

        parser = get_parser()
        logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
        survey, kwargs = get_runbrick_kwargs(**optdict)
        debug('kwargs:', kwargs)
        run_brick(opt.brick, survey, **kwargs)

    to::

        parser, args_runbrick = get_parser()
        setup_logging(lvl)
        survey, kwargs = get_runbrick_kwargs(args_runbrick,**optdict)
        logger.debug('kwargs: %s' % kwargs)
        run_brick(opt, survey, **kwargs)

    Parameters
    ---------
    args : list, default=None
        To overload command line arguments.
    """
    import datetime
    from legacypipe.survey import get_git_version

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
    print('legacypipe git version:', get_git_version())
    if args is None:
        print('Command-line args:', sys.argv)
        cmd = 'python'
        for vv in sys.argv:
            cmd += ' {}'.format(vv)
        print(cmd)
    else:
        print('Args:', args)
    print()

    parser, args_runbrick = get_parser()
    parser.add_argument(
        '--ps', help='Run "ps" and write results to given filename?')
    parser.add_argument(
        '--ps-t0', type=int, default=0, help='Unix-time start for "--ps"')

    opt = parser.parse_args(args=args)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    ps_file = optdict.pop('ps', None)
    ps_t0   = optdict.pop('ps_t0', 0)
    verbose = optdict.pop('verbose')

    survey, kwargs = get_runbrick_kwargs(args_runbrick,**optdict)

    if kwargs in [-1, 0]:
        return kwargs
    kwargs.update(command_line=' '.join(sys.argv))

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    setup_logging(lvl)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    if ps_file is not None:
        import threading
        from collections import deque
        from legacypipe.utils import run_ps_thread
        ps_shutdown = threading.Event()
        ps_queue = deque()
        def record_event(msg):
            from time import time
            ps_queue.append((time(), msg))
        kwargs.update(record_event=record_event)
        if ps_t0 > 0:
            record_event('start')

        ps_thread = threading.Thread(
            target=run_ps_thread,
            args=(os.getpid(), os.getppid(), ps_file, ps_shutdown, ps_queue),
            name='run_ps')
        ps_thread.daemon = True
        print('Starting thread to run "ps"')
        ps_thread.start()

    logger.debug('kwargs: %s' % kwargs)

    rtn = -1
    try:
        run_brick(opt, survey, **kwargs)
        rtn = 0
    except NothingToDoError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = 0
    except RunbrickError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = -1

    if ps_file is not None:
        # Try to shut down ps thread gracefully
        ps_shutdown.set()
        print('Attempting to join the ps thread...')
        ps_thread.join(1.0)
        if ps_thread.isAlive():
            print('ps thread is still alive.')

    return rtn


if __name__ == '__main__':

    from astrometry.util.ttime import Time,CpuMeas,MemMeas
    Time.add_measurement(CpuMeas)
    Time.add_measurement(MemMeas)
    import time as time_builtin
    logger.info('runbrick.py started at %s' % time_builtin.strftime("%Y-%m-%d %H:%M:%S"))
    main()
    logger.info('runbrick.py finished at %s' % time_builtin.strftime("%Y-%m-%d %H:%M:%S"))
