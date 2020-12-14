"""**Obiwan** main executable, equivalent of ``legacypipe.runbrick``."""

import os
import sys
import logging
import numpy as np
from legacypipe import runbrick
from legacypipe.utils import RunbrickError, NothingToDoError
from obiwan import SimCatalog,utils,setup_logging

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

    args_runbrick = utils.get_parser_dests(parser,exclude=['verbose','help'])
    # Obiwan arguments
    group = parser.add_argument_group('Obiwan', 'Obiwan-specific arguments')
    parser.add_argument('--log-fn', type=str, default=None, help='Log to given filename instead of stdout')
    group.add_argument('--subset', type=int, default=0,
                        help='COSMOS subset number [0 to 4, 10 to 12], only used if --run cosmos')
    group.add_argument('--ran-fn', default=None, help='Randoms filename; if not provided, run equivalent to legacypipe.runbrick')
    group.add_argument('--fileid', type=int, default=0, help='Index of ran-fn')
    group.add_argument('--rowstart', type=int, default=0,
                        help='Zero indexed, row of ran-fn, after it is cut to brick, to start from')
    group.add_argument('--nobj', type=int, default=-1,
                        help='Number of objects to inject in the given brick; if -1, all objects in ran-fn are added')
    group.add_argument('--skipid', type=int, default=0, help='Inject collided objects from ran-fn of previous skipid-1 run.\
                       In this case, no cut based on --nobj and --rowstart is applied')
    group.add_argument('--col-radius', type=float, default=5., help='Collision radius in arcseconds, used to define collided simulated objects.\
                        Ignore if negative')
    group.add_argument('--sim-stamp', type=str, choices=['tractor','galsim'], default='tractor', help='Method to simulate objects')
    group.add_argument('--add-sim-noise', type=str, choices=['gaussian','poisson'], default=False, help='Add noise from the simulated source to the image.')
    group.add_argument('--image-eq-model', action='store_true', default=False, help='Set image ivar by model only (ignore real image ivar)?')
    group.add_argument('--sim-blobs', action='store_true', default=False,
                        help='Process only the blobs that contain injected sources')
    group.add_argument('--seed', type=int, default=None, help='Random seed to add noise to injected sources of ran-fn')
    return parser,args_runbrick

def get_runbrick_kwargs(args_runbrick, **opt):
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
    from obiwan.kenobi import get_randoms_id
    opt['kwargs_file'] = get_randoms_id.as_dict(**opt)

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
    # legacypipe-only run if opt.skipid == 0 and random filename not provided
    if (not (opt.skipid > 0)) and (opt.ran_fn is None):
        survey.simcat = None
        runbrick.run_brick(opt.brick, survey, **kwargs)
        return

    if opt.skipid > 0:
        filename = survey.find_file('randoms',output=True)
        simcat = SimCatalog(filename)
        simcat.cut(simcat.collided)
    else:
        simcat = SimCatalog(opt.ran_fn)
        simcat.fill_obiwan(survey=survey)
        simcat.cut(simcat.brickname == opt.brick)
        if opt.nobj >= 0:
            simcat = simcat[opt.rowstart:opt.rowstart+opt.nobj]
            logger.info('Cutting to nobj = %d' % opt.nobj)
    logger.info('SimCatalog size = %d' % len(simcat))

    if opt.col_radius > 0.:
        simcat.collided = simcat.mask_collisions(radius_in_degree=opt.col_radius/3600.)
    else:
        logger.info('Ignore collisions.')
        simcat.collided = simcat.falses()
    ncollided = simcat.collided.sum()
    mask_simcat = ~simcat.collided

    if ncollided > 0:
        logger.info('Found %d collisions! You will have to run runbrick.py with --skipid = %d.' % (ncollided,opt.skipid+1))

    survey.simcat = simcat[mask_simcat]
    ran_fn = survey.find_file('randoms',brick=opt.brick,output=True)

    if len(survey.simcat) == 0:
        # write catalog to ease run checks
        survey.simcat.writeto(ran_fn,header=vars(opt))
        raise ValueError('Empty SimCatalog, aborting!')

    if opt.sim_blobs:
        logger.info('Fitting blobs of input catalog.')
        blobradec = np.array([survey.simcat.ra,survey.simcat.dec]).T
        kwargs.update(blobradec=blobradec)

    runbrick.run_brick(opt.brick, survey, **kwargs)
    simcat[mask_simcat] = survey.simcat

    simcat.writeto(ran_fn,header=vars(opt))


def main(args=None):
    """
    Main routine which parses the optional inputs.

    Simple copy-paste from ``legacypipe.runbrick.main()``, main changes::

        parser = get_parser()
        survey, kwargs = get_runbrick_kwargs(**optdict)
        run_brick(opt.brick, survey, **kwargs)

    to::

        parser, args_runbrick = get_parser()
        survey, kwargs = get_runbrick_kwargs(args_runbrick,**optdict)
        run_brick(opt, survey, **kwargs)

    Parameters
    ---------
    args : list, default=None
        To overload command line arguments.
    """
    setup_logging('info')
    args = utils.get_parser_args(args=args)
    if args is None:
        logger.info('command-line args: %s' % sys.argv)
    else:
        logger.info('args: %s' % args)

    parser, args_runbrick = get_parser()
    parser.add_argument('--ps', nargs='?', type=str, default=False, const=True,
                        help='Run "ps" and write results to filename. '\
                        'If filename is not provided, used the default filename (in "metrics")')
    parser.add_argument('--ps-t0', type=int, default=0, help='Unix-time start for "--ps"')

    opt = parser.parse_args(args=args)
    optdict = vars(opt)
    ps_fn = optdict.pop('ps', False)
    ps_t0 = optdict.pop('ps_t0', 0)
    verbose = optdict.pop('verbose')
    log_fn = optdict.pop('log_fn')

    if verbose == 0:
        level = 'info'
    else:
        level = 'debug'

    setup_logging(level,filename=log_fn)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(logging.WARNING)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    survey, kwargs = get_runbrick_kwargs(args_runbrick,**optdict)

    if kwargs in [-1, 0]:
        return kwargs
    kwargs.update(command_line=' '.join(sys.argv))

    if opt.plots:
        if opt.plot_base is not None:
            utils.mkdir(os.path.dirname(opt.plot_base))
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    if ps_fn:
        if not isinstance(ps_fn,str): ps_fn = survey.find_file('ps',brick=opt.brick,output=True)
        utils.mkdir(os.path.dirname(ps_fn))
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
            args=(os.getpid(), os.getppid(), ps_fn, ps_shutdown, ps_queue),
            name='run_ps')
        ps_thread.daemon = True
        logger.info('Starting thread to run "ps"')
        ps_thread.start()

    logger.debug('kwargs: %s' % kwargs)

    toret = -1
    try:
        run_brick(opt, survey, **kwargs)
        toret = 0
    except NothingToDoError as e:
        if hasattr(e, 'message'):
            logger.info(e.message)
        else:
            logger.info(e)
        toret = 0
    except RunbrickError as e:
        if hasattr(e, 'message'):
            logger.info(e.message)
        else:
            logger.info(e)
        toret = -1

    if ps_fn:
        # Try to shut down ps thread gracefully
        ps_shutdown.set()
        logger.info('Attempting to join the ps thread...')
        ps_thread.join(1.0)
        if ps_thread.isAlive():
            logger.info('ps thread is still alive.')

    return toret


if __name__ == '__main__':

    from astrometry.util.ttime import Time,CpuMeas,MemMeas
    Time.add_measurement(CpuMeas)
    Time.add_measurement(MemMeas)
    import time
    setup_logging('info')
    logger.info('runbrick.py started at %s' % time.strftime("%Y-%m-%d %H:%M:%S"))
    main()
    logger.info('runbrick.py finished at %s' % time.strftime("%Y-%m-%d %H:%M:%S"))
