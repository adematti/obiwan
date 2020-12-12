"""Run directly obiwan.runbrick.main."""

import os
import time
from obiwan import runbrick,RunCatalog,get_randoms_id
from obiwan.batch import TaskManager
import settings

ntasks = os.getenv('SLURM_NTASKS',1)
threads = os.getenv('OMP_NUM_THREADS',1)

runcat = RunCatalog.from_input_cmdline(dict(brick=settings.get_bricknames(),
                                        fileid=settings.fileid,rowstart=settings.rowstart,skipid=settings.skipid))
runcat.seed = runcat.index()*42

with TaskManager(ntasks=ntasks) as tm:

    for run in tm.iterate(runcat):

        log_fn = os.path.join(settings.output_dir,'logs',run.brickname[:3],run.brickname,
                        get_randoms_id(fileid=run.fileid,rowstart=run.rowstart,skipid=run.skipid),'%s.log' % run.brickname)

        command = ['--brick',run.brickname,'--threads',threads,'--outdir',settings.output_dir,'--run',settings.run,
                        '--ran-fn',settings.randoms_fn,'--fileid',run.fileid,'--rowstart',run.rowstart,
                        '--skipid',run.skipid,'--sim-blobs','--sim-stamp','tractor','--no-wise','--no-write',
                        #'--log-fn',log_fn,
                        '--ps','--ps-t0',int(time.time()),'--seed',run.seed]

        print('Launching ' + ' '.join(map(str,command)))

        runbrick.main(command)
