"""Call ./runbrick.sh."""

import os
from obiwan.batch import TaskManager,run_shell
import settings

ntasks = os.getenv('SLURM_NTASKS',1)
threads = os.getenv('OMP_NUM_THREADS',1)

with TaskManager(ntasks=ntasks) as tm:

    for brick in tm.iterate(settings.get_bricknames()):

        output = run_shell(['./runbrick.sh','--brick',brick,'--threads',threads,'--outdir',settings.output_dir,'--run',settings.run,
                        '--ran-fn',settings.randoms_fn,'--fileid',settings.fileid,'--rowstart',settings.rowstart,
                        '--skipid',settings.skipid,'--sim-blobs','--sim-stamp','tractor','--no-wise'])
        print(output)
