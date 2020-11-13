#!/bin/bash

# Script for running the obiwan code within a Shifter container at NERSC

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# Limit memory to avoid killing the whole MPI job...
ncores=8

# 128 GB / Cori Haswell node = 134217728 kbytes, 32 cores
maxmem=134217728
let usemem=${maxmem}*${ncores}/32

# Can detect Cori KNL node (96 GB) via:
# grep -q "Xeon Phi" /proc/cpuinfo && echo Yes

ulimit -Sv "$usemem"

cd /src/obiwan/py

# Defaults
outdir=${OUTPUT_DIR}
ran_fn=${RANDOMS_FN}
rowstart=0
skipid=0
fileid=0
# Cmd line arguments
others=""
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
  --brick)
  brick="$2"
  shift # past argument
  shift # past value
  ;;
  --outdir)
  outdir="$2"
  shift # past argument
  shift # past value
  ;;
  --ran-fn)
  ran_fn="$2"
  shift # past argument
  shift # past value
  ;;
  --fileid)
  fileid="$2"
  shift # past argument
  shift # past value
  ;;
  --rowstart)
  rowstart="$2"
  shift # past argument
  shift # past value
  ;;
  --skipid)
  skipid="$2"
  shift # past argument
  shift # past value
  ;;
  *)
  others+=" $1" # save it in an array for later
  shift # past argument
  ;;
esac
done

mkdir -p "${output_dir}"
mkdir -p "${ran_fn%/*}"

rsname="file${fileid}_rs${rowstart}_skip${skipid}"
bri=$(echo "$brick" | head -c 3)
log="${outdir}/logs/${bri}/${rsname}/${brick}.log"
mkdir -p "${log%/*}"
psfn="${outdir}/metrics/${bri}/${rsname}/ps-${brick}-${SLURM_JOB_ID}.fits"
mkdir -p "${psfn%/*}"

#log=''
echo "Logging to: $log"
echo "Running on $(hostname)"

echo -e "\n\n\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"
echo "PWD: $(pwd)" >> "$log"
echo >> "$log"
echo "Environment:" >> "$log"
set | grep -v PASS >> "$log"
echo >> "$log"
ulimit -a >> "$log"
echo >> "$log"

echo -e "\nStarting on $(hostname)\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"

python -O obiwan/runbrick.py \
      --brick "$brick" \
      --threads "${ncores}" \
      --outdir "${outdir}" \
      --ran-fn "${ran_fn}" \
      --fileid "${fileid}" \
      --rowstart "${rowstart}" \
      --skipid "${skipid}" \
      --ps "${psfn}" \
      --ps-t0 "$(date "+%s")" \
      "$others"
      >> "$log" 2>&1
