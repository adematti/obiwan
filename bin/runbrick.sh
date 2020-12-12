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

# Can detect Cori KNL node (96 GB) via:
# grep -q "Xeon Phi" /proc/cpuinfo && echo Yes

#cd /src/obiwan/py
cd ../py

# Defaults
outdir=${OUTPUT_DIR}
rowstart=0
skipid=0
fileid=0
threads=1
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
  --threads)
  threads="$2"
  shift # past argument
  shift # past value
  ;;
  *)
  others+=" $1" # save it in an array for later
  shift # past argument
  ;;
esac
done

# 128 GB / Cori Haswell node = 134217728 kbytes, 32 cores
maxmem=134217728
let usemem=${maxmem}*${threads}/32

ulimit -Sv "$usemem"

mkdir -p "${output_dir}"

rsname="file${fileid}_rs${rowstart}_skip${skipid}"
bri=$(echo "$brick" | head -c 3)
logfn="${outdir}/logs/${bri}/${brick}/${rsname}/${brick}.log"
mkdir -p "${logfn%/*}"

echo "Logging to: $logfn"
echo "Running on $(hostname)"

echo -e "\n\n\n" >> "$logfn"
echo "-----------------------------------------------------------------------------------------" >> "$logfn"
echo "PWD: $(pwd)" >> "$logfn"
echo >> "$logfn"
echo "Environment:" >> "$logfn"
set | grep -v PASS >> "$logfn"
echo >> "$logfn"
ulimit -a >> "$logfn"
echo >> "$logfn"

echo -e "\nStarting on $(hostname)\n" >> "$logfn"
echo "-----------------------------------------------------------------------------------------" >> "$logfn"

python -O obiwan/runbrick.py \
      --brick "$brick" \
      --threads "${threads}" \
      --outdir "${outdir}" \
      --fileid "${fileid}" \
      --rowstart "${rowstart}" \
      --skipid "${skipid}" \
      --ps \
      --ps-t0 "$(date "+%s")" \
      $others
      >> "$logfn" 2>&1
