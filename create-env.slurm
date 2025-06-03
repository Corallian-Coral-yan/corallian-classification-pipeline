#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="[Corallian] Create environment"
#SBATCH --output="create-env.out"
#SBATCH --error="create-env.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin_lim@dlsu.edu.ph  

## For more `sbatch` options, use `man sbatch` in the HPC, or go to https://slurm.schedmd.com/sbatch.html.

## Set stack size to unlimited.
ulimit -s unlimited

## Benchmarking.
start_time=$(date +%s.%N)

## Print job parameters.
echo "Submitted on $(date)"
echo "JOB PARAMETERS"
echo "SLURM_JOB_ID          : ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME        : ${SLURM_JOB_NAME}"
echo "SLURM_JOB_NUM_NODES   : ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST    : ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS          : ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE : ${SLURM_NTASKS_PER_NODE}"
echo "SLURM_MEM_PER_NODE    : ${SLURM_MEM_PER_NODE}"

## Create a unique temporary folder in the node. Using a local temporary folder usually results in faster read/write for temporary files.
custom_tmpdir="yes"

if [[ $custom_tmpdir == "yes" ]]; then
   JOB_TMPDIR=/tmp/${USER}/SLURM_JOB_ID/${SLURM_JOB_ID}
   mkdir -p ${JOB_TMPDIR}
   export TMPDIR=${JOB_TMPDIR}
   echo "TMPDIR                : $TMPDIR"
fi

## Reset modules.
module purge
module load anaconda

## Main job. Run your codes and executables here; `srun` is optional.

# Remove existing environment (optional, use only if you want a fresh install)

# Create the Conda environment from environment.yml
conda create -y -n corallian-env

CONDA_OVERRIDE_CUDA="11.8" mamba install -y -n corallian-env -c pytorch -c nvidia --file requirements.txt

## Flush the TMPDIR.
if [[ $custom_tmpdir == "yes" ]]; then
   rm -rf $TMPDIR
   echo "Cleared the TMPDIR (${TMPDIR})"
fi

## Benchmarking
end_time=$(date +%s.%N)
echo "Finished on $(date)"
run_time=$(python -c "print($end_time - $start_time)")
echo "Total runtime (sec): ${run_time}"
