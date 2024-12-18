#!/bin/bash
#SBATCH --job-name=exp_01_p2 # short name for your job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=martin.robins@mi.unc.edu.ar
#SBATCH --output=slurm-%x.%j.out # %j job id, ½x job name
#SBATCH --error=slurm-%x.%j.err
#SBATCH --partition=multi
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=1-00:00           # total run time limit (<days>-<hours>:<minutes>)

. /etc/profile
module purge
ulimit -c unlimited  # core dump
ulimit -s unlimited  # stack

echo " job \"${SLURM_JOB_NAME}\""
echo " id: ${SLURM_JOB_ID}"
echo " partition: ${SLURM_JOB_PARTITION}"
echo " node(s): ${SLURM_JOB_NODELIST}"
echo " gres: ${SBATCH_GRES}"
echo " gpus: ${SBATCH_GPUS}"
date +"start %F - %T"
echo ""

source ${HOME}/.bashrc

cd ${HOME}/
micromamba activate env_thalamus

cd ${HOME}/01\ PI-Thalamus/PI-Thalamus-RNS/Paper_iESPnet_pruebas/05-Train-Test/

srun python3 example_run_test_v1_p2.py
