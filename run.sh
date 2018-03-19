#!/bin/bash

#SBATCH --job-name=MC-DigiCam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono
#SBATCH --output=$HOME/output/slurm-%A.out
#SBATCH --error=$HOME/output/slurm-%A.err
#SBATCH --mail-user=alispach
#SBATCH --mail-type=END
#SBATCH --mem=2000
#SBATCH --verbose
# SBATCH --export=HOME


echo "-----START JOB $SLURM_JOB_ID at `date`-----"

# module load intel/2017a Python/3.5
srun source $HOME/.anaconda3/envs/digicamtoy/bin/activate
srun python produce_data.py -y config_files/commissioning/ac_$1_dc_$2_id_$3.yml


echo "-----END JOB $SLURM_JOB_ID at `date`-----"

# EOF
