#!/bin/bash

#SBATCH --job-name=MC-DigiCam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.err
#SBATCH --mail-user=cyril.alispach@unige.ch --mail-type=END
#SBATCH --mem=2000
#SBATCH --verbose


echo "-----START JOB at `date`-----"

module load intel/2017a Python/3.5
srun source /home/alispach/.anaconda3/envs/digicamtoy/bin/activate
srun python produce_data.py -y config_files/commissioning/ac_$1_dc_$2_id_$3.yml


echo "-----END   JOB at `date`-----"

# EOF
