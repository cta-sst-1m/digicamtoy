#!/bin/bash

#SBATCH --job-name=MC-DigiCam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --partition=mono
#SBATCH --output=/home/alispach/output/slurm-%A.out
#SBATCH --error=/home/alispach/output/slurm-%A.err
#SBATCH --mail-user=alispach
#SBATCH --mail-type=END
#SBATCH --mem=1000 # 8000
#SBATCH --verbose

echo "-----START JOB $SLURM_JOB_ID at `date`-----"

# srun source /home/alispach/.anaconda3/envs/digicamtoy/bin/activate
source $HOME/.anaconda3/envs/digicamtoy/bin/activate digicamtoy
# srun python produce_data.py -y config_files/commissioning/ac_$1_dc_$2_id_$3.yml
python generate.py

echo "-----END JOB $SLURM_JOB_ID at `date`-----"

# EOF
