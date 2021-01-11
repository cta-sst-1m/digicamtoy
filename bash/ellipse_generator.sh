#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --partition=mono-EL7,mono-shared-EL7,dpnc-EL7
#SBATCH --mem=8G
#SBATCH --output=/home/%u/job_logs/%x-%A_%a.out
#SBATCH --job-name='toy-ellipses'
#SBATCH --array=0-4

source $HOME/.miniconda3/bin/activate digicampipe

nsb=('0.0' '0.04' '0.08' '0.2' '0.6')
nsb=${nsb[$SLURM_ARRAY_TASK_ID]}
n_events=1000

for i in 1 2 3 4 5 6 7 8 9 10
do
    output='/sst1m/MC/digicamtoy/ellipses/ellipse_images_'$nsb'GHz_v12_id_'$i'.hdf5'
    python produce_ellipse.py --n_images=$n_events --output=$output --nsb=$nsb
done