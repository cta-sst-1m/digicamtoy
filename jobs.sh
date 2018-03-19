#!/bin/bash
N_AC=5
N_DC=30
N_FILES=100

for (( i=0;i<$N_AC;i++)); do
    for (( j=0;j<$N_DC;j++)); do
        for (( k=0;k<$N_FILES;k++)); do
        sbatch run.sh ${i} ${j} ${k}
        done
    done
done
