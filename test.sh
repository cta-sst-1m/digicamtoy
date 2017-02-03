#!/bin/bash
# units in p.e, ns, GHz, ns^{-1}

N_CHERENKOV=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18')
NSB_RATE=('0.0' '0.003' '0.0' '0.003')
XT=('0.0' '0.0' '0.06' '0.06')
FILENAME=('toy_data_' 'toy_data_dark_' 'toy_data_xt_' 'toy_data_xt_dark_')

N_FILENAME=${#FILENAME[@]}
N_NSB_RATE=${#NSB_RATE[@]}
N_N_CHERENKOV=${#N_CHERENKOV[@]}
N_XT=${#XT[@]}

for (( i=0;i<$N_FILENAME;i++)); do
for (( j=0;j<$N_N_CHERENKOV;j++)); do

    python hd5_maker.py --crosstalk ${XT[${i}]} --photon_times -50 50 4 --photon_range ${N_CHERENKOV[${j}]} 0 0  --nsb_range ${NSB_RATE[${i}]} 0 0 --poisson_signal 0 -f ${FILENAME[${i}]}${N_CHERENKOV[${j}]} -d data_calibration_cts/


done
done