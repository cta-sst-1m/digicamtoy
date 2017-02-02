#!/bin/bash
# units in p.e, ns, GHz, ns^{-1}

N_CHERENKOV=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40')
NSB_RATE=('0.0' '0.0003')
XT=('0.0' '0.06')

N_NSB_RATE=${#NSB_RATE[@]}
N_N_CHERENKOV=${#N_CHERENKOV[@]}

MC_NUMBER=1

FILENAME='toy_data_'
#FILENAME='toy_data_poisson_signal_'

for (( j=0;j<$N_N_CHERENKOV;j++)); do

    python hd5_maker.py --crosstalk ${XT[0]} --photon_times -100 100 4 --photon_range ${N_CHERENKOV[${j}]} 0 0  --nsb_range ${NSB_RATE[0]} 0 0 --poisson_signal 0 -f $FILENAME${N_CHERENKOV[${j}]} -d data_calibration_cts/
    #python hd5_maker.py --crosstalk ${XT[${0}]} --photon_times -100 100 4 --photon_range ${N_CHERENKOV[${j}]} 0 0  --nsb_range ${NSB_RATE[${1}]} 0 0 --poisson_signal 0 -f $FILENAME${N_CHERENKOV[${j}]} -d data_calibration_cts/
    #python hd5_maker.py --crosstalk ${XT[${1}]} --photon_times -100 100 4 --photon_range ${N_CHERENKOV[${j}]} 0 0  --nsb_range ${NSB_RATE[${1}]} 0 0 --poisson_signal 0 -f $FILENAME${N_CHERENKOV[${j}]} -d data_calibration_cts/

done