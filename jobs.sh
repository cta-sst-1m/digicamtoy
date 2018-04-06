#!/bin/bash

AC_START=0
DC_START=1
FILE_ID_START=0
AC_END=29
DC_END=29
FILE_ID_END=199

for (( i=$AC_START;i<=$AC_END;i++)); do
    for (( j=$DC_START;j<=$DC_END;j++)); do
        for (( k=$FILE_ID_START;k<=$FILE_ID_END;k++)); do
        sbatch run.sh ${i} ${j} ${k}
        done
    done
done
