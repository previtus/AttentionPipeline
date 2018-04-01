#!/bin/bash

USER="KilgoreTrout"
ADDRESS="StreetsOfNewYork"
KEY_PATH=""

# gpu[001-003], start at 0 = 900[0] and 910[0]
gpu_start=21
gpu_end=21
i=9
for (( c=gpu_start; c<=gpu_end; c++ ))
do
    num=$( printf '%03d' $c )
    let "first_server_num=9000+$i"
    let "second_server_num=9100+$i"

    # echo " ... "
    ssh -i $KEY_PATH -N -f -L  $first_server_num:gpu$num:8123 $USER@$ADDRESS
    ssh -i $KEY_PATH -N -f -L  $second_server_num:gpu$num:8666 $USER@$ADDRESS

    let "i += 1"

done