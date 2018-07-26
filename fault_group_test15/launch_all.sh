#!/bin/sh

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
./fault_group_test15 $i
done

echo " "

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
nvprof --profile-from-start off --print-gpu-trace --log-file prof232size8gpage$i.txt --csv ./fault_group_test15 $i
done

echo " "

sh batch_prof.sh

