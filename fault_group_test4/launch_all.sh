#!/bin/sh

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152
do
./fault_group_test4 $i
done

echo " "

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152
do
nvprof --profile-from-start off --print-gpu-trace --log-file prof832size8gpage$i.txt --csv ./fault_group_test4 $i
done

sh batch_prof.sh

