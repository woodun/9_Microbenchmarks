#!/bin/sh

dimx=8
dimy=32

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
cat temp_fault_group_test21.cu | sed "s/#define STRIDE_64K 256/#define STRIDE_64K $i/" | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test21.cu
make > dump_make.txt
./fault_group_test21
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test21 > dump_profile.txt
done

echo " "

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt
done

#sh batch_prof.sh