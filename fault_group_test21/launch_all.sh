#!/bin/sh

dimx=8192
dimy=512

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
./fault_group_test18 $i
cat fault_group_test18.cu | sed "s/#define STRIDE_64K 256/#define STRIDE_64K $i/" > temp_fault_group_test18.txt
done

echo " "

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test4 $i
done

echo " "

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt
done

#sh batch_prof.sh

