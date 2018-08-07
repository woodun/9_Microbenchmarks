#!/bin/sh

dimx=64
dimy=1024

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
####### 8192 * 512 / 32 * 65536 = 8589934592 (8g) ####### 1024 * 512 / 32 * 524288  = 8589934592 (8g) ####### 512 * 512 / 32 * 1048576  = 8589934592 (8g) 
####### 1024 * 1024 / 32 * 262144  = 8589934592 (8g) ####### 512 * 1024 / 32 * 524288  = 8589934592 (8g) 
####### 1024 * 256 / 32 * 1048576  = 8589934592 (8g) ####### 512 * 256 / 32 * 2097152  = 8589934592 (8g) ####### 256 * 256 / 32 * 4194304  = 8589934592 (8g) 
####### 1024 * 128 / 32 * 2097152  = 8589934592 (8g) ####### 512 * 128 / 32 * 4194304  = 8589934592 (8g) ####### 256 * 128 / 32 * 8388608  = 8589934592 (8g) 
####### 4096 * 64 / 32 * 1048576  = 8589934592 (8g) ####### 512 * 64 / 32 * 8388608  = 8589934592 (8g) ####### 256 * 64 / 32 * 16777216  = 8589934592 (8g) ####### 2 * 32 / 32 * 4294967296  = 8589934592 (8g) ####### 
####### 8192 * 32 / 32 * 1048576  = 8589934592 (8g) ####### 1024 * 32 / 32 * 8388608  = 8589934592 (8g) ####### 512 * 32 / 32 * 16777216  = 8589934592 (8g) ####### 2 * 32 / 32 * 4294967296  = 8589934592 (8g) #######