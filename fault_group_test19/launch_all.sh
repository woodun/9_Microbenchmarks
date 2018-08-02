#!/bin/sh

dimx=8192
dimy=512
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

dimx=512
dimy=512
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

dimx=1
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

dimx=1
dimy=64
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

dimx=2
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

echo " "

dimx=8192
dimy=512
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=512
dimy=512
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1
dimy=32
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1
dimy=64
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2
dimy=32
sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


#sh batch_prof.sh