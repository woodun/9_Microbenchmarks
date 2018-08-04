#!/bin/sh

i=0

###############################1024
echo "###########1024:"

dimx=128
dimy=1024
cat temp_fault_group_test20.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test20.cu
make > dump_make.txt
./fault_group_test20
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test20 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt





#sh batch_prof.sh












