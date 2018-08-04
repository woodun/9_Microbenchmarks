#!/bin/sh

i=0

###############################1024
echo "###########1024:"

dimx=128
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=64
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=32
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=16
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


dimx=8
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=4
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt
###############################32
echo "###########32:"

dimx=4096
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2048
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1024
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=512
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=256
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=128
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=64
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=32
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=16
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=8
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=4
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt




#sh batch_prof.sh












