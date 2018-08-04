#!/bin/sh

i=0

###############################1024
echo "###########1024:\n"

dimx=2048
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1024
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=512
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=256
dimy=1024
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

###############################512
echo "###########512:\n"

dimx=4096
dimy=512
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2048
dimy=512
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1024
dimy=512
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=512
dimy=512
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


###############################256
echo "###########256:\n"

dimx=8192
dimy=256
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=4096
dimy=256
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2048
dimy=256
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=1024
dimy=256
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


###############################128
echo "###########128:\n"

dimx=16384
dimy=128
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=8192
dimy=128
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=4096
dimy=128
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=2048
dimy=128
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


###############################64
echo "###########64:\n"

dimx=32768
dimy=64
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=16384
dimy=64
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=8192
dimy=64
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=4096
dimy=64
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


###############################32
echo "###########32:\n"

dimx=65536
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=32768
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=16384
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt

dimx=8192
dimy=32
cat temp_fault_group_test19.cu | sed "s/#define dimx 512/#define dimx $dimx/" | sed "s/#define dimy 512/#define dimy $dimy/" > fault_group_test19.cu
make > dump_make.txt
./fault_group_test19
nvprof --profile-from-start off --print-gpu-trace --log-file prof$dimx${dimy}size8gpage$i.txt --csv ./fault_group_test19 > dump_profile.txt

sh profhandler.sh prof$dimx${dimy}size8gpage$i.txt


#sh batch_prof.sh












