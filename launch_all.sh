#!/bin/sh

for configs in pr_tlb_miss_second_iteration_P100
do
printf "currently running: %s\r\n" $configs
cd $configs
make
./$configs
cd ..
done
