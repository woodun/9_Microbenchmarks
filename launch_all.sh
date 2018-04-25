#!/bin/sh

for configs in pr_tlb_miss_first_iteration_P100 pr_tlb_miss_second_iteration_P100
do
printf "\r\n##############currently running: %s##############\r\n\r\n" $configs
cd $configs
make
./$configs
cd ..
done
