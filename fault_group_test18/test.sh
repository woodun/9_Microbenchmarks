#!/bin/sh

for i in 256 
do
cat fault_group_test18.cu | sed "s/#define STRIDE_64K 256/#define STRIDE_64K $i/" > temp_fault_group_test18.txt
done


