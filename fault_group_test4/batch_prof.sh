#!/bin/sh

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576	2097152 4194304 8388608 16777216 33554432 67108864 134217728
do
sh profhandler.sh prof132nopragmasize8gpage$i.txt
done

