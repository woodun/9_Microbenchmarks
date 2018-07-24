#!/bin/sh

for i in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288
do
sh profhandler.sh prof512512size8gpage$i
done

