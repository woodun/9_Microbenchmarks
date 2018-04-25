#!/bin/sh


for configs in $(ls -d *)

do
cd $configs
cp ../Makefile .
cd ..
done
