#!/bin/sh

grep 'Unified Memory GPU page faults' $1 | sed 's/.*"\([0-9][0-9]*\)",.*,"\[Unified Memory GPU page faults\]"/\1/' > temp_faults.txt
grep 'Unified Memory Memcpy HtoD' $1 > temp_migrations.txt

#number of fault groups
v1=`cat temp_faults.txt | wc -l`
#number of faults
v2=`cat temp_faults.txt | awk '{ SUM += $1} END { print SUM }'`
#number of page migration
v3=`cat temp_migrations.txt | wc -l`

printf "%d %d %d\n" $v1 $v2 $v3

rm temp_faults.txt
rm temp_migrations.txt

