#!/bin/sh

grep 'Unified Memory GPU page faults' $1 | sed 's/.*"\([0-9][0-9]*\)",.*,"\[Unified Memory GPU page faults\]"/\1/' > temp_faults.txt
grep 'Unified Memory Memcpy HtoD' $1 > temp_migrations.txt

#number of fault groups
cat temp_faults.txt | wc -l
#number of faults
cat temp_faults.txt | awk '{ SUM += $1} END { print SUM }'
#number of page migration
cat temp_migrations.txt | wc -l

rm temp_faults.txt
rm temp_migrations.txt

