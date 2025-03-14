#!/bin/bash

infile=$1

echo -n > tmp; 

head -n 1 $infile >> tmp 

cat $infile \
|  sed /Entity/d \
|  sed /ID/d \
|  sed s/GPU\ 0/GPU_0/ \
>> tmp

cat tmp

rm tmp
