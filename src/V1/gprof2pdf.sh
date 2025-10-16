#!/bin/bash

rootdir=`pwd`
gprofFile=$1

filename=$(basename "$gprofFile")
extension="${filename##*.}"
filename="${filename%.*}"
pdfFile=${filename}.pdf
dotFile=${filename}.dot

# for all functions
#-e0 means show all edges regardless of how small their contribution is
./gprof2dot.py -e0 -n0 --skew=0.01 ${gprofFile} > $dotFile

# for some top contributing functions
#the -n0.5 flag adds a threshold of 0.5 to the nodes 
# only show functions whose total time fraction is at least 0.5% of the program’s runtime.
# ./gprof2dot.py -e0 -n0.5 --skew=0.01 ${gprofFile} > $dotFile

dot -Tpdf -o $pdfFile $dotFile

