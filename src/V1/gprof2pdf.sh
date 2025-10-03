#!/bin/bash

rootdir=`pwd`
gprofFile=$1

filename=$(basename "$gprofFile")
extension="${filename##*.}"
filename="${filename%.*}"
pdfFile=${filename}.pdf
dotFile=${filename}.dot

# for all functions
./gprof2dot.py -e0 -n0 --skew=0.01 ${gprofFile} > $dotFile

# for some top contributing functions
# ./gprof2dot.py -e0 -n0 --skew=0.01 ${gprofFile} > $dotFile

dot -Tpdf -o $pdfFile $dotFile

