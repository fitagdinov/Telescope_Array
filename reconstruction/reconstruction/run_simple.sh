#!/bin/bash


if [ $# -ne 1 ]; then
    echo "(1): DST file search path"
    exit 0
fi
dstdatadir=${1%/}
sditeratordir=$(pwd)

# prepare a list of runs to analyze 
# (this file will be passed to iterator):
wantfile=./dstfilelist.txt
find $dstdatadir -name *.dst.gz >$wantfile
sditeratorprog=$sditeratordir/bin/sditerator.run
outfile=sdevents.asc
$sditeratorprog -i $wantfile -o $outfile 
