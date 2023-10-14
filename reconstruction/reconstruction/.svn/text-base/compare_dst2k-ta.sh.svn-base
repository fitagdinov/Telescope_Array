#!/usr/bin/env bash

mydir=$(dirname $(readlink -f $0))

if [ $# -ne 1 ]; then
   echo "Compare this dst2k-ta bundle with some other" >&2
   echo "report different/missing *.c, *.h, *.inc files in the other dst2k-ta bundle" >&2
   echo "goliath tasoft dst2k-ta is in general cleaner than general ta dst:" >&2 
   echo "gcc warnings have been cleaned in goliath dst2k-ta," >&2 
   echo "icc compilation + 64bit compatibility for bz2, gz files have been fixed" >&2
   echo "Until general ta dst gets updated, there will (expected) differences"
   echo "USAGE:"
   echo "(1): search path for the alternative dst2k-ta bundle" >&2
   exit 2
fi
altdstpath=${1%/}
test ! -d ${altdstpath} && echo "error: directory ${altdstpath} not found!" >&2 && exit 2
altdstpath=$(readlink -f ${altdstpath})

dsth=$(find ${mydir} -name "*.h")
dsti=$(find ${mydir} -name "*.inc")
dstc=$(find ${mydir} -name "*.c")

#total number of differences
ndftot=0

# checking c headers
for file in $dsth; do
    altfile=$(find $altdstpath -name $(basename $file) | head -1)
    if [ ! -z $altfile ]; then
	ndf=$(diff $file $altfile | wc -l)
	if [ $ndf -ne 0 ]; then
	    echo "$file and $altfile are different"
	    ndftot=$((ndftot+1))  
	fi
    else
	echo "$file not in $altdstpath"
	ndftot=$((ndftot+1))
    fi
done

# checking fortran headers
for file in $dsti; do
    altfile=$(find $altdstpath -name $(basename $file) | head -1)
    if [ ! -z $altfile ]; then
	ndf=$(diff $file $altfile | wc -l)
	if [ $ndf -ne 0 ]; then
	    echo "$file and $altfile are different"
	    ndftot=$((ndftot+1))  
	fi
    else
	echo "$file not in $altdstpath"
	ndftot=$((ndftot+1))
    fi
done

# checking c sources
for file in $dstc; do
    altfile=$(find $altdstpath -name $(basename $file) | head -1)
    if [ ! -z $altfile ]; then
	ndf=$(diff $file $altfile | wc -l)
	if [ $ndf -ne 0 ]; then
	    echo "$file and $altfile are different"
	    ndftot=$((ndftot+1))  
	fi
    else
	echo "$file not in $altdstpath"
	ndftot=$((ndftot+1))
    fi
done

echo "TOTAL DIFFERENCES: $ndftot"
