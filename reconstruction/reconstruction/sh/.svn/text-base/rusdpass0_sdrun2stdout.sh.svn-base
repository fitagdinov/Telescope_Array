#!/bin/bash

if [ $# -ne 1 ]; then
    echo "print the contents of the raw SD data file to stdout">&2
    echo "(1): sd run file (.tar.bz2)" >&2
    exit 1
fi
f=$(readlink -f $1)
test ! -f $f && echo "error: '$f' not found">&2 && exit 2
yy=$(echo $f | sed 's/.*TASD\([0-9][0-9]\)_[0-9][0-9][0-9].*/\1/' | awk '{print $1/1}')
yyyy=$((2000+yy))
d=$(basename $1 .tar.bz2)'.Y'$yyyy
tar -xjf $f $d -O
exit 0
