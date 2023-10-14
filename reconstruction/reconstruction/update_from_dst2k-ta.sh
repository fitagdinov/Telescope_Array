#!/usr/bin/env bash


mydir=$(dirname $(readlink -f $0))

if [ $# -ne 2 ]; then
   echo "Update a particular bank of this dst2k-ta bundle from some other" >&2
   echo "USAGE:"
   echo "(1): bank name" >&2
   echo "(2): search path for the alternative dst2k-ta bundle" >&2
   exit 2
fi


bname=$1
altdstpath=${2%/}

test ! -d ${altdstpath} && echo "error: directory ${altdstpath} not found!" >&2 && exit 2
altdstpath=$(readlink -f ${altdstpath})


new_dsth=${altdstpath}/inc/${bname}_dst.h
new_dsti=${altdstpath}/inc/${bname}_dst.inc
new_dstc=${altdstpath}/src/bank/lib/${bname}_dst.c


cur_dsth=${mydir}/inc/${bname}_dst.h
cur_dsti=${mydir}/inc/${bname}_dst.inc
cur_dstc=${mydir}/src/bank/lib/${bname}_dst.c


test -f $new_dsth && echo "cp ${new_dsth} ${cur_dsth}" && cp ${new_dsth} ${cur_dsth}
test -f $new_dsti && echo "cp ${new_dsti} ${cur_dsti}" && cp ${new_dsti} ${cur_dsti}
test -f $new_dstc && echo "cp ${new_dstc} ${cur_dstc}" && cp ${new_dstc} ${cur_dstc}

