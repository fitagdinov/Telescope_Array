#!/usr/bin/env bash

# look for the SDDIR at most two directories above that of the script
SDDIR=$(dirname $(dirname $(readlink -f $0)))

test ! -d $SDDIR/legacy && \
    echo "ERROR: failed to find \$SDDIR/legacy (/full/path/to/sdanalysis/legacy)!" && \
    exit 2

rm -rf ${SDDIR}/bin
rm -rf ${SDDIR}/lib
rm -rf ${SDDIR}/sdscripts

rm -f ${SDDIR}/Makefile
rm -f ${SDDIR}/makefileset.mk
rm -f ${SDDIR}/sdanalysis_dst2k-ta.mk
rm -f ${SDDIR}/sdanalysis_env.sh
rm -f ${SDDIR}/sdanalysis_checks.mk
rm -f ${SDDIR}/sdanalysis_checks.cpp

for f in ${SDDIR}/legacy/*_makefile; do
    d=$(basename $f _makefile)
    rm -f ${SDDIR}/${d}/${d}.mk
    rm -f ${SDDIR}/${d}/makefile
done
