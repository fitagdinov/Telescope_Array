#!/usr/bin/env bash

# look for the SDDIR at most two directories above that of the script
SDDIR=$(dirname $(dirname $(readlink -f $0)))

test ! -d $SDDIR/legacy && \
    echo "ERROR: failed to find \$SDDIR/legacy (/full/path/to/sdanalysis/legacy)!" && \
    exit 2

mkdir -p ${SDDIR}/bin
mkdir -p ${SDDIR}/lib

cp ${SDDIR}/legacy/Makefile ${SDDIR}/.
cp ${SDDIR}/legacy/makefileset.mk ${SDDIR}/.
cp ${SDDIR}/legacy/sdanalysis_dst2k-ta.mk ${SDDIR}/.
cp ${SDDIR}/legacy/sdanalysis_env.sh ${SDDIR}/.
cp ${SDDIR}/legacy/sdanalysis_checks.mk ${SDDIR}/.
cp ${SDDIR}/legacy/sdanalysis_checks.cpp ${SDDIR}/.

cp -r ${SDDIR}/legacy/sdscripts ${SDDIR}/.

for f in ${SDDIR}/legacy/*_makefile; do
    d=$(basename $f _makefile)
    cp ${SDDIR}/legacy/${d}.mk ${SDDIR}/${d}/.
    cp ${f} ${SDDIR}/${d}/makefile
done
