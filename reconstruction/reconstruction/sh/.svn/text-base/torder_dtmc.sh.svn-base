#!/bin/bash

NDTMCHISTWCUTS=29
NDTMCHISTCALIB=4
NRESHISTWCUTS=8
NCUTLEVELS=8


# dummy date and time
# first, set them to the current date and time
yr=$(date +'%-Y')
mo=$(date +'%-m')
da=$(date +'%-d')
hr=$(date +'%-H')
mi=$(date +'%-M')
sec=$(date +'%-S')
TZ=$(date +"%Z")

function chngFile
{
    if [ ! -f $1 ]; then
	echo "warning: "$1" not found" 1>&2
	return
    fi
    dtstr=$(printf "%d-%d-%d %d:%d:%d %s" $yr $mo $da $hr $mi $sec $TZ)
    yr=$(echo date --date=\"$dtstr '+1 minute'\" +\"%-Y\" | bash)
    mo=$(echo date --date=\"$dtstr '+1 minute'\" +\"%-m\" | bash)
    da=$(echo date --date=\"$dtstr '+1 minute'\" +\"%-d\" | bash)
    hr=$(echo date --date=\"$dtstr '+1 minute'\" +\"%-H\" | bash)
    mi=$(echo date --date=\"$dtstr '+1 minute'\" +\"%-M\" | bash)
    sec=$(echo date --date=\"$dtstr '+1 minute'\" +\"%-S\" | bash)
    echo touch --date=\"$dtstr\" $1 | bash
}



# -------- base names for the plots --------------


# data/mc comparison histograms
dtmchist_wcuts[0]=hTheta
dtmchist_wcuts[1]=hPhi
dtmchist_wcuts[2]=hGfChi2Pdof
dtmchist_wcuts[3]=hLdfChi2Pdof
dtmchist_wcuts[4]=hXcore
dtmchist_wcuts[5]=hYcore
dtmchist_wcuts[6]=hS800
dtmchist_wcuts[7]=hEnergy
dtmchist_wcuts[8]=hNgSd
dtmchist_wcuts[9]=hQtot
dtmchist_wcuts[10]=hQtotNoSat
dtmchist_wcuts[11]=hQpSd
dtmchist_wcuts[12]=hQpSdNoSat
dtmchist_wcuts[13]=hNsdNotClust
dtmchist_wcuts[14]=hQpSdNotClust
dtmchist_wcuts[15]=hPdErr
dtmchist_wcuts[16]=hSigmaS800oS800
dtmchist_wcuts[17]=hHa
dtmchist_wcuts[18]=hSid
dtmchist_wcuts[19]=hRa
dtmchist_wcuts[20]=hDec
dtmchist_wcuts[21]=hL
dtmchist_wcuts[22]=hB
dtmchist_wcuts[23]=hSgl
dtmchist_wcuts[24]=hSgb
dtmchist_wcuts[25]=pNgSdVsEn
dtmchist_wcuts[26]=pNsdNotClustVsEn
dtmchist_wcuts[27]=pQtotVsEn
dtmchist_wcuts[28]=pQtotNoSatVsEn

# Calibration
dtmchist_calib[0]=hFadcPmip
dtmchist_calib[1]=hFwhmMip
dtmchist_calib[2]=hPchPed
dtmchist_calib[3]=hFwhmPed

# Resolution
reshist_wcuts[0]=hThetaRes
reshist_wcuts[1]=hPhiRes
reshist_wcuts[2]=hXcoreRes
reshist_wcuts[3]=hYcoreRes
reshist_wcuts[4]=hEnergyResRat
reshist_wcuts[5]=hEnergyResLog
reshist_wcuts[6]=hEnergyRes2D
reshist_wcuts[7]=pEnergyRes


# S800 vs sec(theta) profile plots
pS800vsSecTheta=pS800vsSecTheta

if [ $# -ne 1 ]; then
    echo "Utility for time-ordering data/mc plots" 1>&2
    echo "(1): data/mc directory name" 1>&2
    exit 2
fi 
tst=${1%\.}
dtmcplotdir=${tst%/}

prefix=$(basename $dtmcplotdir)

# Data/MC histograms with cuts
i=0
while [ $i -lt $NDTMCHISTWCUTS ]; do
    icut=0
    while [ $icut -lt $NCUTLEVELS ]; do
	fname=$dtmcplotdir/$prefix'_'${dtmchist_wcuts[i]}$icut'.png'
	chngFile $fname
	icut=$((icut+1))
    done
    i=$((i+1))
done 

# Data and MC histograms with cuts on top of each other with cuts
i=0
while [ $i -lt $NDTMCHISTWCUTS ]; do
    icut=0
    while [ $icut -lt $NCUTLEVELS ]; do
	fname=$dtmcplotdir/$prefix'_'${dtmchist_wcuts[i]}$icut'_ontop.png'
	chngFile $fname
	icut=$((icut+1))
    done
    i=$((i+1))
done 

# Calibration DATA/MC comparison
i=0
while [ $i -lt $NDTMCHISTCALIB ]; do
    ilayer=0
    while [ $ilayer -lt 2 ]; do
	fname=$dtmcplotdir/$prefix'_'${dtmchist_calib[i]}$ilayer'.png'
	chngFile $fname
	ilayer=$((ilayer+1))
    done
    i=$((i+1))
done

# Calibration with data and MC on top of each other
i=0
while [ $i -lt $NDTMCHISTCALIB ]; do
    ilayer=0
    while [ $ilayer -lt 2 ]; do
	fname=$dtmcplotdir/$prefix'_'${dtmchist_calib[i]}$ilayer'_ontop.png'
	chngFile $fname
	ilayer=$((ilayer+1))
    done
    i=$((i+1))
done

# Resolution
i=0
while [ $i -lt $NRESHISTWCUTS ]; do
    icut=0
    while [ $icut -lt $NCUTLEVELS ]; do
	fname=$dtmcplotdir/$prefix'_'${reshist_wcuts[i]}$icut'.png'
	chngFile $fname
	icut=$((icut+1))
    done
    i=$((i+1))
done 

# S800 vs sec(theta) profiles with cuts
icut=0
while [ $icut -lt $NCUTLEVELS ]; do
    fname=$dtmcplotdir/$prefix'_'$pS800vsSecTheta$icut'.png'
    chngFile $fname
    icut=$((icut+1))
done
