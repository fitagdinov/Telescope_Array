#!/usr/bin/env bash

# number of days per calibration epoch
SDCALIB_NDAYS=30

# make sure that the resource environmental variable is defined
test -z $COMMON_RESOURCE_DIRECTORY_LOCAL && \
    echo "warning: COMMON_RESOURCE_DIRECTORY_LOCAL not set" >&2 \
    && export COMMON_RESOURCE_DIRECTORY_LOCAL=$(readlink -f $(dirname $0)) && echo "using $COMMON_RESOURCE_DIRECTORY_LOCAL" >&2

# Set the TA SD MC binaries directory

test ! -d $COMMON_RESOURCE_DIRECTORY_LOCAL &&  echo "Error: $COMMON_RESOURCE_DIRECTORY_LOCAL directory not found" && exit 2


TASDMC_BINDIR=$COMMON_RESOURCE_DIRECTORY_LOCAL
test ! -z $TASDMC_BINDIR && TASDMC_BINDIR=$(readlink -f $TASDMC_BINDIR)
export TASDMC_BINDIR

sdmc_calib_extract=$TASDMC_BINDIR/sdmc_calib_extract.run
test ! -x $sdmc_calib_extract && echo "error: executable $sdmc_calib_extract not found" >&2 && exit 2

if [ $# -ne 3 ]; then
    echo "Runs calibration extraction program sdmc_calib_extract.run on ICRR calibration directory" >&2
    echo "sdmc_calib_extract.run reads ICRR calibration DST files and produces sdcalib.bin files" >&2
    echo "that can be readily used by the sdmc_spctr propgram."
    echo "(1): Input ICRR Calib-2 directory" >&2
    echo "(2): output directory for sdcalib.bin files" >&2
    echo "(3): Number of threads">&2
    exit 2
fi

njobs=$3
test $njobs -lt 1 && echo "error: number of threads must be >= 1" >&2 && exit 2

test ! -d $1 && echo "error: input Calib-2 directory $1 not found" >&2 && exit 2
indir=$(readlink -f $1)

test ! -d $1 && echo "error: output directory $2 not found" >&2 && exit 2
outdir=$(readlink -f $2)

tasdconst_pass2=$indir/const/tasdconst_pass2.dst
test ! -f $tasdconst_pass2 && echo "error: $tasdconst_pass2 not found" >&2 && exit 2

tmpdir=$outdir/tmp
mkdir -p $tmpdir
test ! -d $tmpdir && echo "error: failed to create temporary directory $tmpdir" >&2 && exit 2

calibdir=$indir/calib
ls $calibdir/tasdcalib_pass2_??????.dst >$tmpdir/in_all_files
ndstfiles=$(cat $tmpdir/in_all_files | wc -l)
test $ndstfiles -lt 1 && echo "error: no $calibdir/tasdcalib_pass2_??????.dst files found" >&2 && exit 2
nfiles=$((ndstfiles/SDCALIB_NDAYS))
test $((nfiles*SDCALIB_NDAYS)) -lt $ndstfiles && nfiles=$((nfiles+1))
npad=$(printf "%d" $nfiles | wc -m)
split -a $npad -d --numeric-suffixes=1 -l $SDCALIB_NDAYS $tmpdir/in_all_files $tmpdir/in_
nfiles_per_job=$((nfiles/njobs))
test $((nfiles_per_job*njobs)) -lt $nfiles && nfiles_per_job=$((nfiles_per_job+1))

function process_list_files_range() {
    for i in $(seq -w $1 1 $2); do
	ifl=$tmpdir/in_${i}
	ofl=$outdir/sdcalib_${i}.bin
	log=$outdir/out_${i}.log
	cat $ifl | xargs $sdmc_calib_extract -c $tasdconst_pass2 -o $ofl >& $log
    done
}

i=1
while [ $i -lt $nfiles ]; do
    i1=$(printf "%0"$npad"d" $i)
    i=$((i+nfiles_per_job))
    test $i -gt $nfiles && i=$((nfiles+1))
    i2=$(printf "%0"$npad"d" $((i-1)))
    process_list_files_range $i1 $i2 &
done

# wait until all processes finish
wait

# then clean up
rm -rf $tmpdir

exit 0
