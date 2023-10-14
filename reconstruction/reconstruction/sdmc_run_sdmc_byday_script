#!/usr/bin/env bash

# in case if something doesn't work then save the information that describes why it didn't work
fail=0
cmdargs_given=1
temp_log='/tmp/'$(basename $0)'_'${USER}_${RANDOM}_'temp.log'
touch $temp_log
echo "START: "${HOSTNAME}" "$(date +"%Y%m%d %H%M%S %Z") >> $temp_log



test ${#ROOTSYS} -eq 0 && fail=$((fail+1)) && echo "error(${fail}): ROOTSYS environmental variable is not set" >> $temp_log


# CERN Root run - time environment
if [ ${#ROOTSYS} -gt 0 ]; then 
    thisroot_sh=$ROOTSYS/bin/thisroot.sh
    test ! -f $thisroot_sh && fail=$((fail+1)) && echo "error(${fail}): $thisroot_sh not found" >> $temp_log
    test -f $thisroot_sh && source $thisroot_sh
fi


# Program
sdmc_byday=$SDDIR/bin/sdmc_byday.run

test ${#SDDIR} -eq 0 && fail=$((fail+1)) && echo "error(${fail}): SDDIR environmental variable is not set" >> $temp_log
if [ ${#SDDIR} -gt 0 ]; then
    if [ -d ${SDDIR} ]; then
	test ! -x $sdmc_byday && fail=$((fail+1)) && echo "error(${fail}): $sdmc_byday executable not found" >> $temp_log
    else
	fail=$((fail+1))
	echo "error(${fail}): SDDIR=${SDDIR} directory not found" >> $temp_log
    fi
fi


gzip_used=gzip
which --skip-alias pigz >& /dev/null
test $? -eq 0 && gzip_used=pigz

my_name=$(basename $0)
if [ $# -ne 3 ]; then
    echo "" >> $temp_log
    echo "" >> $temp_log
    echo "Split SD MC DST files into daily parts">> $temp_log
    echo "Author: Dmitri Ivanov <dmiivanov@gmail.com>">> $temp_log
    echo "" >> $temp_log
    echo "Usage: $my_name [infile] [prodir] [outdir]">> $temp_log
    echo "(1): input ASCII list file with paths to DST files" >> $temp_log
    echo "(2): processing directory for any partially completed work">> $temp_log
    echo "(3): output directory for the results" >> $temp_log
    cmdargs_given=0
fi

infile=""
prodir=""
outdir=""

if [ $cmdargs_given -eq 1 ]; then
    infile=$1
    if [ ! -f $infile ]; then
        fail=$((fail+1))
        echo "error(${fail}): input file infile '$infile' doesn't exist">> $temp_log
    else
        infile=$(readlink -f $infile)
    fi
    prodir=$2
    if [ ! -d $prodir ]; then
        fail=$((fail+1))
        echo "error(${fail}): processing directory '$prodir' doesn't exist">> $temp_log
        prodir=""
    else
        prodir=$(readlink -f $prodir)
    fi
    outdir=$3
    if [ ! -d $outdir ]; then
        fail=$((fail+1))
        echo "error(${fail}): output directory '$outdir' doesn't exist">> $temp_log
        outdir=""
    else
        outdir=$(readlink -f $outdir)
    fi
fi

test -z $infile && infile=$(basename $0)'_'${USER}_${RANDOM}_'something_failed'
fbase=$(basename $infile)
rnd_suf='_'$(date +%s)'_'$RANDOM
prodir_local=${prodir}/${fbase}${rnd_suf}

while [ $fail -eq 0 -a $cmdargs_given -eq 1 ]; do
    
    # create a local sub-directory
    rm -rf ${prodir_local}
    mkdir -p ${prodir_local}
    test ! -d ${prodir_local} && fail=$((fail+1)) && echo "error(${fail}): failed to create ${prodir_local}" >> $temp_log && break
    
    # create local output directory
    byday=${prodir_local}/${fbase}'_byday'
    rm -rf ${byday}
    mkdir -p ${byday}
    test ! -d ${byday} && fail=$((fail+1)) && echo "error(${fail}): failed to create ${byday}" >> $temp_log && break
    
    # copy the file to the local processing directory
    rsync -au $infile $prodir_local/.
    test ! -f ${prodir_local}/$fbase && fail=$((fail+1)) && echo "error(${fail}): failed to copy ${infile}" >> $temp_log && break
    
    # copy the DST files and create a local list file with paths to local DST files
    infile_local=${prodir_local}/${fbase}'.local'
    rm -f $infile_local
    touch $infile_local
    for dstfile in $(cat ${prodir_local}/$fbase); do
	dstfile_local=${prodir_local}/$(basename $dstfile)
	rsync -au $dstfile ${prodir_local}/.
	test $? -ne 0 && fail=$((fail+1)) && echo "error(${fail}): failed to copy ${dstfile}" >> $temp_log && break
	test ! -f ${dstfile_local} && fail=$((fail+1)) && echo "error(${fail}): failed to copy ${dstfile}" >> $temp_log && break
    done
    test $fail -gt 0 && break
    for dstfile in $(cat ${prodir_local}/$fbase); do
        dstfile_local=${prodir_local}/$(basename $dstfile)
        test ! -f ${dstfile_local} && fail=$((fail+1)) && echo "error(${fail}): failed to copy ${dstfile}" >> $temp_log && break
        if [ $(basename $dstfile_local .gz) != $(basename $dstfile_local) ]; then
            $gzip_used -f --decompress ${dstfile_local}
	    test $? -ne 0 && fail=$((fail+1)) && echo "error(${fail}): failed to decompress ${dstfile_local}" >> $temp_log && break
            dstfile_local=${prodir_local}/$(basename ${dstfile_local} .gz)
	    test ! -f ${dstfile_local} && fail=$((fail+1)) && \
		echo "error(${fail}): failed to decompress ${dstfile_local}.gz" >> $temp_log && break
        fi
        echo ${dstfile_local} >> ${infile_local}
    done
    test $fail -gt 0 && break

    # run the splitting program
    $sdmc_byday -i $infile_local -o ${byday} -p ${fbase} 1>> $temp_log 2>> $temp_log
    if [ $? -ne 0 ]; then
	fail=$((fail+1))
	echo "error(${fail}): ${sdmc_byday} has failed" >> $temp_log
	break
    fi

     # copy the result to the output directory
    rsync -au ${byday} ${outdir}/.
    test $? -ne 0 && fail=$((fail+1)) && echo "error(${fail}): failed to copy ${byday}" >> $temp_log && break

    # break out of the main loop
    break
done

# always clean the temporary processing directory
rm -rf ${prodir_local}

# if something failed, indicate the fail count
if [ $cmdargs_given -eq 0 ]; then
    echo "Command line arguments not given properly" >> $temp_log
    test $fail -gt 0 && echo "Besides the command line arguments, something else has failed, fail count = ${fail}" >> $temp_log
else
    test $fail -gt 0 && echo "Something has failed, fail count = ${fail}" >> $temp_log
fi
echo "FINISH: "${HOSTNAME}" "$(date +"%Y%m%d %H%M%S %Z") >> $temp_log
if [ ! -z $outdir ]; then
    done_file=$outdir/$fbase'.done'
    failed_file=$outdir/$fbase'.failed'
    status_file=${done_file}
    test $fail -gt 0 && status_file=${failed_file}
    if [ $fail -gt 0 ]; then
        echo "fail count = ${fail}" >> ${status_file}
    fi
    cat $temp_log >> ${status_file}
else
    cat $temp_log >&2
fi

rm -f $temp_log

# Exit with the flag.  If success, then fail should be zero.
exit $fail

