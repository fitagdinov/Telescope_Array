#!/usr/bin/env bash

# default executable suffix
EXE_SUFFIX=.run

# this is necessary to run sdmc_spctr, so that there's no limit
# on the size of the stack
ulimit -s unlimited

# number of attempts running sdmc_spctr on a given calibration epoch on a given shower library tile file
NTRY=10

# in case if something doesn't work then save the information that describes why it didn't work
fail=0
cmdargs_given=1
temp_log='/tmp/'$(basename $0)'_'${USER}_${RANDOM}_'temp.log'
touch $temp_log

# make sure that the resource environmental variable is defined
if [ ${#COMMON_RESOURCE_DIRECTORY_LOCAL} -eq 0 ]; then
    export COMMON_RESOURCE_DIRECTORY_LOCAL=$(readlink -f $(dirname $0))
    echo "warning: COMMON_RESOURCE_DIRECTORY_LOCAL not set; " \
	"setting  COMMON_RESOURCE_DIRECTORY_LOCAL to ${COMMON_RESOURCE_DIRECTORY_LOCAL}" >>$temp_log
fi


# Set the TA SD MC resources directory
test ! -d $COMMON_RESOURCE_DIRECTORY_LOCAL && fail=$((fail+1)) && echo "error(${fail}): $COMMON_RESOURCE_DIRECTORY_LOCAL directory not found" >> $temp_log


# by default, smear energies in 0.1 log10(E/eV) bin
# so that events follow E^-2 spectrum in the bin
# if SDMC_SPCTR_SMEAR_ENERGIES enviromental variable
# is set, then use the value of that variable instead
if [ ${#SDMC_SPCTR_SMEAR_ENERGIES} -eq 0 ]; then
    export SDMC_SPCTR_SMEAR_ENERGIES=1
fi

azi=${COMMON_RESOURCE_DIRECTORY_LOCAL}/azi.txt
if [ -f ${azi} ]; then
    echo "NOTICE: using predefined event azimuthal angles from ${azi}" >> $temp_log
else
    echo "NOTICE: sampling event azimuthal angles randomly (if needed, create ${azi} with predefined values)" >> $temp_log
fi


# look for the SDDIR at most two directories above that of the script
if [ ${#SDDIR} -eq 0 ]; then
    SDDIR=$(dirname $(readlink -f $0))
    test ! -d $SDDIR/sdmc && SDDIR=$(dirname ${SDDIR})
    test ! -d $SDDIR/sdmc && SDDIR=$(dirname ${SDDIR})
    test ! -d $SDDIR/sdmc && \
        echo "ERROR: failed to find SDDIR (path/to/sdanalysis); try setting SDDIR environmental variable!" && \
        exit 2
    export SDDIR
else
    if [ -d ${SDDIR} ]; then
        test ! -d $SDDIR/sdmc && \
            echo "ERROR: failed to find SDDIR/sdmc ($SDDIR/sdmc); set SDDIR environmental variable correctly!" && \
            exit 2
    else
        echo "ERROR: SDDIR=${SDDIR} directory not found; set SDDIR environmental variable correctly!" >&2
        exit 2
    fi
fi

# binary files
sdmc_spctr=sdmc_spctr${EXE_SUFFIX}

test ! -x $SDDIR/bin/$sdmc_spctr && fail=$((fail+1)) && echo "error(${fail}): executable $SDDIR/bin/$sdmc_spctr not found!" >> $temp_log
res=$($SDDIR/bin/$sdmc_spctr 2>&1 | grep Usage | wc -l)
if [ $res -ne 1 ]; then
    fail=$((fail+1))
    echo "error(${fail}): $sdmc_spctr program is not working!" >> $temp_log
fi


# atmosheric muon file needs to be present in the binary directory
atmos_bin=$COMMON_RESOURCE_DIRECTORY_LOCAL/atmos.bin
test ! -f $atmos_bin && fail=$((fail+1)) && echo "error(${fail}): file atmos.bin not found in COMMON_RESOURCE_DIRECTORY_LOCAL=${COMMON_RESOURCE_DIRECTORY_LOCAL}" >> $temp_log

# SD calibration files
n_sdcalib=$(find ${COMMON_RESOURCE_DIRECTORY_LOCAL} -name "sdcalib_*.bin" 2>/dev/null | wc -l)
test $n_sdcalib -lt 1 && fail=$((fail+1)) && echo "error(${fail}): not a single sdcalib_*.bin file found in COMMON_RESOURCE_DIRECTORY_LOCAL=${COMMON_RESOURCE_DIRECTORY_LOCAL}" >> $temp_log

# check for other required programs
test -z $SDDIR && fail=$((fail+1)) && echo "error(${fail}): SDDIR environmental variable is not set; point SDDIR to /full/path/to/sdanalysis" >> $temp_log
dstcat=$SDDIR/bin/dstcat${EXE_SUFFIX}
test ! -x $dstcat && fail=$((fail+1)) && echo "error(${fail}): executable ${dstcat} not found" >> $temp_log
sdmc_tsort=$SDDIR/bin/sdmc_tsort${EXE_SUFFIX}
test ! -x $sdmc_tsort && fail=$((fail+1)) && echo "error(${fail}): executable ${sdmc_tsort} not found" >> $temp_log


if [ $# -ne 3 ]; then
    echo "" >> $temp_log
    echo "Usage: "$(basename $0)" [DATXXXXID.*.tothrow.txt] [processing_directory] [output_directory]" >> $temp_log
    echo "runs sdmc_spctr on a DATXXXXID.*.tothrow.txt input card file that tells how many events to throw" >> $temp_log
    echo "(1): input DATXXXXID.*.tothrow.txt card file" >> $temp_log
    echo "(2): processing directory" >> $temp_log
    echo "(3): output directory" >> $temp_log
    echo "" >> $temp_log
    echo "Input DATXXXXID.*.tothrow.txt card file must contain the following lines" >> $temp_log
    echo "(CONTENTS of DATXXXXID.*.tothrow.txt):" >> $temp_log
    echo "     SHOWLIB_FILE /full/path/to/DATXXXXID_gea.dat" >> $temp_log
    echo "     NPARTICLE_PER_EPOCH <float>" >> $temp_log
    echo "NOTES:" >> $temp_log
    echo "File atmpos.bin is the simulation of the atmospheric muons that must be found in \$COMMON_RESOURCE_DIRECTORY_LOCAL folder" >> $temp_log
    echo "Files sdcalib_[epoch_number].bin must be placed in \$COMMON_RESOURCE_DIRECTORY_LOCAL folder" >> $temp_log
    echo "where epoch_number are integers that enumerate each 30 day period of TA SD data" >> $temp_log
    echo "sdcalib_[epoch_number].bin files are produced by running ${SDDIR}/bin/sdmc_run_sdmc_calib_extract on tasdcalibev_pass2_YYMMDD.dst files" >> $temp_log
    echo "NPARTICLE_PER_EPOCH is the Poisson mean for the number of particles to throw for every 30 day period" >> $temp_log
    echo "Instead of /full/path/to/DATXXXXID_gea.dat, you can use /full/path/to/DATXXXXID.*.tar.gz archive" >> $temp_log 
    echo "that contains DATXXXXID_gea.dat in it. The file DATXXXXID_gea.dat will then be unpacked automatically." >> $temp_log
    echo "DATXXXXID.*.tothrow.txt files are produced by running ${SDDIR}/bin/sdmc_prep_sdmc_run" >> $temp_log
    echo "on a set of DATXXXXID_gea.dat and/or DATXXXXID.*.tar.gz files" >> $temp_log
    echo "The meaning of XXXXID in the DATXXXXID should be interpreted as follows."  >> $temp_log
    echo "XXXX is a 4-digit event number" >> $temp_log
    echo "ID is the (logarithmic) energy channgel: " >> $temp_log
    echo "   ID=00-25: energy from 10^18.0 to 10^20.5 eV" >> $temp_log
    echo "   ID=26-39: energy from 10^16.6 to 10^17.9 eV" >> $temp_log
    echo "   ID=80-85: energy from 10^16.0 to 10^16.5 eV" >> $temp_log
    echo "" >> $temp_log
    echo "" >> $temp_log
    cmdargs_given=0
fi

infile=""
prodir=""
outdir=""

if [ $cmdargs_given -eq 1 ]; then
    infile=$1
    if [ ! -f $infile ]; then
	fail=$((fail+1))
	echo "error(${fail}): input file $infile doesn't exist" >> $temp_log
    else
	infile=$(readlink -f $infile)
    fi
    prodir=$2
    if [ ! -d $prodir ]; then
	fail=$((fail+1))
	echo "error(${fail}): processing directory $prodir doesn't exist" >> $temp_log
	prodir=""
    else
	prodir=$(readlink -f $prodir)
    fi
    outdir=$3
    if [ ! -d $outdir ]; then
	fail=$((fail+1))
	echo "error(${fail}): output directory $outdir doesn't exist" >> $temp_log
	outdir=""
    else
	outdir=$(readlink -f $3)
    fi
fi
test -z $infile && infile=$(basename $0)'_'${USER}_${RANDOM}_'something_failed.in'
fbase=$(basename $infile)
fbase0=$(basename $infile .tothrow.txt)
outfile=$prodir/$fbase0'.sdmc_spctr.out'
errfile=$prodir/$fbase0'.sdmc_spctr.err'
targzlog=$prodir/$fbase0'.sdmc_spctr_logs.tar.gz'

while [ $cmdargs_given -eq 1 -a $fail -eq 0 ]; do
    showlib_file=$(cat $infile | grep SHOWLIB_FILE | awk '{print $2}')
    nparticle_per_epoch=$(cat $infile | grep NPARTICLE_PER_EPOCH | awk '{print $2}')
    showlib_file_local=$prodir/$(basename $showlib_file)
    rsync -au $showlib_file $showlib_file_local
    showlib_dat_file_local_basename=$(basename $showlib_file_local | sed 's/\(DAT[0-9]\{6\}\).*/\1/')'_gea.dat'
    showlib_dat_file_local=${prodir}/${showlib_dat_file_local_basename}
    # check if this is a .tar.gz shower library file.  If so, unpack the gea_dat file first
    if [[ $showlib_file_local == *".tar.gz" ]]; then
	cd $prodir
	tar -xzvf $showlib_file_local $showlib_dat_file_local_basename 1>>$outfile 2>>$errfile
	if [ $? -ne 0 -o ! -f $showlib_dat_file_local ]; then
	    fail=$((fail+1))
	    echo "error(${fail}): failed to unpack ${showlib_dat_file_local_basename} from ${showlib_file_local}" >> $temp_log
	fi
    fi
    test ! -f $showlib_dat_file_local && fail=$((fail+1)) && echo "error(${fail}): failed to obtain ${showlib_dat_file_local}" >> $temp_log
    
    # if something failed at this point then stop
    test $fail -gt 0 && break
    
    echo $(basename $showlib_file)" processed on ${UUFSCELL} ${HOSTNAME} using ${sdmc_spctr}" >>$outfile
    # go over each calibration file and generate $nparticle_per_epoch
    # out of the shower library file for each of the calibration files (epochs)
    generated_dst_files=""
    for sdcalib_file in $COMMON_RESOURCE_DIRECTORY_LOCAL/sdcalib_*.bin; do
    	
       	# obtain the calibration epoch number from the name of the calibration file
	epoch_number=$(basename $sdcalib_file .bin | sed 's/sdcalib_//')
	    
	# DST output file
	sdmc_spctr_dst_gz_file_epoch=${showlib_file_local}'_'${epoch_number}'.dst.gz'

      
	# run sdmc_spctr.  Attempt ${NTRY} times with different random seeds before quitting.
	itry=0
	err_status=0
	while [ ${itry} -lt ${NTRY} ]; do
	    
	    # count the number of tries
	    itry=$((itry+1))
	    echo "attempt ${itry} / ${NTRY} running ${sdmc_spctr}" >>$outfile
	    sleep 5s
	    
	    # random seed is the number of nanoseconds from the date command
	    random_seed=$(date +%N)
	    

	    ######### sdmc_spctr program is executed here (below) #################################
	    #########################################################################################
            # run sdmc_spctr program on the shower library file with the specified
	    # number of events for the current calibration epoch; use system time for a random seed
	    if [ ! -f ${azi} ]; then
		# azimuthal angles are sampled randomly
		$sdmc_spctr $showlib_dat_file_local $sdmc_spctr_dst_gz_file_epoch \
		    $nparticle_per_epoch $random_seed $epoch_number $sdcalib_file \
		    $atmos_bin ${#SDMC_SPCTR_SMEAR_ENERGIES} 1>>$outfile 2>>$errfile
	    else
		# predefined azimuthal angles are taken from an ASCII file
		$sdmc_spctr $showlib_dat_file_local $sdmc_spctr_dst_gz_file_epoch \
		    $nparticle_per_epoch $random_seed $epoch_number $sdcalib_file \
		    $atmos_bin ${#SDMC_SPCTR_SMEAR_ENERGIES} ${azi} 1>>$outfile 2>>$errfile
	    fi
	    ##########################################################################################
	    
	    # check the status of execution of sdmc_spctr
	    err_status=$?
	    if [ $err_status -eq 0 ]; then
		echo "successful attempt ${itry} / ${NTRY} running ${sdmc_spctr}" >>$outfile
		break
	    fi
	    
	    # if failed, print a notice and try more times if the tries have not been exhausted
	    echo "failed attempt ${itry} / ${NTRY} running ${sdmc_spctr} with arguments:" >> $errfile
	    echo "${sdmc_spctr} ${showlib_dat_file_local} ${sdmc_spctr_dst_gz_file_epoch} ${nparticle_per_epoch}" \
		"${random_seed} ${epoch_number} ${sdcalib_file} ${atmos_bin}" >> $errfile
	done
	
	# if couldn't get the sdmc_spctr to run after ntry attempts, then quit with failed flag
	if [ $err_status -gt 0 ]; then
	    fail=$((fail+1))
	    echo "error(${fail}): ${sdmc_spctr} failed while attempted running on ${HOSTNAME} ${NTRY} times" >> $temp_log
	    cat $errfile >> $temp_log
	    break
	fi
	
	# number of events thrown
	nevents_thrown=$(cat $outfile | grep 'Number of Events Thrown:' | tail -1 | awk '{print $5}')
	test -z $nevents_thrown && nevents_thrown=0
	
        # if number of thrown events is greater than zero then time sort the file and add the result file to the list
	if [ $nevents_thrown -gt 0 ]; then
	    sdmc_spctr_dst_gz_file_epoch_tsort=${showlib_file_local}'_'${epoch_number}'.tsort.dst.gz'
	    $sdmc_tsort $sdmc_spctr_dst_gz_file_epoch -o1f $sdmc_spctr_dst_gz_file_epoch_tsort 1>>$outfile 2>>$errfile
	    if [ $? -ne 0 ]; then
		fail=$((fail+1))
		echo "error(${fail}): ${sdmc_tsort} failed while running on ${HOSTNAME}" >> $temp_log
		echo "with arguments:" >> $temp_log
		echo "${sdmc_spctr_dst_gz_file_epoch} -o1f ${sdmc_spctr_dst_gz_file_epoch_tsort}" >> $temp_log
		cat $errfile >> $temp_log
		break
	    fi
	    if [ ! -f $sdmc_spctr_dst_gz_file_epoch_tsort ]; then
		fail=$((fail+1))
		echo "error(${fail}): failed to produce ${sdmc_spctr_dst_gz_file_epoch_tsort}" >> $temp_log
		break
	    fi
	    rm -f $sdmc_spctr_dst_gz_file_epoch
	    mv $sdmc_spctr_dst_gz_file_epoch_tsort $sdmc_spctr_dst_gz_file_epoch
	    generated_dst_files=$generated_dst_files$sdmc_spctr_dst_gz_file_epoch" "
	else
	    rm -f $sdmc_spctr_dst_gz_file_epoch
	fi
    done

    # break out of the main loop if something has failed
    test $fail -gt 0 && break
    
    # combine the results of the simulation over all epochs into one output file
    sdmc_spctr_dst_gz_file=$showlib_file_local'.dst.gz'
    $dstcat -o $sdmc_spctr_dst_gz_file $generated_dst_files 1>>$outfile 2>>$errfile
    
    # combine the log files into a tar.gz file
    cd $prodir
    tar -czf $targzlog $(basename $outfile) $(basename $errfile)
    
    # clean up
    rm -f $showlib_file_local $showlib_dat_file_local $generated_dst_files $outfile $errfile
    
    # move the results to the output directory
    mv $sdmc_spctr_dst_gz_file $targzlog $outdir/.
    
    # done processing, break out of the main loop
    break
done

# if something failed, indicate the fail count
if [ $cmdargs_given -eq 0 ]; then
    echo "Command line arguments not given properly" >> $temp_log
    test $fail -gt 0 && echo "Besides the command line arguments, something else has failed, fail count = ${fail}" >> $temp_log
else
    test $fail -gt 0 && echo "Something has failed, fail count = ${fail}" >> $temp_log
fi
if [ ! -z $outdir ]; then
    done_file=$outdir/$fbase'.done'
    failed_file=$outdir/$fbase'.failed'
    status_file=${done_file}
    test $fail -gt 0 && status_file=${failed_file}
    echo ${HOSTNAME}" "$(date +"%Y%m%d %H%M%S %Z") >> $status_file
    if [ $fail -gt 0 ]; then
	echo "fail count = ${fail}" >> ${status_file}
    fi
    cat $temp_log >> ${status_file}
    rm -f $temp_log

else
    cat $temp_log >&2
    rm $temp_log
    echo ${HOSTNAME}" "$(date +"%Y%m%d %H%M%S %Z") >&2
fi

# Exit with the flag.  If success, then fail should be zero.
exit $fail
