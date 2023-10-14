
# in case if something doesn't work then save the information that describes why it didn't work
fail=0; temp_log='/tmp/'$RANDOM'_temp.log'
touch $temp_log

sdmc_conv_e2_to_spctr_run=$SDDIR/bin/sdmc_conv_e2_to_spctr.run
test ! -x $sdmc_conv_e2_to_spctr_run && echo "error: executable $sdmc_conv_e2_to_spctr_run not found" >> $temp_log && fail=1

if [ $# -ne 3 ]; then
    echo "runs sdmc_conv_e2_to_spctr.run on an input shower library DST file" >> $temp_log
    echo "(1): input card file" >> $temp_log
    echo "(2): processing directory">> $temp_log
    echo "(3): output directory" >> $temp_log
    fail=1
fi

infile=$1
if [ ! -f $infile ] || [ -z $infile ]; then
    echo "error: input file $infile doesn't exist">> $temp_log && fail=1
else
    infile=$(readlink -f $infile)
fi
prodir=$2
if [ ! -d $prodir ] || [ -z $prodir ]; then
    echo "error: processing directory $prodir doesn't exist">> $temp_log && fail=1
else
    prodir=$(readlink -f $2)
fi
outdir=$3
if [ ! -d $outdir ] || [ -z $outdir ]; then 
    echo "error: output directory $outdir doesn't exist">> $temp_log && fail=1
else
    outdir=$(readlink -f $3)
fi

if [ -z $outdir ]; then
    cat $temp_log >& 2
fi

test -z $outdir && outdir='/tmp'
test -z $infile && infile=$(basename $0)'_'${USER}_${RANDOM}_'something_failed.in'

fbase=$(basename $infile)
fbase0=$fbase
fbase0=$(basename $fbase0 .gz)
fbase0=$(basename $fbase0 .bz2)
fbase0=$(basename $fbase0 .dst)

outfile=$prodir/$fbase0'.sdmc_conv_e2_to_spctr.out'
errfile=$prodir/$fbase0'.sdmc_conv_e2_to_spctr.err'
targzlog=$prodir/$fbase0'.sdmc_conv_e2_to_spctr_logs.tar.gz'

if [ $fail -eq 0 ]; then
    
    showlib_dst_file_local=$prodir/$fbase
    rsync -au $infile $showlib_dst_file_local
    
    # initialize the list of all output files
    OUTFILES=""
    
    # files that have spectral sets in them, starting at 10^17.45 eV
    spctr1_1745_dst_gz_file=$prodir/$fbase0'.spctr1.1745.dst.gz'
    spctr2_1745_dst_gz_file=$prodir/$fbase0'.spctr2.1745.dst.gz'
    spctr3_1745_dst_gz_file=$prodir/$fbase0'.spctr3.1745.dst.gz'
    $sdmc_conv_e2_to_spctr_run -e0 0.3162 -s 1 -o $spctr1_1745_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    $sdmc_conv_e2_to_spctr_run -e0 0.3162 -s 2 -o $spctr2_1745_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    $sdmc_conv_e2_to_spctr_run -e0 0.3162 -s 3 -o $spctr3_1745_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    OUTFILES=$OUTFILES$spctr1_1745_dst_gz_file" "$spctr2_1745_dst_gz_file" "$spctr3_1745_dst_gz_file" "
    
    # files that have spectral sets in them, starting at 10^18.95 eV
    spctr1_1895_dst_gz_file=$prodir/$fbase0'.spctr1.1895.dst.gz'
    spctr2_1895_dst_gz_file=$prodir/$fbase0'.spctr2.1895.dst.gz'
    spctr3_1895_dst_gz_file=$prodir/$fbase0'.spctr3.1895.dst.gz'
    $sdmc_conv_e2_to_spctr_run -e0 8.9125 -s 1 -o $spctr1_1895_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    $sdmc_conv_e2_to_spctr_run -e0 8.9125 -s 2 -o $spctr2_1895_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    $sdmc_conv_e2_to_spctr_run -e0 8.9125 -s 3 -o $spctr3_1895_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    OUTFILES=$OUTFILES$spctr1_1895_dst_gz_file" "$spctr2_1895_dst_gz_file" "$spctr3_1895_dst_gz_file" "

    # files that have spectral sets in them, starting at 10^19.45 eV
    spctr1_1945_dst_gz_file=$prodir/$fbase0'.spctr1.1945.dst.gz'
    spctr2_1945_dst_gz_file=$prodir/$fbase0'.spctr2.1945.dst.gz'
    spctr3_1945_dst_gz_file=$prodir/$fbase0'.spctr3.1945.dst.gz'
    $sdmc_conv_e2_to_spctr_run -e0 28.1838 -s 1 -o $spctr1_1945_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    $sdmc_conv_e2_to_spctr_run -e0 28.1838 -s 2 -o $spctr2_1945_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    $sdmc_conv_e2_to_spctr_run -e0 28.1838 -s 3 -o $spctr3_1945_dst_gz_file $showlib_dst_file_local 1>>$outfile 2>>$errfile
    OUTFILES=$OUTFILES$spctr1_1945_dst_gz_file" "$spctr2_1945_dst_gz_file" "$spctr3_1945_dst_gz_file" "
    
  
    # combine the log files into a tar.gz file
    cd $prodir
    tar -czf $targzlog $(basename $outfile) $(basename $errfile)
    OUTFILES=$OUTFILES$targzlog
    
    # clean up
    rm $showlib_dst_file_local $outfile $errfile
    
    # move the results to the output directory
    mv $OUTFILES $outdir/.
    
else
    echo "Something has failed" >> $temp_log 
fi
done_file=$outdir/$fbase'.done'
cat $temp_log > $done_file
rm $temp_log
echo $(date +"%Y%m%d %H%M%S %Z") >> $done_file
