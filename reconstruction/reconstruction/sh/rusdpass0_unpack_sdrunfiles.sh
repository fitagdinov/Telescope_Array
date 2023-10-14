#!/bin/bash

if [ $# -ne 3 ]; then
    echo ""
    echo "Unpack the SD run files (.tar.bz2) that are needed to parse events on a particular date">&2
    echo "into a specified output directory.  Also, print out a list of unpacked files (to stdout)">&2
    echo "that can be used for making the input list-file required for running the rusdpass0.run program.">&2
    echo ""
    echo "(1): general path that contains the files for that date ">&2
    echo "     (OK if the *.tar.bz2 run files are located in some directories or subdirectories within that path)">&2
    echo "(2): date, yymmdd format">&2
    echo "(3): output directory for the unpacked files" >&2
    echo ""
    exit 1
fi

run_search_path=$(readlink -f $1)
test ! -d $run_search_path && echo "error: '$run_search_path' directory not found">&2 && exit 2

yymmdd=$2

outdir=$(readlink -f $3)
test  ! -d $outdir && echo "error: '$outdir' directory not found">&2 && exit 2

function get_sdrun_date
{
    if [ $# -ne 1 ]; then
	echo "error: get_sdrun_date: wrong number of arguments">&2
	return
    fi
    f=$(readlink -f $1)
    test ! -f $f && echo "error: "$1" not found" 1>&2 && return
    tline=$(bunzip2 --stdout $f | grep --binary-files=text '#T ........ ...... ......' | head -1)
    n_tline_ent=$(echo $tline | awk '{print NF}')
    test $n_tline_ent -lt 3 && echo 0 && return
    run_start_date=$(echo $tline | awk '{print $3}')
    echo $run_start_date
    return
}

flist_all[0]=$(find -L $run_search_path -type f -name "BR??????.tar.bz2" | \
    awk '{("basename " $1 " .tar.bz2") | getline bname; fname=$1; print $fname,$bname}' \
    | sort -k2 | awk '{print $1}')
flist_all[1]=$(find -L $run_search_path -type f -name "LR??????.tar.bz2" | \
    awk '{("basename " $1 " .tar.bz2") | getline bname; fname=$1; print $fname,$bname}' \
    | sort -k2 | awk '{print $1}')
flist_all[2]=$(find -L $run_search_path -type f -name "SK??????.tar.bz2" | \
    awk '{("basename " $1 " .tar.bz2") | getline bname; fname=$1; print $fname,$bname}' \
    | sort -k2 | awk '{print $1}')
flist[0]=""
flist[1]=""
flist[2]=""

itower=0
while [ $itower -lt 3 ]; do
    nf=$(echo ${flist_all[$isite]} | awk '{print NF}')
    if [ $nf -lt 2 ]; then
	echo "     ^^^^^WARNING: number of files found for itower="$itower " is less than 2">&2
	itower=$((itower+1))
	continue
    fi
    first_file=$(echo ${flist_all[$isite]} | awk '{print $1}')
    yymmdd_f_first=$(get_sdrun_date $first_file)
    if [ $yymmdd_f_first -eq 0 ]; then
	first_file=$(echo ${flist_all[$isite]} | awk '{print $2}')
	yymmdd_f_first=$(get_sdrun_date $first_file)
    fi
    last_file=$(echo ${flist_all[$isite]} | awk '{print $NF}')
    yymmdd_f_last=$(get_sdrun_date $last_file)
    if [ $yymmdd_f_last -eq 0 ]; then
	last_file=$(echo ${flist_all[$isite]} | awk 'a=NF-1; {print $a}')
	yymmdd_f_last=$(get_sdrun_date $last_file)
    fi
    if [ $yymmdd_f_first -ge $yymmdd ]; then
	echo "Need tower files with run numbers less than that of "$(basename $first_file) \
	    "( date = $yymmdd_f_first )">&2
	exit 2
    fi
    if [ $yymmdd_f_last -le $yymmdd ]; then
	echo "Need tower files with run numbers greater than that of "$(basename $last_file) \
	    "( date = $yymmdd_f_last )">&2
	exit 2
    fi
    got_date=0
    for file in ${flist_all[$itower]}; do
	yymmdd_f=$(get_sdrun_date $file)
	test $yymmdd_f -eq 0 && continue;
	if [ $got_date -eq 0 -a $yymmdd_f -eq $yymmdd ]; then
	    got_date=1
	    # save 1 run right before the desired date
	    flist[$itower]=$first_file" "$file
	elif [ $got_date -eq 1 -a $yymmdd_f -eq $yymmdd ]; then
	    flist[$itower]=${flist[$itower]}" "$file
	elif [ $got_date -eq 1 -a $yymmdd_f -ne $yymmdd ]; then
	    # save 1 run right after the desired date
	    flist[$itower]=${flist[$itower]}" "$file
	    break
	else
	    first_file=$file
	    continue
	fi
    done
    if [ $got_date -eq 1 ]; then
	for file in ${flist[$itower]}; do
	    yymmdd_f=$(get_sdrun_date $file)
	    yy_f=${yymmdd_f:0:2}
	    year_f=$((2000+yy_f))
	    dtfl=$(basename $file .tar.bz2)'.Y'$year_f
	    tar -C $outdir -xjf $file $dtfl
	    fl=$outdir/$dtfl
	    test ! -f $fl && echo "warning: $fl not found (did not unpack properly)">&2
	    echo $fl
	done
    fi
    itower=$((itower+1))
done
