#!/usr/bin/env bash

############################################################

# Take the most useful Root classes from TDSTio package 
# combine them into single files which describe all of 
# SD root tree classes and some of the useful FD root tree classes. 
# These combined Root tree classes will use one 
# common LinkDef file that's needed by rootcint.
# Results of this script are meant to be put into sdfdrt folder
# Dmitri Ivanov <dmiivanov@gmail.com>
# Last Modified: 20200115
############################################################

echo "Update source code using latest TDSTio package [Y/N/?] ?"
doit=0
while [ 1 ]; do
    read junk
    case "$junk" in
        y|Y) 
            doit=1
            break
            ;;
        n|N) 
            doit=0
            break
            ;;
        ?)
            echo "This will update fdrt_class.h,fdrt_class.cxx,sdrt_class.h,sdrt_class.cxx,LinkDef.h"
	    echo "with latest classes in TDSTio package, if TDSTio is installed. Continue [Y/N/?] ?"
            ;;
        *)  
            echo "[Y/N] ?" 
            continue
            ;;
    esac
done

test $doit -eq 0 && echo "Quit ..." && exit 0


# Prototype DST class that is common to both SD and FD
sdfdlist="dstbank"

# List of useful SD classes
sdlist="talex00 rusdraw rusdmc rusdmc1 showlib bsdinfo sdtrgbk"
sdlist=$sdlist" tasdevent tasdcalib tasdcalibev"
sdlist=$sdlist" rufptn rusdgeom rufldf etrack"
sdlist=$sdlist" atmpar"

# List of useful FD classes
fdlist="fdraw brraw  lrraw"
fdlist=$fdlist" fdplane brplane lrplane"
fdlist=$fdlist" fdprofile brprofile lrprofile"
fdlist=$fdlist" fdtubeprofile brtubeprofile lrtubeprofile"
fdlist=$fdlist" hbar hraw1 mc04 mcraw stps2 stpln hctim hcbin prfc"
fdlist=$fdlist" fdatmos_param gdas"

if [ -z $TDSTio ]; then
    echo "error: environmental variable TDSTio is not set">&2
    echo "set it to /full/path/to/properly/installed/TDSTio">&2
    exit 2
fi
if [ ! -d $TDSTio ]; then
    echo "error: environmental variable TDSTio is not properly set">&2
    echo "set it to /full/path/to/properly/installed/TDSTio">&2
    exit 2
fi

echo "Synchronizing with the latest TDSTio ($TDSTio) ..."

# where to find includes for these classes
inc=$TDSTio/inc

# where to find the sources for these classes
src=$TDSTio/src/lib


#//////////////////// Common LinkDef Requests (BELOW) ///////////////
echo '#ifdef __CINT__'>LinkDef.h
echo 'using namespace std;'>>LinkDef.h
echo '#pragma link off all globals;'>>LinkDef.h
echo '#pragma link off all classes;'>>LinkDef.h
echo '#pragma link off all functions;'>>LinkDef.h
echo "#pragma link C++ class vector <Byte_t>;">>LinkDef.h
echo "#pragma link C++ class vector <Char_t>;">>LinkDef.h
echo "#pragma link C++ class vector <Short_t>;">>LinkDef.h
echo "#pragma link C++ class vector <Int_t>;">>LinkDef.h
echo "#pragma link C++ class vector <Float_t>;">>LinkDef.h
echo "#pragma link C++ class vector <Double_t>;">>LinkDef.h
echo "#pragma link C++ class vector <vector <Byte_t> >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <vector <Byte_t> > >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <Char_t> >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <vector <Char_t> > >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <Short_t> >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <vector <Short_t> > >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <Int_t> >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <vector <Int_t> > >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <Float_t> >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <vector <Float_t> > >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <Double_t> >;">>LinkDef.h
echo "#pragma link C++ class vector <vector <vector <Double_t> > >;">>LinkDef.h
#//////////////////// Common LinkDef Requests (ABOVE) ///////////////

#//////////////////// Prototype DST bank class (BELOW) /////////////
echo '#include <stdio.h>'>sdfdrt_class.h
echo '#include <stdlib.h>'>>sdfdrt_class.h
echo '#include <string.h>'>>sdfdrt_class.h
echo '#include <vector>'>>sdfdrt_class.h
echo '#include "event.h"'>>sdfdrt_class.h
echo '#include "sdanalysis_icc_settings.h"'>>sdfdrt_class.h
echo '#include "TNamed.h"'>>sdfdrt_class.h
echo '#include "sdfdrt_class.h"'>sdfdrt_class.cxx

for b in $sdfdlist; do
    c=$b'_class'
    incfile=$inc/$c'.h'
    srcfile=$src/$c'.cxx'
    test ! -f ${incfile} && echo "error: ${incfile} is missing!" && exit 2
    test ! -f ${srcfile} && echo "error: ${srcfile} is missing!" && exit 2
    cat $incfile | grep -v '#include' >>sdfdrt_class.h
    cat $srcfile | grep -v '#include' >>sdfdrt_class.cxx    
    echo "#pragma link C++ class $c;" >>LinkDef.h    
done

#//////////////////// Prototype DST bank class (ABOVE) /////////////

#//////////////////// SD bank classes (BELOW) //////////////////////
echo '#include "sdfdrt_class.h"'>sdrt_class.h
echo '#include "sdrt_class.h"'>sdrt_class.cxx
for b in $sdlist; do    
    c=$b'_class'
    incfile=$inc/$c'.h'
    srcfile=$src/$c'.cxx'
    test ! -f ${incfile} && echo "error: ${incfile} is missing!" && exit 2
    test ! -f ${srcfile} && echo "error: ${srcfile} is missing!" && exit 2
    cat $incfile | grep -v '#include' >>sdrt_class.h
    cat $srcfile | grep -v '#include' >>sdrt_class.cxx
    if [ $b == "tasdevent" ]; then
	echo "#pragma link C++ class SDEventSubData_class;">>LinkDef.h
	echo "#pragma link C++ class vector <SDEventSubData_class>;">>LinkDef.h
    fi
    if [ $b == "tasdcalib" ]; then
	echo "#pragma link C++ class SDCalibHostData_class;">>LinkDef.h
	echo "#pragma link C++ class SDCalibSubData_class;">>LinkDef.h
	echo "#pragma link C++ class SDCalibWeatherData_class;">>LinkDef.h
	echo "#pragma link C++ class vector <SDCalibHostData_class>;">>LinkDef.h
	echo "#pragma link C++ class vector <SDCalibSubData_class>;">>LinkDef.h
	echo "#pragma link C++ class vector <SDCalibWeatherData_class>;">>LinkDef.h
    fi    
    if [ $b == "tasdcalibev" ]; then
	echo "#pragma link C++ class SDCalibevData_class;">>LinkDef.h
	echo "#pragma link C++ class SDCalibevWeatherData_class;">>LinkDef.h
	echo "#pragma link C++ class SDCalibevSimInfo_class;">>LinkDef.h
	echo "#pragma link C++ class vector <SDCalibevData_class>;">>LinkDef.h
	echo "#pragma link C++ class vector <SDCalibevWeatherData_class>;">>LinkDef.h
    fi
    echo "#pragma link C++ class $c;" >>LinkDef.h 
done
#//////////////////// SD bank classes (ABOVE) //////////////////////

#//////////////////// FD bank classes (BELOW) //////////////////////
echo '#include "sdfdrt_class.h"'>fdrt_class.h
echo '#include "fdrt_class.h"'>fdrt_class.cxx

for b in $fdlist; do
    c=$b'_class'
    incfile=$inc/$c'.h'
    srcfile=$src/$c'.cxx'
    test ! -f ${incfile} && echo "error: ${incfile} is missing!" && exit 2
    test ! -f ${srcfile} && echo "error: ${srcfile} is missing!" && exit 2
    cat $incfile | grep -v '#include' >>fdrt_class.h
    cat $srcfile | grep -v '#include' >>fdrt_class.cxx    
    echo "#pragma link C++ class $c;" >>LinkDef.h    
done

#//////////////////// FD bank classes (ABOVE) //////////////////////

#//////////////////// Finilize the LinkDef file ////////////////////
echo '#endif'>>LinkDef.h

echo "Done"

exit 0

