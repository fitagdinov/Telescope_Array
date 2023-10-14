# Source this script to set up the SDANALYSIS build that this script is part of.
#
# Conveniently an alias like this can be defined in .bashrc:
#   alias this_sdanalysis=". bin/this_sdanalysis.sh"
#
# This script if for the bash like shells
#
# Original Author: Fons Rademakers, 18/8/2006 for CERN ROOT
#
# Copied and modified for academic research in 2020 by D. Ivanov

drop_from_path()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "drop_from_path: needs 2 arguments"
      return 1
   fi

   local p=$1
   local drop=$2

   newpath=`echo $p | sed -e "s;:${drop}:;:;g" \
                          -e "s;:${drop}\$;;g"   \
                          -e "s;^${drop}:;;g"   \
                          -e "s;^${drop}\$;;g"`
}

clean_environment()
{
    if [ -n "${old_sdanalysis}" ] ; then
	if [ -n "${PATH}" ]; then
            drop_from_path "$PATH" "${old_sdanalysis}/bin"
            PATH=$newpath
	fi
	if [ -n "${LD_LIBRARY_PATH}" ]; then
            drop_from_path "$LD_LIBRARY_PATH" "${old_sdanalysis}/lib"
            LD_LIBRARY_PATH=$newpath
	fi	
	if [ -n "${CMAKE_PREFIX_PATH}" ]; then
            drop_from_path "$CMAKE_PREFIX_PATH" "${old_sdanalysis}"
            CMAKE_PREFIX_PATH=$newpath
	fi	
    fi
}

set_environment()
{
    if [ -z "${PATH}" ]; then
	PATH=$SDDIR/bin; export PATH
    else
	PATH=$SDDIR/bin:$PATH; export PATH
    fi
    if [ ! -z "${LD_LIBRARY_PATH}" ]; then
	export LD_LIBRARY_PATH
    fi
    if [ -z "${CMAKE_PREFIX_PATH}" ]; then
	CMAKE_PREFIX_PATH=$SDDIR; export CMAKE_PREFIX_PATH       # Linux, ELF HP-UX
    else
	CMAKE_PREFIX_PATH=$SDDIR:$CMAKE_PREFIX_PATH; export CMAKE_PREFIX_PATH
    fi
}


### main ###


if [ -n "${SDDIR}" ] ; then
   old_sdanalysis=${SDDIR}
fi


SOURCE=${BASH_ARGV[0]}
if [ "x$SOURCE" = "x" ]; then
   SOURCE=${(%):-%N} # for zsh
fi


if [ "x${SOURCE}" = "x" ]; then
   if [ -f bin/this_sdanalysis.sh ]; then
      SDDIR="$PWD"; export SDDIR
   elif [ -f ./this_sdanalysis.sh ]; then
      SDDIR=$(cd ..  > /dev/null; pwd); export SDDIR
   else
      echo ERROR: must "cd where/sdanalysis/is" before calling ". bin/this_sdanalysis.sh" for this version of bash!
      SDDIR=; export SDDIR
      return 1
   fi
else
   # get param to "."
   this_sdanalysis=$(dirname ${SOURCE})
   SDDIR=$(cd ${this_sdanalysis}/.. > /dev/null;pwd); export SDDIR
fi

clean_environment
set_environment


if [ "x`root-config --arch | grep -v win32gcc | grep -i win32`" != "x" ]; then
   SDDIR="`cygpath -w $SDDIR`"
fi

unset old_sdanalysis
unset this_sdanalysis
unset -f drop_from_path
unset -f clean_environment
unset -f set_environment
