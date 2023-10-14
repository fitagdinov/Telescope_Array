#!/usr/bin/env python

import os
import sys
import argparse
import re
import math
from array import array
import struct
import numpy
from bisect import bisect_left

# start of the line in the ASCII files for the theta values
# for each calibration epoch
ICAL_THETA_LINE = "ICAL_THETA"

pattern_file_info=re.compile(r"""
.*DAT(?P<dat_file_id>\d\d\d\d\d\d)_gea.dat_cal_(?P<cal1>\d+)_(?P<cal2>\d+).tothrow.txt
""",re.VERBOSE)


def get_file_info(fname):
    dat_file_id=int(-1)
    cal1=int(-1)
    cal2=int(-1)
    match=pattern_file_info.match(fname)
    if(match != None):
        dat_file_id=int(match.group("dat_file_id"))
        cal1=int(match.group("cal1"))
        cal2=int(match.group("cal2"))
    else:
        sys.stderr.write("Error! fail to parse {0:s}".format(fname))
    return (dat_file_id,cal1,cal2)

class tothrow_file:
    def __init__(self,name):
        self.name=str(name)
        a=get_file_info(name)
        self.dat_file_id=a[0]
        self.cal1 = a[1]
        self.cal2 = a[2]
        self.dat_file_name="DAT{0:06d}_gea.dat".format(self.dat_file_id)
    def __eq__(self,another):
        return hasattr(another, 'dat_file_id') and \
            self.dat_file_id == another.dat_file_id and \
            self.cal1 == another.cal1 and \
            self.cal2 == another.cal2
    def __lt__(self,another):
        return self.cal2 < another.cal1
    def __str__(self):
        return self.name
    






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files',nargs='*',
                        help='pass tothrow files DAT??????_gea.dat_cal_?_?.tothrow.txt from the command line')
    parser.add_argument('-i',dest='listfile',default=None,\
                            help='ASCII list file with paths to DAT??????_gea.dat_cal_?_?.tothrow.txt files')
    parser.add_argument('-outdir', dest='outdir',default='./', help='output directory, default is ./')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    flist_rel=[]
    for fname in args.files:
        if not os.path.isfile(fname):
            sys.stderr.write("warning: file {0:s} not found\n".format(fname))
            continue
        if get_file_info(fname)[0] < 0:
            sys.stderr.write("warning: unable {0:s} is not of DAT??????_gea.dat_cal_?_?.tothrow.txt type\n"\
                                 .format(fname))
            continue
        flist_rel.append(fname)


    if args.listfile != None:
        with open(args.listfile,"r") as f:
            flist_rel.extend(map(lambda s: s.strip(), f.readlines()))

    flist=map(lambda s: os.path.abspath(s), flist_rel)


    if not os.path.isdir(args.outdir):
        sys.stderr.write("error: {0:s} directory not found!\n".format(args.outdir));
        exit(2)

    outdir=os.path.abspath(args.outdir)
    Tothrow_files={}
    
    sys.stdout.write("Found {0:d} DAT??????_gea.dat_cal_?_?.tothrow.txt files\n".\
                         format(len(flist)));
    sys.stdout.flush();
    for fname in flist:
        tothrowfile=tothrow_file(fname)
        if tothrowfile.dat_file_id not in Tothrow_files.keys():
            Tothrow_files[tothrowfile.dat_file_id] = []
        Tothrow_files[tothrowfile.dat_file_id].append(tothrowfile)
        
    for dat_file_id in Tothrow_files.keys():
        flist=sorted(Tothrow_files[dat_file_id])
        f1=flist[0]
        f2=flist[len(flist)-1]
        cal1=f1.cal1
        cal2=f2.cal2
        outfile=outdir
        outfile+="/"
        outfile+=f1.dat_file_name
        outfile+="_cal_{0:d}_{1:d}".format(cal1,cal2)
        outfile+=".tothrow.txt"
        with open(outfile,"w") as ofl:
            for i in range(0,len(flist)):
                if i== 0:
                    with open(flist[i].name,"r") as f:
                        for line in f:
                            if line.startswith("CALIBRATION_EPOCHS"):
                                s="CALIBRATION_EPOCHS {0:d} {1:d}\n".format(cal1,cal2)
                                ofl.write(s)
                            else:
                                ofl.write(line)
                else:
                    with open(flist[i].name,"r") as f:
                        for line in f:
                            if line.startswith(ICAL_THETA_LINE):
                                ofl.write(line)
                            
        
if __name__=="__main__":
    main()
