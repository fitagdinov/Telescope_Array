#!/usr/bin/env python

import os
import sys
import argparse
import ROOT
from array import array
import datetime
import math


def convert_ascii_to_root_tree(ascii_files,tOut):
    Mon_Cycles={} # all monitoring cycles indexed by date and monitoring cycle
    Bad_Counters={} # monitoring cycles with bad counters indexed by date and monitoring cycle
    for asciifile in ascii_files:
        with open(asciifile,"r") as f:
            for line in f:
                s=line.split()
                yymmdd_read=int(s[0])
                hhmmss_read=int(s[1])
                xxyy_read=int(s[2])
                bitf_read=int(s[3])
                ICRRdontUse_read=int(s[4])
                mon_cycle_read=yymmdd_read*1000000+hhmmss_read
                if mon_cycle_read not in Bad_Counters:
                    Bad_Counters[mon_cycle_read] = []
                Bad_Counters[mon_cycle_read].append((xxyy_read,bitf_read,ICRRdontUse_read))
    # determine the date range using earliest and the latest date in the ASCII file
    yymmdd1=999999
    yymmdd2=0
    for mon_cycle_cur in Bad_Counters.keys():
        yymmdd_cur=mon_cycle_cur//1000000
        if yymmdd_cur < yymmdd1:
            yymmdd1 = yymmdd_cur
        if yymmdd_cur > yymmdd2:
            yymmdd2 = yymmdd_cur
    d1=datetime.date(year=(2000*10000+yymmdd1)//10000,month=(yymmdd1%10000)//100,day=yymmdd1%100)
    d2=datetime.date(year=(2000*10000+yymmdd2)//10000,month=(yymmdd2%10000)//100,day=yymmdd2%100)
    d=d1
    while d <= d2:
        yymmdd_cur=10000*(d.year-2000)+100*d.month+d.day
        for sec in range(0,86400,600):
            hhmmss_cur=10000*(sec//3600)+100*((sec%3600)//60)+(sec%60)
            mon_cycle_cur=yymmdd_cur*1000000+hhmmss_cur
            Mon_Cycles[mon_cycle_cur] = []
        d=d+datetime.timedelta(days=1)
    # put together all monitoring cycles and fill bad counter information
    # when it is appropriate
    for mon_cycle_cur,bad_counters in Bad_Counters.iteritems():
        Mon_Cycles[mon_cycle_cur].extend(bad_counters)
    # fill the ROOT tree with the monitoring cycle information
    yymmdd = array("i",[0])
    hhmmss = array("i",[0])
    day_since_080511 = array("i",[0])
    sec_since_midnight = array("i",[0])
    nsd = array("i",[0])
    xxyy = array("i",1024*[0])
    bitf = array("i",1024*[0])
    ICRRdontUse = array("i",1024*[0])
    tOut.Branch("yymmdd",yymmdd,"yymmdd/I")
    tOut.Branch("hhmmss",hhmmss,"hhmmss/I")
    tOut.Branch("day_since_080511",day_since_080511,"day_since_080511/I")
    tOut.Branch("sec_since_midnight",sec_since_midnight,"sec_since_midnight/I")
    tOut.Branch("nsd",nsd,"nsd/I")
    tOut.Branch("xxyy",xxyy,"xxyy[nsd]/I")
    tOut.Branch("bitf",bitf,"bitf[nsd]/I")
    tOut.Branch("ICRRdontUse",ICRRdontUse,"ICRRdontUse[nsd]/I")
    d0=datetime.date(year=2008,month=5,day=11)
    for mon_cycle_cur in sorted(Mon_Cycles.keys()):
        yymmdd[0] = mon_cycle_cur//1000000
        hhmmss[0] = mon_cycle_cur%1000000
        d=datetime.date(year=(2000*10000+yymmdd[0])//10000,month=(yymmdd[0]%10000)//100,day=yymmdd[0]%100)
        day_since_080511[0]=(d-d0).days
        sec_since_midnight[0]=3600*(hhmmss[0]//10000)+60*((hhmmss[0]%10000)//100)+(hhmmss[0]%100)
        counters = Mon_Cycles[mon_cycle_cur]
        nsd[0] = len(counters)
        for i in range (0,nsd[0]):
            (xxyy[i],bitf[i],ICRRdontUse[i]) = counters[i]
        tOut.Fill()
        
def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument('files',nargs='*',
                        help='pass sdmc_calib_check ASCII w/o prefixes or switches')
    parser.add_argument('-i', action='store', dest='listfile',
                        help=' <string> ascii list file with paths to sdmc_calib_check ASCII files')
    parser.add_argument('--tty', action='store_true', 
                        default=False, 
                        dest='tty_input',
                        help='pipe input sdmc_calib_check ASCII files from stdin')
    parser.add_argument('-o', action='store', dest='outdir', default="./",
                        help='<string >output directory for the root tree files')
    parser.add_argument('-o1f', action='store', dest='outfile', default=None,
                        help='<string > single output root tree file, overrides -o option') 
    parser.add_argument('-f', action='store_true', dest='fOverwrite', default=False,
                        help='force overwrite output files') 
    args = parser.parse_args()

    
    sdmc_calib_check_ascii_files_rel=[]
    if args.files != None:
        sdmc_calib_check_ascii_files_rel.extend(args.files)
    if args.listfile != None:
        try:
            f=open(args.listfile)
        except IOError,e:
            sys.stderr.write('Error: can\'t open %s\n' % (args.listfile))
            sys.exit(2)
        sdmc_calib_check_ascii_files_rel.extend(map(lambda s: s.strip(), f.readlines()))
    if args.tty_input :
        sdmc_calib_check_ascii_files_rel.extend(map(lambda s: s.strip(), sys.stdin.readlines()))
    sdmc_calib_check_ascii_files=map(lambda s: os.path.abspath(s), sdmc_calib_check_ascii_files_rel)
    sdmc_calib_check_ascii_files_all = []
    sdmc_calib_check_ascii_files_all.extend(sdmc_calib_check_ascii_files)
    for fname in sdmc_calib_check_ascii_files_all:
        if(not os.path.isfile(fname)) :
            sys.stderr.write("Warning: file \'%s\' not found;" % (fname))
            sys.stderr.write(" SKIPPING\n");
            sdmc_calib_check_ascii_files.remove(fname)
    
    # sort the files and remove any duplicates
    sdmc_calib_check_ascii_files = sorted(set(sdmc_calib_check_ascii_files))
    if(len(sdmc_calib_check_ascii_files) < 1):
        sys.stderr.write("\nError: no input files\n\n")
        sys.stderr.write("           MANUAL:\n")
        sys.stdout.write("\n")
        sys.stdout.write("Convert sdmc_calib_check ASCII files to ROOT trees\n")
        parser.print_help()
        sys.stdout.write("\n\n")
        sys.exit(1)

    # output directory
    outdir=str(args.outdir).rstrip('/')
    if not os.path.isdir(outdir):
        raise Exception("error: output directory \'{0:s}\' doesn\'t exist!\n".format(outdir));
    outdir=os.path.abspath(outdir)
    
    # function for initializing the ROOT tree
    def init_root_tree_file(outfile_name):
        fOut = ROOT.TFile(outfile_name,"recreate")
        if fOut.IsZombie():
            sys.exit(2)
        tOut = ROOT.TTree("tMon","Monitoring cycles that include mis-calibrated SD information")
        return (fOut, tOut)

    if args.outfile != None:
        if not args.fOverwrite and os.path.isfile(args.outfile):
            sys.stderr.write("error: {:s} exists; use -f to overwrite files!\n".format(args.outfile))
            sys.exit(2)
        (fOut,tOut) = init_root_tree_file(args.outfile)
        convert_ascii_to_root_tree(sdmc_calib_check_ascii_files,tOut)
        tOut.Write()
        fOut.Close()
    else:
        for asciifile in sdmc_calib_check_ascii_files:
            # make the output root tree file
            bname=os.path.basename(asciifile)
            if bname.endswith(".txt"):
                bname = bname[:-4]
            outfile=outdir+"/"+bname+".root"
            if not args.fOverwrite and os.path.isfile(outfile):
                sys.stderr.write("error: {:s} exists; use -f to overwrite files!\n".format(outfile))
                sys.exit(2)
            (fOut,tOut) = init_root_tree_file(outfile)
            convert_ascii_to_root_tree([asciifile],tOut)
            tOut.Write()
            fOut.Close()
        

if __name__ == "__main__":
    main()
