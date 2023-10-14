#! /usr/bin/env python
import os
import sys
import subprocess
import fnmatch
import datetime
from signal import signal, SIGPIPE, SIG_DFL
import argparse

def yymmdd2date(YYMMDD):
    yr=2000+int(int(YYMMDD)/10000)
    mo=int(int(int(YYMMDD)%10000)/100)
    da=int(int(YYMMDD)%100)
    return datetime.date(year=yr,month=mo,day=da)

def date2yymmdd(date):
    yr=int(date.year)-2000
    mo=int(date.month)
    da=int(date.day)
    return (10000*yr+mo*100+da)

# get the start date of the tale run
def get_sdrun_start_date(fname):
    if not os.path.isfile(fname):
        sys.stderr.write('warning: failed to get date for {0:s}, the file doesn\'t exist\n'
                         .format(fname))
        return None
    if not fname.endswith('.tar.bz2'):
        sys.stderr.write('warning: failed to get date for {0:s}, the file doesn\'t end with \'.tar.bz2\'\n'
                         .format(fname))
        return None
    cmd='bunzip2 --stdout {0:s}'.format(fname)
    cmd=cmd+' | grep --binary-files=text \'#T ........ ...... ......\' | head -1'
    p=subprocess.Popen(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell = True,
                       preexec_fn = lambda: signal(SIGPIPE, SIG_DFL))
    out, err = p.communicate()
    tline=out.split()
    if len(tline) < 3 :
        sys.stderr.write('warning: failed to get date for {0:s}\n'.format(fname))
        return None
    yymmdd=int(tline[2])
    return yymmdd2date(yymmdd)

# get the run id of the tale run
def get_sdrun_runid(fname):
    bname=os.path.basename(fname)
    try:
        runid=int(bname[2:8])
    except:
        sys.stderr.write('warning: failed to get runid for {0:s}\n'.format(fname))
        return None
    return runid


def find_sdrun_files(path):
# to find all tale .tar.bz2 run files in some path
    result = [[],[],[]]
    pattern='BR??????.tar.bz2'
    for root, dirs, files in os.walk(path,followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result[0].append(os.path.join(root, name))
    pattern='LR??????.tar.bz2'
    for root, dirs, files in os.walk(path,followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result[1].append(os.path.join(root, name))
    pattern='SK??????.tar.bz2'
    for root, dirs, files in os.walk(path,followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result[2].append(os.path.join(root, name))
    return result


def main():

    # getting inputs from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', action='store', dest='yymmdd1', default=None,
                        help='<int> specify the start date, yymmdd format')
    parser.add_argument('-d2', action='store', dest='yymmdd2', default=None,
                        help='<int> specify the end date, yymmdd format')
    parser.add_argument('-i', action='store', dest='indir', default=None,
                        help='<string> specify the search path for the .tar.bz2 files')
    parser.add_argument('-o', action='store', dest='outdir', default=None,
                        help='<string >specify the output directory for the list files')
    # print help and exit if no arguments are given
    if (len(sys.argv)==1):
        sys.stdout.write("\n");
        sys.stdout.write("Make lists of SD run files (.tar.bz2 files) for some specified range of dates\n");
        parser.print_help()
        sys.stdout.write("\n\n")
        sys.exit(1)
    # parsing the arguments
    args = parser.parse_args()
    if(args.yymmdd1 == None):
        sys.stdout.write("error: start date not given\n");
        sys.exit(2)
    yymmdd1=int(args.yymmdd1)
    if(args.yymmdd2 == None):
        sys.stdout.write("error: end date not given\n");
        sys.exit(2)
    yymmdd2=int(args.yymmdd2)
    if(args.indir == None):
        sys.stdout.write("error: .tar.bz2 file search path not given\n");
        sys.exit(2)
    indir=str(args.indir)
    if not os.path.isdir(indir):
        sys.stdout.write("error: .tar.bz2 file search path doesn\'t exist!\n");
        sys.exit(2)
    indir=os.path.abspath(indir)
    if(args.outdir == None):
        sys.stdout.write("error: output directory not given\n");
        sys.exit(2)
    outdir=str(args.outdir).rstrip('/')
    if not os.path.isdir(outdir):
        sys.stdout.write("error: output directory doesn\'t exist!\n");
        sys.exit(2)
    outdir=os.path.abspath(outdir)
    start_date=yymmdd2date(yymmdd1)
    end_date=yymmdd2date(yymmdd2)
    if((end_date-start_date).days < 0):
        sys.stdout.write("error: end date smaller than the start date!\n");
        sys.exit(2)
    wanted_dates=[] # create a list of wanted dates
    cur_date=start_date
    while cur_date <= end_date:
        wanted_dates.append(cur_date)
        cur_date=cur_date+datetime.timedelta(days=1)

    # Class that describes one sd run file
    class sdrunfile_class:
        def __init__(self):
            self.name=str('')
            self.date=None
            self.runid=int(0)
        def GetFileInfo(self,fname):
            self.name=fname
            self.date=get_sdrun_start_date(fname)
            self.runid=get_sdrun_runid(fname)
            return


    # find all the SD run files
    flist=find_sdrun_files(indir)

    # SD run files indexed by the date 
    sdrun_files=[{},{},{}]
    # parse dates for all files

    for itower in range(0,3):
        for f in flist[itower]:
            t = sdrunfile_class()
            t.GetFileInfo(f)
            # skip runs for which the date or the run id weren't determined
            if t.date == None or t.runid == None :
                continue
            if(t.date < start_date-datetime.timedelta(days=1) or 
               t.date > end_date+datetime.timedelta(days=1)):
                continue
            if not t.date in sdrun_files[itower]:
                sdrun_files[itower][t.date]={}
            sdrun_files[itower][t.date][t.runid]=t

   
    # prepare a list of tale run files for a particular date
    def prepare_flist(d):
        
        runlist=[] # contains the result: a list of run files for the given date
        problems=[] # list of problems for the date, if any
        towerName=['BR','LR','SK']
        full_runid_check=[{},{},{}] # for checking the run ID continuity
        
        for itower in range(0,3):
            # First take 1 (latest) run from the previous date
            d1=d-datetime.timedelta(days=1)
            t=None
            if d1 in sdrun_files[itower]:
                if len(sdrun_files[itower][d1]) != 0:
                    runid=int(max(k for k in sdrun_files[itower][d1].iterkeys()))
                    t=sdrun_files[itower][d1][runid]
            if t != None:        
                runlist.append(sdrun_files[itower][d1][runid])
                full_runid = d1.year * 1000000 + runid
                full_runid_check[itower][full_runid] = d1;
            else:
                problems.append('warning: failed to find a 1-day earlier run for date {0:06d} tower {1:s}\n'
                                 .format(date2yymmdd(d),towerName[itower]))

            # Now take all the runs (in a sorted fashion) for the current date
            extlist=[]
            if d in sdrun_files[itower]:
                if len(sdrun_files[itower][d]) != 0:
                    for runid in sorted(sdrun_files[itower][d].keys()):
                        extlist.append(sdrun_files[itower][d][runid])
                        full_runid = d.year * 1000000 + runid
                        full_runid_check[itower][full_runid] = d;
            if len(extlist) != 0 :
                runlist.extend(extlist)
            else:
                problems.append('warning: no runs found for date {0:06d} tower {1:s}\n'
                                .format(date2yymmdd(d),towerName[itower]))

            # Finally take 1 (earliest) run from the next date
            d2=d+datetime.timedelta(days=1)
            t=None
            if d2 in sdrun_files[itower]:
                if len(sdrun_files[itower][d2]) != 0:
                    runid=int(min(k for k in sdrun_files[itower][d2].iterkeys()))
                    t=sdrun_files[itower][d2][runid]
            if t != None:        
                runlist.append(sdrun_files[itower][d2][runid])
                full_runid = d2.year * 1000000 + runid
                full_runid_check[itower][full_runid] = d2;
            else:
                problems.append('warning: failed to find a 1-day later run for date {0:06d} tower {1:s}\n'
                                .format(date2yymmdd(d),towerName[itower]))
            # check the continuity of the runid
            full_runids=sorted(full_runid_check[itower].keys())
            n_full_runids=len(full_runids)
            for i in range(0,n_full_runids-1):
                rndiff=full_runids[i+1]-full_runids[i]-1
                if rndiff != 0:
                    if (full_runids[i+1] / 1000000 - full_runids[i] / 1000000 == 1 \
                            and
                        full_runids[i+1] % 1000000 == 1):
                        rndiff=0
                    else:
                        problems.append('warning: missing runs {0:06d} to {1:06d} for date {2:06d} tower {3:s}\n'
                                        .format(full_runids[i] % 1000000, \
                                                    full_runids[i+1] % 1000000, \
                                                    date2yymmdd(d),towerName[itower]))
        # return the answer
        return (runlist,problems)

    
    # iterate over requested dates and prepare the run lists
    for d in wanted_dates:
        (runlist,problems)=prepare_flist(d)
        suf = str('ok')
        if(len(problems) > 0):
            suf=str('bad')
        if len(runlist) > 0:
            outfile=outdir+'/'+'want_tasd_{0:6d}.{1:s}.txt'.format(date2yymmdd(d),suf) # output file name
            with open(outfile, 'w') as f:
                for t in runlist:
                    f.write(t.name)
                    f.write('\n')
        if(len(problems) == 0):
            sys.stdout.write('No problems on {0:06d}\n'.format(date2yymmdd(d)))
        else:
            sys.stderr.write('******** Problems for {0:06d} (below): *********** \n'.format(date2yymmdd(d)))
            for p in problems:
                sys.stderr.write(p)
            sys.stderr.write('******** Problems for {0:06d} (above): *********** \n'.format(date2yymmdd(d)))

if __name__ == "__main__":
    main()

