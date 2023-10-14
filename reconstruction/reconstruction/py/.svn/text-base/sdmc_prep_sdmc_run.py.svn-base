#!/usr/bin/env python

import os
import sys
import subprocess
import argparse
import re
import math
from array import array
import struct

from signal import signal, SIGPIPE, SIG_DFL


# Gives the total number of particles to generate at the given
# logarithmic energy bin center, when the number of particles at the minimum energy
# (bin center) is specified.  This is according to E^-1 power law, because by default,
# sdmc2_spctr is set up so that the DIFFERENTIAL spectrum follows E^-2 power law in each bin
# (and another program, on top of that, selects events in such a way as to make the
# correct power law indices)
def get_npart_e1(npart_log10emin,log10emin_bin_center,log10en_bin_center):
    npart_e1=float(npart_log10emin)*math.pow(10.0,-1.0*(log10en_bin_center-log10emin_bin_center))
    return npart_e1

# Gives the total number of particles to generate at the given
# logarithmic energy bin center, when the number of particles at the minimum energy
# (bin center) is specified.  Returns a constant number of events so that the
# DIFFERENTIAL flux follows E^-1 spectrum.  This is necessary only in
# special circumstances (e.g. when gamma ray showers are being generated) and
# one wants to have a spectrum that's constant in log10(E/eV) (or E^-1 in linear
# energy)
def get_npart_e0(npart_log10emin,log10emin_bin_center,log10en_bin_center):
    npart_e0=float(npart_log10emin)
    return npart_e0


pattern_energy_id=re.compile(r"""
.*DAT\d\d\d\d(?P<energy_id>\d\d).*
""",re.VERBOSE)

def get_energy_id(fname):
    energy_id=int(-1)
    match=pattern_energy_id.match(fname)
    if(match != None):
        energy_id=int(match.group("energy_id"))
    return energy_id

def exe_shell_cmd(cmd):
    p=subprocess.Popen(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell = True,
                       preexec_fn = lambda: signal(SIGPIPE, SIG_DFL))
    return p.communicate()

def get_gea_dat_bytes_from_tar_gz_file(fname,nbytes):
    if not os.path.isfile(fname):
        sys.stderr.write('warning: failed to get info from {0:s}, the file doesn\'t exist\n'
                         .format(fname))
        return None
    if not fname.endswith('.tar.gz'):
        sys.stderr.write('warning: failed to get info from {0:s}, the file doesn\'t end with \'.tar.gz\'\n'
                         .format(fname))
        return None
    gea_dat=os.path.basename(fname)[0:9]+"_gea.dat"
    cmd='tar -O -xzf {0:s} {1:s} | head -c {2:d}'.format(fname,gea_dat,nbytes)
    buf, err = exe_shell_cmd(cmd)
    if(len(err) > 0):
        sys.stderr.write("{0:s}\n".format(err))
        return None
    return buf

def get_log10en(energy_id):
    log10en=float(0)
    if energy_id >= 0 and energy_id <= 25:
        log10en = 18.0 + float(energy_id-0)*0.1
    elif energy_id >=26 and energy_id <= 39:
        log10en = 16.6 + float(energy_id-26)*0.1
    elif energy_id >= 80 and energy_id <= 85:
        log10en = 16.0 + float(energy_id-80)*0.1
    else:
        log10en = 0.0
    return log10en

def get_log10en_int(log10en):
    return int(math.floor(log10en*100+0.5))

def get_log10en_from_log10en_int(log10en_int):
    return float(log10en_int)/100.0;


def get_showlib_file_info(fname):
    NDWORD = int(273)
    ptype  = int(-1)
    energy = float(-1)
    theta  = float(-1.0)
    eventbuf=None
    if (fname.endswith("_gea.dat")):
        with open(fname,"rb") as f:
            eventbuf=struct.unpack(NDWORD*'f', f.read(NDWORD*4))
    elif (fname.endswith(".tar.gz")):
        buf = get_gea_dat_bytes_from_tar_gz_file(fname,NDWORD*4)
        if(buf != None):
            eventbuf=struct.unpack(NDWORD*'f', buf)
    else:
        sys.stderr.write('Error: file \'%s\' doesn not end with known extensions _gea.dat or .tar.gz;' % (fname))
    if eventbuf != None:
        ptype=int(math.floor(eventbuf[2]+0.5))
        energy=9.0+math.log10(eventbuf[3])
        theta=180.0/math.pi*eventbuf[10]
    return (ptype,energy,theta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files',nargs='*',
                        help='pass ALL available DAT??????_gea.dat and/or DAT??????.*.tar.gz shower library files from the command line')
    parser.add_argument('-e1', dest='e1', 
                        default=float(17.45), help='minimum log10(energy/EeV), default = 17.45')
    parser.add_argument('-e2', dest='e2', 
                        default=float(20.55), help='maximum log10(energy/EeV), default = 20.55')
    parser.add_argument('-n0', dest='n0', default=int(1e6),\
                            help='N_particles per epoch (typical epoch is 30 days) at the smallest log10[E/eV (bin center)], default=1e6')
    parser.add_argument('-outdir', dest='outdir',default='./', help='output directory, default is ./')
    parser.add_argument('-flat_logE', action='store_true',default=False,dest='flat_logE',
                        help='Prepare runs with flat log10(E/eV) spectrum (E^-1 power law in linear energy)?')
    args = parser.parse_args()
    if len(args.files) < 1:
        sys.stdout.write("\nGenerate .tothrow.txt files which contain information on what\n")
        sys.stdout.write("shower library tile file to use and how many events per calibration\n")
        sys.stdout.write("epoch (typical epoch is 30 days) from that file to generate so that\n")
        sys.stdout.write("the differential spectrum follows either E^-2 power law (default)\n")
        sys.stdout.write("or E^-1 power law (if the option -flat_logE is used)\n\n")
        parser.print_help()
        sys.stdout.write("\n")
        return
    flist_rel=[]
    for fname in args.files:
        if not os.path.isfile(fname):
            sys.stderr.write("warning: file {0:s} not found\n".format(fname))
            continue
        if get_energy_id(fname) < 0:
            sys.stderr.write("warning: unable {0:s} is not of DAT??????_gea.dat or DAT??????.*.tar.gz type\n"\
                                 .format(fname))
            continue
        flist_rel.append(fname)
    flist=map(lambda s: os.path.abspath(s), flist_rel)


    if not os.path.isdir(args.outdir):
        sys.stderr.write("error: {0:s} directory not found!\n".format(args.outdir));
        exit(2)

    outdir=os.path.abspath(args.outdir)


    flat_logE=bool(args.flat_logE)
    
    Showlib_files={}
    
    # integer represenation of minimum and maximum energies
    emin_int = get_log10en_int(float(args.e1))
    emax_int = get_log10en_int(float(args.e2))
    
    # minimum energy that corresponds to the shower library ID
    # (this is centered on a bin, without interpolating by +/- 0.05 in log10(E/eV))
    # this value is determined when we iterate over all shower library files while
    # applying minimum and maximum energy cut bounds
    emin_showlib_file=21.0
    
    for fname in flist:
        energy_id = get_energy_id(fname)
        log10en = get_log10en(energy_id)
        log10en_int=get_log10en_int(log10en)
        if log10en_int < emin_int or log10en_int > emax_int:
            continue
        if log10en < emin_showlib_file:
            emin_showlib_file = log10en;
        if log10en_int not in Showlib_files.keys():
            Showlib_files[log10en_int]=[]
        Showlib_files[log10en_int].append(fname)
        
    for log10en_int in sorted(Showlib_files.keys()):
        n_showlib_files=len(Showlib_files[log10en_int])
        log10en = get_log10en_from_log10en_int(log10en_int)
        npart_per_energy_bin_per_epoch=get_npart_e1(float(args.n0),emin_showlib_file,log10en)
        # if flat log10(E/eV) spectrum option is used then generate a constant number of events
        # per data epoch per energy bin
        if flat_logE:
            npart_per_energy_bin_per_epoch=get_npart_e0(float(args.n0),emin_showlib_file,log10en)
        npart_per_showlib_file_per_epoch=float(npart_per_energy_bin_per_epoch)/float(n_showlib_files)
        for fname in Showlib_files[log10en_int]:
            (ptype,energy,theta)=get_showlib_file_info(fname)
            outfname=outdir
            outfname+="/"
            outfname+=os.path.basename(fname)
            outfname+=".tothrow.txt"
            outbuf =""
            outbuf += "SHOWLIB_FILE {0:s}\n".format(fname)
            outbuf += "NPARTICLE_PER_EPOCH {0:.9e}\n".format(npart_per_showlib_file_per_epoch)
            outbuf += "PTYPE {0:d}\n".format(ptype)
            outbuf += "ENERGY {0:.2f}\n".format(energy)
            outbuf += "THETA {0:.6f}\n".format(theta)
            with open(outfname,"w") as f:
                f.write(outbuf)
            
        
if __name__=="__main__":
    main()
