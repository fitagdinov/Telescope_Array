#ifndef _sdascii_h_
#define _sdascii_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"
#include "sddstio.h"

class listOfOpt
{
public:
  char   wantfile[0x400];     // input dst list file
  char   outfile[0x400];      // ASCII output file
  char   sfoutfile[0x400];    // ASCII output file for studying shower front structure 
  char   dstoutfile[0x400];   // dst output file, if one wants to save the full event information
  bool   f_etrack;            // fill etrack dst bank if this flag is set
  int    format;              // output format option flag
  bool   stdout_opt;          // dump event information into stdout
  bool   tb_opt;              // trigger back up option, needs sdtrgbk DST bank to work
  int    tb_delta_ped;        // if other than zero, raises or to lowers pedestal requirements for the trigger backup  
  double za_cut;              // maximum zenith angle cut, degree
  int    brd_cut;             // border cut flag
  
  double enscale;             // energy scale: multiply MC-derived SD energies
                              // by this value
  double emin;                // minimum energy cut, EeV
  
  bool   rescale_err_opt;     // re-scale theta/phi errors so that 
                              // they are 68% C.L.
  bool   fOverwrite;          // force-overwrite mode
  bool   bank_warning_opt;    // print warnings about missing banks

  bool getFromCmdLine(int argc, char **argv);
  void printOpts();
  listOfOpt();                // show the options
  void printMan();            // show the manual
  ~listOfOpt();
private:
  char progName[0x400];       // name of the program
  bool checkOpt();
};

 
#endif
