#ifndef _sdtrgbk_
#define _sdtrgbk_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"

/***********  GLOBAL CPP DEFINITIONS ***********************/

// length of the file names
#define sd_fname_size 1024

// maximum number of banks
#define SDTRGBK_MAX_BANKLIST_SIZE 0x200

/**********************************************************/

////////////////////////////////////////////////////////////


/******* CLASS FOR HANDLING THE PROGRAM ARGUMENTS *********/

class listOfOpt
{
public:
  int  icrrbankoption; // if one uses ICRR banks then this specifies which banks to use: 0 - tasdevent, 2 - tasdcalibev
  char wantFile[sd_fname_size];
  char dout[0x400]; // output directory
  char outfile[0x400]; // output file in case all dst output goes to one file
  bool ignore_bsdinfo; // if set, then bsdinfo DST bank is not used (even if it's present)
  bool write_trig_only; // write out only the events that triggered
  bool write_notrig_only; // write out only events that didn't trigger
  bool fOverwriteMode; // overwrite the output files if exist
  int  verbosity; // verbosity level
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan(); // print out the manual

  listOfOpt();
  ~listOfOpt();
private:
  bool checkOpt(); // check & make sure that the options make sense
  char progName[0x400]; // save the program name
};

#endif
