#ifndef _rufldf_h_
#define _rufldf_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"
#include "time.h"


class listOfOpt
{
 public:
  char wantFile [0x400];
  char dout[0x400];       // output directory
  char outfile[0x400];    // output file in case all dst output goes to one file
  bool fOverwriteMode;    // overwrite the output files if exist
  int  verbose;           // verbosity level
  bool bank_warning_opt;  // warnings about missing banks
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan();  // print out the manual

  listOfOpt();
  ~listOfOpt();
 private:
  bool checkOpt();  // check & make sure that the options make sense
  char progName[0x400]; // save the program name
};

#endif
