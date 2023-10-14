#ifndef _rusdhist_h_
#define _rusdhist_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"
#include "sddstio.h"

/**************************  GLOBAL CPP DEFINITIONS *********************/

// length of the file names
#define sd_fname_size 1024


/**********  CLASS FOR HANDLING THE PROGRAM ARGUMENTS *******************/

class listOfOpt
{
 public:
  char wantFile[sd_fname_size];
  char outfile[sd_fname_size];
  bool verbose; // verbose mode flag
  bool tbopt;  // trigger backup cut option
  int tbflag; // trigger backup flag
  int yymmdd_start; // start date, yymmdd format
  int yymmdd_stop;  // stop date, yymmdd format
  int e3wopt; // weight E^-3 MC to get the ankle (either 18.65 or 18.75)
  bool bank_warning_opt;
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan();
  listOfOpt();
  ~listOfOpt();
 private:
  char progName[0x400];
  bool checkOpt();  // check & make sure that the options make sense
};

#endif
