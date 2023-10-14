#ifndef _prepmc_h_
#define _prepmc_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"


// lenght of the file names
#define sd_fname_size 1024

/*********************  CLASS FOR HANDLING THE PROGRAM ARGUMENTS **************/
class listOfOpt
{
 public:
  char wantFile[sd_fname_size];
  char dout[0x400];  // output directory
  char outpr[0x400]; // prefix for the output files
  bool verbose;      // verbose mode flag
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan();  // print the manual
  listOfOpt();
  ~listOfOpt();
 private:
  char progName[0x400];
  bool checkOpt();  // check & make sure that the options make sense
};


#endif
