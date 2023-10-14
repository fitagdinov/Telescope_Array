#ifndef _sditerator_h_
#define _sditerator_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"
#include "sddstio.h"

/*****************************  GLOBAL CPP DEFINITIONS *****************/

// lenght of the file names
#define sd_fname_size 1024

/***********************************************************************/


/******************  C++ ANALYSIS **************************************/


extern void cppanalysis( FILE *outFL);

/***********************************************************************/



////////////////////////////////////////////////////////////////////////


/******  CLASS FOR HANDLING THE PROGRAM ARGUMENTS **********************/

class listOfOpt
{
 public:
  char wantFile[sd_fname_size];
  char outFile[sd_fname_size];
  bool verbose;     // verbose mode flag
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  
  listOfOpt();
  ~listOfOpt();
 private:
  bool checkOpt();  // check & make sure that the options make sense
};

 
#endif
