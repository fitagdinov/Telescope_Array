#ifndef _rufptn_h_
#define _rufptn_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"


/********  CLASS FOR COMMAND LINE ARGUMENTS **************************/
class listOfOpt
{
 public:
  // Indicate whether ICRR dst bank is present so that conversion happens and the bank
  // is saved in the output dst files
  bool useICRRbank;
  char wantFile [0x400];
  char dout[0x400];        // output directory
  char outfile[0x400];     // output file in case all dst output goes to one file
  bool fOverwriteMode;     // overwrite the output files if exist
  int  verbose;            // verbosity level
  bool bank_warning_opt;   // display warnings about missing banks
  bool ignore_bsdinfo;     // ignore bsdinfo DST bank if present 
  //                          (by default, use bsdinfo DST bank and remove bad counters from reconstruction)
  double stc;              // speed of light to use in space-time cluster analysis: 
                           // one can set it to be less than one, if necessary
  char bad_sd_file[0x400]; // information on bad SDs ecountered during the analysis
  FILE *bad_sd_fp;         // file pointer to the bad SD file ( if zero then SD information is not written)
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
