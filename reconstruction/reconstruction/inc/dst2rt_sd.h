#ifndef _dst2rt_sd_h_
#define _dst2rt_sd_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"
#include "sddstio.h"

/*************************  GLOBAL CPP DEFINITIONS ************************/

// lenght of the file names
#define SD_INBUF_SIZE 0x400
#define SD_DOUT_SIZE  0x400
#define SD_PREF_SIZE  0x100 
#define SD_FNAME_SIZE (SD_DOUT_SIZE+SD_PREF_SIZE+0x16)

/**************************************************************************/


/**  CLASS FOR HANDLING THE PROGRAM ARGUMENTS *****************************/

class listOfOpt
{
 public:
  char wantFile[SD_FNAME_SIZE];
  char dout[SD_DOUT_SIZE];   // output directory
  char outpr[SD_PREF_SIZE];  // prefix for the output files
  int  wt;                   // 0=make detailed and result trees, 1=only result tree, 2=only detailed tree
  char rtof[SD_FNAME_SIZE];  // result root tree output file
  char dtof[SD_FNAME_SIZE];  // detailed root tree output file
  bool verbose;       // verbose mode flag
  bool fOverwrite;    // overwrite the output files if they exist
  bool atmparopt;     // atmospheric prameter option
  bool gdasopt;       // gdas variables option
  bool etrackopt;     // event track bank
  bool sdopt;         // SD passes banks (true by default)
  bool mcopt;         // MC banks
  bool mdopt;         // MD banks
  bool tbopt;         // SD trigger backup banks
  bool fdplane_opt;   // fdplane banks (brplane and/or lrplane)
  bool fdprofile_opt; // fdprofile banks (brprofile and/or lrprofile)
  bool bsdinfo;       // information on bad sds
  bool tasdevent;     // icrr tasdevent bank
  bool tasdcalibev;   // icrr tasdcalibev bank
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan();  // print out the usage manual

  listOfOpt();
  ~listOfOpt();
 private:
  char progName[0x400];
  bool checkOpt();  // check & make sure that the options make sense
};

#endif
