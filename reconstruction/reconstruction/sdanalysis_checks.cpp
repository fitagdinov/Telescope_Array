//
// For checking the presence of required libraries and header files
//
// C/C++
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <map>

// libz libbz2
#include <zlib.h>
#include <bzlib.h>

// DST
#include "event.h"
#include "filestack.h"

// CERN Root stuff
#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TMath.h"
#include "TMinuit.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TProfile.h"
#include "TSpectrum.h"
#include "TGraphErrors.h"

int main(void) 
{ 
  fprintf(stdout,"\nPASS CHECKS\n");
  return 0; 
}
