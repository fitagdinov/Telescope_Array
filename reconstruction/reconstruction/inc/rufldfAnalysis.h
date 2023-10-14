#ifndef _rufldfAnalysis_h_
#define _rufldfAnalysis_h_


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "event.h"

#include "p2geomfitter.h" // Refit pass1 geometry
#include "p2ldffitter.h"  // Fit LDF alone
#include "p2gldffitter.h" // Combined LDF and geometry fit
#include "sdxyzclf_class.h"
#include "rufldf.h"

class rufldfAnalysis
{
 public:
  
  const listOfOpt& opt;   // options passed from the  command line
  
  bool verbose;
  
  rufldfAnalysis(listOfOpt& passed_opt);
  ~rufldfAnalysis();
  void analyzeEvent();
  
  // To re-fit pass1 geometry with different clean-up cuts.
  p2geomfitter_class *p2geomfitter;
  void redo_pass1geom();   // Refits, and applies changes to DST banks
  bool refit_pass1geom();  // Just do the re-fitting
  void changeStclustAgr(); // Change the S-T cluster after geometry was refitted
  void put2rusdgeomAgr();  // Update in rusdgeom bank after geometry refit.
  
  
  
  // To do the LDF fitting and energy determination
  p2ldffitter_class *p2ldffitter;
  void ldfFit();
  
  
  // To do the combined Geometry/LDF fitting
  p2gldffitter_class *p2gldffitter;
  void gldfFit();
  
  
  // Adds information on how far event core is from the border to pass2 DST banks
  void addCoreBorderInfo();
  
 private:
   sdxyzclf_class *sdcoorclf;
  
  
};


#endif
