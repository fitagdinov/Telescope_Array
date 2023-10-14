#ifndef _atmparfitter_h_
#define _atmparfitter_h_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TObject.h"
#include "TMath.h"
#include "event.h"
#include "fdcalib_util.h"
#include "sdrt_class.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TProfile.h"
#include "fdrt_class.h"
#include "sdrt_class.h"

class atmparfitter: public TObject
{
public:
  
  UInt_t dateFrom; // date range, sec since 1970/1/1
  UInt_t dateTo;
  Int_t yymmddFrom;
  Int_t hhmmssFrom;
  Int_t yymmddTo;
  Int_t hhmmssTo;
  bool loadVariables(gdas_dst_common* gdas); 
  bool loadVariables(gdas_class* gdas);
  bool loadFromRhoVsHgraph(TGraph* g_rho_vs_h);
  bool loadFromRhoVsHgraphWerr(TGraphErrors* g_rho_vs_h);
  bool put2atmpar(atmpar_dst_common* atmpar);
  bool put2atmpar(atmpar_class* atmpar);
  
  TGraphErrors* GetMoVsHderiv()
  // density [g/cm^3] vs height [cm]
  { return  gMoVsHderiv; }


  TGraph* GetPvsH()
  // pressure [Pa] vs height [cm]
  { return  gPvsH; }
    
  TF1* GetMoVsHderivfit()
  { return fMoVsHderiv; }
  
  TF1* GetMoVsH()
  { return fMoVsH; }

  bool Fit(bool verbose=true);

  atmparfitter();
  virtual ~atmparfitter();

  
private:
  TGraphErrors *gMoVsHderiv;     // density [g/cm^3] vs height [cm]
  TGraph       *gPvsH;           // pressure [Pa] vs height [cm]
  TF1          *fMoVsHderiv;     // fit function for density vs height
  TF1          *fMoVsH;          // vertical mass overburden vs height
  ClassDef(atmparfitter,1)
};

#endif
