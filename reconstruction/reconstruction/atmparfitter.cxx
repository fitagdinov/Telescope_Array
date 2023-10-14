#include "atmparfitter.h"
#include "TString.h"
#include "sduti.h"

/////////// STARTING VALUES /////////////////////////////////////////////
///////// ATMOSPHERIC PARAMETERS BEGIN WITH DEFAULTS /////////////////
/// (CORSIKA U.S. STANDARD ATMOSPHERE PARAMETRIZED BY KEILHAUER.) ////
#define ATMPAR_NHEIGHT_USED 5 // 5 altitude (above sea level) values
const Double_t HEIGHT_US_STD_KEILHAUER[ATMPAR_NHEIGHT_USED] = {0.0, 7.0e5, 11.4e5, 37.0e5, 100.0e5};
const Double_t ATMPAR_US_STD_KEILHAUER[ATMPAR_NHEIGHT_USED][3] =
  {
    { -149.801663,   1183.6071, 954248.34 },
    { -57.932486,    1143.0425, 800005.34 },
    { 0.63631894,    1322.9748, 629568.93 },
    { 4.35453690e-4, 655.67307, 737521.77 },
    { 0.01128292,    1.0,       1.e9      }
  };

static Double_t ATMPARFITTER_HEIGHT_FIT[ATMPAR_NHEIGHT_USED];
static Double_t ATMPARFITTER_ATMPAR_FIT[ATMPAR_NHEIGHT_USED][3];


// Function that's used for fitting the atmospheric data
// number of free parameters is 6
// x[0] - heigh above sea level [cm]
// par[0]: h1 [cm]
// par[1]: h2 [cm]
// par[2]: h3 [cm]
// par[3]: c1 [cm]
// par[4]: c2 [cm]
// par[5]: c3 [cm]
// Return: density of the atmopsher in [g/cm^3] vs height in [cm], or the derivative
// of the mass overburden with respect to height
static Double_t atmpar_fit_fun_mo_vs_h_deriv(Double_t *x, Double_t *par)
{
  Double_t (&hei)[ATMPAR_NHEIGHT_USED]    = ATMPARFITTER_HEIGHT_FIT; // height boundary values
  Double_t (&atm)[ATMPAR_NHEIGHT_USED][3] = ATMPARFITTER_ATMPAR_FIT; // atmospheric paramters that are being fitted
  // boundary heights
  hei[1] = par[0];
  hei[2] = par[1];
  hei[3] = par[2];
  // re-calculate the atmospheric coefficients based on the altitude boundary values
  // and the c-coefficients that are provided as the fitting parameters
  Double_t h   = 0; // height at the right boundary
  Double_t val = 0; // mo value at the right boundary
  Double_t der = 0; // mo derivative value at the right boundary
  Double_t a = 0, b = 0, c = 0; // coefficients for a particular layer
  for (Int_t i = 2; i >= 0; i--)
    {
      h = hei[i+1];
      val = atm[i+1][0]+atm[i+1][1]*exp(-h/atm[i+1][2]);
      der = -atm[i+1][1]/atm[i+1][2]*exp(-h/atm[i+1][2]);
      c = par[3+i]; // c-coefficients are the ones that are specified as parameters
      a = val + der*c;
      b = -der * c / exp(-h/c);
      atm[i][0] = a;
      atm[i][1] = b;
      atm[i][2] = c;
    }
  h = x[0];
  for (Int_t i=0; i<ATMPAR_NHEIGHT_USED-1; i++)
    {
      if(h < hei[i+1])
	return atm[i][1]/atm[i][2]*exp(-h/atm[i][2]);
    }
  return atm[ATMPAR_NHEIGHT_USED-1][1]/atm[ATMPAR_NHEIGHT_USED-1][2];
}

// x[0] - heigh above sea level [cm]
// par[0]: h1 [cm]
// par[1]: h2 [cm]
// par[2]: h3 [cm]
// par[3]: c1 [cm]
// par[4]: c2 [cm]
// par[5]: c3 [cm]
// Return: vertical mass overburden in [g/cm^2] vs height in [cm]
static Double_t atmpar_fun_mo_vs_h(Double_t *x, Double_t *par)
{
  Double_t (&hei)[ATMPAR_NHEIGHT_USED]    = ATMPARFITTER_HEIGHT_FIT; // height boundary values
  Double_t (&atm)[ATMPAR_NHEIGHT_USED][3] = ATMPARFITTER_ATMPAR_FIT; // atmospheric paramters that are being fitted
  // boundary heights
  hei[1] = par[0];
  hei[2] = par[1];
  hei[3] = par[2];
  // re-calculate the atmospheric coefficients based on the altitude boundary values
  // and the c-coefficients that are provided as the fitting parameters
  Double_t h   = 0; // height at the right boundary
  Double_t val = 0; // mo value at the right boundary
  Double_t der = 0; // mo derivative value at the right boundary
  Double_t a = 0, b = 0, c = 0; // coefficients for a particular layer
  for (Int_t i = 2; i >= 0; i--)
    {
      h = hei[i+1];
      val = atm[i+1][0]+atm[i+1][1]*exp(-h/atm[i+1][2]);
      der = -atm[i+1][1]/atm[i+1][2]*exp(-h/atm[i+1][2]);
      c = par[3+i]; // c-coefficients are the ones that are specified as parameters
      a = val + der*c;
      b = -der * c / exp(-h/c);
      atm[i][0] = a;
      atm[i][1] = b;
      atm[i][2] = c;
    }
  h = x[0];
  for (Int_t i=0; i < ATMPAR_NHEIGHT_USED-1; i++)
    {
      if(h < hei[i+1])
  	return atm[i][0]+atm[i][1]*exp(-h/atm[i][2]);
    }
  val = atm[ATMPAR_NHEIGHT_USED-1][0]-atm[ATMPAR_NHEIGHT_USED-1][1]*h/atm[ATMPAR_NHEIGHT_USED-1][2];
  return (val >=0 ? val : 0);
}



ClassImp(atmparfitter)
atmparfitter::atmparfitter()
{
  dateFrom       = 0; // date range, sec since 1970/1/1
  dateTo         = 0;
  yymmddFrom     = 0;
  hhmmssFrom     = 0;
  yymmddTo       = 0;
  hhmmssTo       = 0;
  gMoVsHderiv    = 0;
  gPvsH          = 0;
  fMoVsHderiv    = new TF1("fMoVsHderiv",atmpar_fit_fun_mo_vs_h_deriv,0.0,100e5,6);
  fMoVsH         = new TF1("fMoVsH",atmpar_fun_mo_vs_h,0.0,100e5,6);
}

atmparfitter::~atmparfitter()
{  
  if(gMoVsHderiv)
    delete gMoVsHderiv;
  if(fMoVsHderiv)
    delete fMoVsHderiv;
  if(fMoVsH)
    delete fMoVsH;
}

bool atmparfitter::loadVariables(gdas_dst_common* gdas)
{
  Int_t npts = 0;
  dateFrom = (UInt_t)gdas->dateFrom;
  dateTo   = (UInt_t)gdas->dateTo;

  Int_t year,month,day,hour,minute,second;
  convertSec2Date((time_t)dateFrom,&year,&month,&day,&hour,&minute,&second);
  yymmddFrom = (year-2000)*10000+month*100+day;
  hhmmssFrom = hour*10000+minute*100+second;
  convertSec2Date((time_t)dateTo,&year,&month,&day,&hour,&minute,&second);
  yymmddTo   = (year-2000)*10000+month*100+day;
  hhmmssTo   = hour*10000+minute*100+second;

  if(gMoVsHderiv)
    delete gMoVsHderiv;
  gMoVsHderiv = new TGraphErrors(0);
  gMoVsHderiv->SetMarkerStyle(25);
  

  if(gPvsH)
    delete gPvsH;
  gPvsH = new TGraph(0);
  gPvsH->SetMarkerStyle(25);

  for (Int_t i=0; i < gdas->nItem; i++)
    {
      // make sure that corrupted data are not used
      if(TMath::IsNaN(gdas->height[i])      || 
	 TMath::IsNaN(gdas->pressure[i])    ||
	 TMath::IsNaN(gdas->temperature[i]) ||
	 TMath::IsNaN(gdas->dewPoint[i]))
	continue;
      if(gdas->height[i] < 0 || 1e5 * gdas->height[i] > 1e7)
	continue;
      if(gdas->pressure[i] < 0)
	continue;
      if(273.15+gdas->temperature[i] < 0)
	continue;
      if(273.15+gdas->dewPoint[i] < 0)
	continue;    
      Double_t val = 
	SDGEN::Get_Air_Density_g_cm3(100.0*gdas->pressure[i],
				     273.15+gdas->temperature[i],
				     273.15+gdas->dewPoint[i]);
      // Adjust error bars so that mean chi2/ndof is unity and
      // average residuals for all heights check in to within 1%.
      Double_t err = 0.03 * val;
      if(1e5*gdas->height[i] > 10.0e5)
	err = 0.0075 * val;
      if(1e5*gdas->height[i] > 15.0e5)
	err = 0.0045 * val;
      if(1e5*gdas->height[i] > 20.0e5)
	err = 0.0075 * val;
      gMoVsHderiv->SetPoint(npts,1e5*gdas->height[i],val);
      gMoVsHderiv->SetPointError(npts,0,err);
      gPvsH->SetPoint(npts,1e5*gdas->height[i],100.0*gdas->pressure[i]);
      npts ++;
    }
  if(gMoVsHderiv->GetN() < 3)
    {
      fprintf(stderr,"warning: number of valid data fit points is less than 3, not fitting\n");
      return false;
    }  
  return true;
}

bool atmparfitter::loadVariables(gdas_class* gdas)
{
  gdas->loadToDST();
  return loadVariables(&gdas_);
}

bool atmparfitter::loadFromRhoVsHgraph(TGraph* g_rho_vs_h)
{
  if(gMoVsHderiv)
    delete gMoVsHderiv;
  gMoVsHderiv = new TGraphErrors(0);
  gMoVsHderiv->SetMarkerStyle(25);
  Int_t npts = 0;
  for (Int_t i=0; i<g_rho_vs_h->GetN(); i++)
    {
      Double_t h   = 0;
      Double_t val = 0;
      g_rho_vs_h->GetPoint(i,h,val);
      Double_t err = 0.0075 * val;
      if(h > 10.0e5)
	err = 0.0075 * val;
      if(h > 15.0e5)
	err = 0.0045 * val;
      if(h > 20.0e5)
	err = 0.0075 * val;
      gMoVsHderiv->SetPoint(npts,h,val);
      gMoVsHderiv->SetPointError(npts,0,err);
      npts ++;
    }
  return true;
}

bool atmparfitter::loadFromRhoVsHgraphWerr(TGraphErrors* g_rho_vs_h)
{
  const Double_t err_scf = 0.25; // scaling factor from RMS values
  if(gMoVsHderiv)
    delete gMoVsHderiv;
  gMoVsHderiv = new TGraphErrors(0);
  gMoVsHderiv->SetMarkerStyle(25);
  Int_t npts = 0;
  for (Int_t i=0; i<g_rho_vs_h->GetN(); i++)
    {
      Double_t h   = 0;
      Double_t val = 0;
      g_rho_vs_h->GetPoint(i,h,val);
      Double_t errX = g_rho_vs_h->GetErrorX(i);
      Double_t errY = g_rho_vs_h->GetErrorY(i);
      gMoVsHderiv->SetPoint(npts,h,val);
      gMoVsHderiv->SetPointError(npts,err_scf*errX,err_scf*errY);
      npts ++;
    }
  return true;
}

bool atmparfitter::Fit(bool verbose)
{
  if(!gMoVsHderiv)
    {
      fprintf(stderr,"error: atmgeomfitter::Fit(): load data into gMoVsHderiv first then do the fit\n");
      return false;
    }
  // starting values are hose of US standard atmosphere
  memcpy(ATMPARFITTER_HEIGHT_FIT,HEIGHT_US_STD_KEILHAUER,sizeof(HEIGHT_US_STD_KEILHAUER));
  memcpy(ATMPARFITTER_ATMPAR_FIT,ATMPAR_US_STD_KEILHAUER,sizeof(ATMPAR_US_STD_KEILHAUER));
  fMoVsHderiv->SetParameters(ATMPARFITTER_HEIGHT_FIT[1],
			     ATMPARFITTER_HEIGHT_FIT[2],
			     ATMPARFITTER_HEIGHT_FIT[3],
			     ATMPARFITTER_ATMPAR_FIT[0][2],
			     ATMPARFITTER_ATMPAR_FIT[1][2],
			     ATMPARFITTER_ATMPAR_FIT[2][2]);
  // fMoVsHderiv->SetParLimits(0,
  // 			    ATMPARFITTER_HEIGHT_FIT[1]-0.5*ATMPARFITTER_HEIGHT_FIT[1],
  // 			    ATMPARFITTER_HEIGHT_FIT[1]+0.5*ATMPARFITTER_HEIGHT_FIT[1]);
  // fMoVsHderiv->SetParLimits(1,
  // 			    ATMPARFITTER_HEIGHT_FIT[2]-0.5*ATMPARFITTER_HEIGHT_FIT[2],
  // 			    ATMPARFITTER_HEIGHT_FIT[2]+0.5*ATMPARFITTER_HEIGHT_FIT[2]);
  // fMoVsHderiv->SetParLimits(2,
  // 			    ATMPARFITTER_HEIGHT_FIT[3]-0.5*ATMPARFITTER_HEIGHT_FIT[3],
  // 			    ATMPARFITTER_HEIGHT_FIT[3]+0.5*ATMPARFITTER_HEIGHT_FIT[3]);
  // fMoVsHderiv->SetParLimits(3,
  // 			    ATMPARFITTER_ATMPAR_FIT[0][2] - 0.5*ATMPARFITTER_ATMPAR_FIT[0][2],
  // 			    ATMPARFITTER_ATMPAR_FIT[0][2] + 0.5*ATMPARFITTER_ATMPAR_FIT[0][2]);
  // fMoVsHderiv->SetParLimits(4,
  // 			    ATMPARFITTER_ATMPAR_FIT[1][2] - 0.5*ATMPARFITTER_ATMPAR_FIT[1][2],
  // 			    ATMPARFITTER_ATMPAR_FIT[1][2] + 0.5*ATMPARFITTER_ATMPAR_FIT[1][2]);
  // fMoVsHderiv->SetParLimits(5,
  // 			    ATMPARFITTER_ATMPAR_FIT[2][2] - 0.5*ATMPARFITTER_ATMPAR_FIT[2][2],
  // 			    ATMPARFITTER_ATMPAR_FIT[2][2] + 0.5*ATMPARFITTER_ATMPAR_FIT[2][2]);
  if(gMoVsHderiv->GetN() < 3)
    return false; // don't fit if number of points less than 3
  TString fit_frm = "";
  if(!verbose)
    fit_frm += ",Q,0";
  //gMoVsHderiv->Fit(fMoVsHderiv,fit_frm,"",0,1.0e6); // Fit first 10km
  gMoVsHderiv->Fit(fMoVsHderiv,fit_frm); // the fit the rest
  fMoVsH->SetParameters(fMoVsHderiv->GetParameters());
  return true;
}


bool atmparfitter::put2atmpar(atmpar_dst_common* atmpar)
{
  if(ATMPAR_NHEIGHT_USED > ATMPAR_NHEIGHT)
    {
      fprintf(stderr,"error: atmparfitter::put2atmpar: number of layers used (%d) is larger than current DST allows (%d)\n",
	      ATMPAR_NHEIGHT_USED,ATMPAR_NHEIGHT);
      return false;
    }
  atmpar->dateFrom = dateFrom;
  atmpar->dateTo   = dateTo;
  atmpar->modelid  = ATMPAR_GDAS_MODELID;
  atmpar->nh = ATMPAR_NHEIGHT_USED;
  for (Int_t i=0; i<ATMPAR_NHEIGHT_USED; i++)
    {
      atmpar->h[i]  = ATMPARFITTER_HEIGHT_FIT[i];
      atmpar->a[i]  = ATMPARFITTER_ATMPAR_FIT[i][0];
      atmpar->b[i]  = ATMPARFITTER_ATMPAR_FIT[i][1];
      atmpar->c[i]  = ATMPARFITTER_ATMPAR_FIT[i][2];
    }
  atmpar->chi2 = fMoVsHderiv->GetChisquare();
  atmpar->ndof = (Int_t)fMoVsHderiv->GetNDF();
  if(gMoVsHderiv->GetN() < 3)
    {
      atmpar->chi2 = 1e6;
      atmpar->ndof = 1;
    }
  return true;
}

bool atmparfitter::put2atmpar(atmpar_class* atmpar)
{
  bool flg = put2atmpar(&atmpar_);
  if(!flg)
    return false;
  atmpar->loadFromDST();
  return true;
}
