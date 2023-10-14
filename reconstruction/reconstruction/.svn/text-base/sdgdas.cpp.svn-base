#include <stdio.h>
#include <stdlib.h>
#include "sduti.h"
#include "event.h"
#include "TMath.h"
#include "TF1.h"

const Double_t GDAS_MAXIMUM_HEIGHT = 30.0; // maximum gdas height in km
static Int_t gdas_npts  = 0;
static Double_t gdas_hei[FDATMOS_PARAM_MAXITEM];
static Double_t gdas_rho[FDATMOS_PARAM_MAXITEM];
static Double_t gdas_t_K[FDATMOS_PARAM_MAXITEM];


static void load_gdas_values()
{
  gdas_npts = 0;
  for (int i=0; i<gdas_.nItem; i++)
    {
      // make sure that corrupted data are not used
      if(TMath::IsNaN(gdas_.height[i])      || 
         TMath::IsNaN(gdas_.pressure[i])    ||
         TMath::IsNaN(gdas_.temperature[i]) ||
         TMath::IsNaN(gdas_.dewPoint[i]))
        continue;
      if(gdas_.height[i] < 0 || 1e5 * gdas_.height[i] > 1e7)
        continue;
      if(gdas_.pressure[i] < 0)
        continue;
      if(273.15+gdas_.temperature[i] < 0)
        continue;
      if(273.15+gdas_.dewPoint[i] < 0)
        continue;    
      gdas_hei[gdas_npts] = 1.0e5 * gdas_.height[i];
      gdas_rho[gdas_npts] = SDGEN::Get_Air_Density_g_cm3(100.0*gdas_.pressure[i],
							 273.15+gdas_.temperature[i],
							 273.15+gdas_.dewPoint[i]);
      gdas_t_K[gdas_npts] = 273.15+gdas_.temperature[i];
      gdas_npts ++;
    }
}

static Double_t get_gdas_rho_function(Double_t *h_cm, Double_t *par)
{
  (void)(par); // par is only needed because ROOT TF1 puts that
  if(gdas_npts < 1)
    {
      fprintf(stderr,"warning: NO GDAS DATA for rho interpolation\n");
      return 0.0;
    }
  Double_t val = SDGEN::linear_interpolation(gdas_npts,gdas_hei,gdas_rho,h_cm[0]);
  if(val < 0)
    val = 0.0;
  return val;
}

static TF1 *f_gdas_rho_function = 0;


double SDGEN::get_gdas_mo_numerically(double h_cm)
{
  if(!f_gdas_rho_function)
    f_gdas_rho_function = new TF1("f_gdas_rho_function",
				  get_gdas_rho_function,
				  0.0,1e5*GDAS_MAXIMUM_HEIGHT,
				  0);
  load_gdas_values();
  if(gdas_npts < 3)
    return -1e6;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,0,0)
  return f_gdas_rho_function->Integral(h_cm,1e5*GDAS_MAXIMUM_HEIGHT,1e-3);
#else
  return f_gdas_rho_function->Integral(h_cm,1e5*GDAS_MAXIMUM_HEIGHT,(const Double_t*)0,1e-3);
#endif
}


bool SDGEN::check_gdas()
{
  load_gdas_values();
  if(gdas_npts < 3)
    return false; // return false if number of good gdas points is less than 3
  return true;    // return true otherwise
}


double SDGEN::get_gdas_rho_numerically(double h_cm)
{
  if(!f_gdas_rho_function)
    f_gdas_rho_function = new TF1("f_gdas_rho_function",
				  get_gdas_rho_function,
				  0.0,1e5*GDAS_MAXIMUM_HEIGHT,
				  0);
  load_gdas_values();
  if(gdas_npts < 3)
    return -1e6;
  return f_gdas_rho_function->Eval(h_cm);
}

double SDGEN::get_gdas_temp(double h_cm)
{
  load_gdas_values();
  if(gdas_npts < 1)
    {
      fprintf(stderr,"warning: no gdas data for temperature interpolation\n");
      return 0.0;
    }
  return SDGEN::linear_interpolation(gdas_npts,gdas_hei,gdas_t_K,h_cm);
}
