#include "ldffitter.h"
#include "sdxyzclf_class.h"
#include "ldffun.h"
#include "sdenergy.h"

using namespace TMath;
#define d2r DegToRad() // for simplicity

ldffitter* fit_var = 0; // accessed by the fcn
static sdxyzclf_class sdcoorclf;


// FCN to minimize when fitting into AGASA LDF.
/*
 * par[0] - core x
 * par[1] - core y
 * par[2] - scaling factor
 */

//______________________________________________________________________________
static void fcn_ldf(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
		    Int_t iflag)
{
  (void)(npar);
  (void)(gin);
  (void)(iflag);
  fit_var->R[0] = par[0];
  fit_var->R[1] = par[1];
  fit_var->S    = par[2];
  fit_var->calc_var();
  f = fit_var->chi2;
}

//____________________________________________________________________________________________

//_______________________________________________________________________________________

ClassImp(ldffitter)

ldffitter::ldffitter()
{
  gMinuit = 0;
  gqvsr = 0;
  grsdvsr = 0;
  ldfFun = 0;
  sdorigin_xy[0] = SD_ORIGIN_X_CLF;
  sdorigin_xy[1] = SD_ORIGIN_Y_CLF;
  reset_takenout_SD();
  iexclude = -1;
  fit_var = this;
}

ldffitter::~ldffitter()
{
}


bool ldffitter::add_takenout_SD(Int_t xxyy)
{
  if (n_takenout_sd >= NGSDS)
    {
      fprintf(stderr,"taken out SDs list is full\n");
      return false;
    }
  takenout_sd[n_takenout_sd] = xxyy;
  n_takenout_sd ++;
  return true;
}
void ldffitter::reset_takenout_SD()
{
  n_takenout_sd = 0;
}
int ldffitter::print_takenout_SD()
{
  int itko;
  fprintf(stdout, "\n");
  for (itko=0; itko < n_takenout_sd; itko++)
    {
      if(itko>0)
	fprintf(stdout," ");
      fprintf(stdout, "%04d",takenout_sd[itko]);
    }
  fprintf(stdout,"\n");
  fflush(stdout);
  return n_takenout_sd;
}

bool ldffitter::loadVariables(rusdraw_class *rusdraw1, rufptn_class *rufptn1,
			      rusdgeom_class *rusdgeom1)
{
  iexclude = -1;
  // Event direction
  theta = rusdgeom1->theta[2];
  phi   = rusdgeom1->phi[2]; 
  for (Int_t i = 0; i < 2; i++)
    {
      R[i]      = rufptn1->tyro_xymoments[2][i];
      coreXY[i] = rufptn1->tyro_xymoments[2][i]; 
    }    
  S = 1.0; // Starting value for LDF scale parameter
  nfpts = 0;
  napts = 0;
  memset(sds_hit, 0, (SDMON_X_MAX*SDMON_Y_MAX*sizeof(Int_t)));
  Double_t smax = 0.0;
  for (Int_t i = 0; i < rufptn1->nhits; i++)
    {
      // any SDs taken out ?
      Bool_t tko=false;
      for (Int_t itko = 0; itko < n_takenout_sd; itko++)
	{
	  if(rufptn1->xxyy[i]==takenout_sd[itko])
	    {
	      tko=true;
	      break;
	    }
	}
      if(tko) continue;
	
      // Hits in space-time cluster will be used for LDF fitting
      if (rufptn1->isgood[i] < 4)
	continue;

      // Important to know if a given SD was hit, so that we don't add the 0's for them
      Int_t ix = rufptn1->xxyy[i] / 100 - 1;
      Int_t iy = rufptn1->xxyy[i] % 100 - 1;
      sds_hit[ix][iy] = 1;

      // Exclude saturated counters
      if (rufptn1->isgood[i] == 5)
	continue;

      Double_t xy[2] = 
	{
	  rufptn1->xyzclf[i][0] - sdorigin_xy[0],
	  xy[1] = rufptn1->xyzclf[i][1] - sdorigin_xy[1]
	};
      Double_t d[2] = 
	{
	  xy[0]-R[0],
	  xy[1]-R[1]
	};
      Double_t cdist = sqrt(d[0]*d[0]+d[1]*d[1]);
      Double_t dotp = Sin(d2r*theta)*(d[0]*Cos(d2r*phi) +d[1]*Sin(d2r*phi));
      Double_t sdist = sqrt(d[0]*d[0]+d[1]*d[1] - dotp*dotp);
      
      // Maximum perpendicular distance from shower axis
      if (sdist > smax)
	smax = sdist;

      if (cdist < 0.5)
	continue;

      X[nfpts][0] = xy[0];
      X[nfpts][1] = xy[1];
      X[nfpts][2] = rufptn1->xyzclf[i][2];
      rho[nfpts] = 0.5 * (rufptn1->pulsa[i][0]+rufptn1->pulsa[i][1])/ 3.0;
      drho[nfpts] = 0.5 * sqrt(rufptn1->pulsaerr[i][0]
			       *rufptn1->pulsaerr[i][0] + rufptn1->pulsaerr[i][1]
			       *rufptn1->pulsaerr[i][1])/ 3.0;

      drho[nfpts] = 0.53 * sqrt(2.0*rho[nfpts] + 0.15*0.15*rho[nfpts]
				*rho[nfpts]);
      pflag[nfpts] = 1; // counter that had a non-zero charge in it
      xxyy_posid[nfpts] = rufptn1->xxyy[i];
      //        if(rho[nfpts] < 1.0)
      //          continue;


      nfpts++;
      napts++;
    }

  // Additional data points (counters that were not hit)


  // This is just a range of X and Y values for counters that 
  // we need to scan and see if if they are inside the event ellipse or not.
  Double_t xlo = (Int_t)(Floor(R[0] - smax/Cos(d2r*theta) ) - 1.0);
  if (xlo < 1)
    xlo = 1;
  Double_t xup = (Int_t)(Ceil(R[0] + smax/Cos(d2r*theta) ) + 1.0);
  if (xup > SDMON_X_MAX)
    xup = SDMON_X_MAX;
  
  Double_t ylo = (Int_t)(Floor(R[1] - smax/Cos(d2r*theta) ) - 1.0);
  if (ylo < 1)
    ylo = 1;
  Double_t yup = (Int_t)(Ceil(R[1] + smax/Cos(d2r*theta) ) + 1.0);
  if (yup > SDMON_Y_MAX)
    yup = SDMON_Y_MAX;

  Double_t smin = 0;
  Double_t smax1 = smax;
  if (smax > 1.5)
    {
      smin = smax - 0.5;
      smax1 = smax + 0.1;
    }
  else if (smax >= 1.0 && smax <= 1.5)
    {
      smin = smax - 0.25;
      smax1 = smax + 0.1;
    }
  else
    {
      smin = smax; // Don't add counters closer than 1200m
      smax1 = smax1;
    }
  smax = smax1;

  Int_t tower_id = rusdraw1->site;
  if (tower_id < 0 || tower_id > RUSDRAW_BRLRSK) 
    return false;

  for (Int_t ix = (xlo-1); ix < xup; ix++)
    {
      for (Int_t iy = (ylo-1); iy < yup; iy++)
	{
	  // Get the tower ID of the location
	  Int_t itowid = sdcoorclf.get_towerid(rusdraw1->yymmdd, (ix+1), (iy+1));
            
	  // For events read out only by one site
	  if ((tower_id < 3) && (itowid != tower_id))
            continue;
            
	  // For multiple site events
	  if ( tower_id >= 3 )
	    {

	      if( (tower_id == RUSDRAW_BRLR) &&
		  (itowid != RUSDRAW_BR) &&
		  (itowid != RUSDRAW_LR) )
                continue;

	      if( (tower_id == RUSDRAW_BRSK) &&
		  (itowid != RUSDRAW_BR) &&
		  (itowid != RUSDRAW_SK) )
                continue;

	      if( (tower_id == RUSDRAW_LRSK) &&
		  (itowid != RUSDRAW_LR) &&
		  (itowid != RUSDRAW_SK) )
                continue;

	      if( (tower_id == RUSDRAW_BRLRSK) &&
		  (itowid != RUSDRAW_BR) &&
		  (itowid != RUSDRAW_LR) &&
		  (itowid != RUSDRAW_SK) )
                continue;

	    }
	    
	  // if this SD position is taken out
	  Bool_t tko = false;
	  for (Int_t itko = 0; itko < n_takenout_sd; itko++)
	    {
	      if(((ix+1)*100+(iy+1))==takenout_sd[itko])
		{
		  tko=true;
		  break;
		}
	    }
	  if(tko) 
	    continue;
	    
	  // Ignore counters which don't have valid GPS coordinates
	  Double_t xyz[3] = {0, 0, 0};
	  if (!sdcoorclf.get_xyz(rusdraw1->yymmdd, (ix+1), (iy+1), &xyz[0]))
	    continue;
	  xyz[0] -= SD_ORIGIN_X_CLF; // Subtract SD origin
	  xyz[1] -= SD_ORIGIN_Y_CLF;
	    
	  if (sds_hit[ix][iy] == 1) continue;
	  Double_t d[2] = 
	    {
	      xyz[0]-R[0],
	      d[1]=xyz[1]-R[1]
	    };
	  Double_t dotp = Sin(d2r*theta)*(d[0]*Cos(d2r*phi)+d[1]*Sin(d2r*phi));
	  Double_t sdist = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
	  if (sdist < smin)
            continue;
	  if (sdist> smax)
            continue;
	  memcpy(&X[nfpts][0],&xyz[0],3*sizeof(Double_t));
	  rho[nfpts] = 0.0;
	  drho[nfpts] = 0.53 * 3.125;
	  pflag[nfpts] = 0; // zero counter put in
	  xxyy_posid[nfpts] = 100 * (ix+1) + (iy+1); // position id of the zero counter
	  nfpts ++;
	}
    }
  
  ndof = napts - NFPARS;
  if(napts<1)
    return false;

  return true;
}

//______________________________________________________________________________


// Form of LDF function for plotting it in root
// x[0]   - perpendicular distance from core, [1200m] units
// par[0] - scaling constant
// par[1] - zenith angle, degree
static Double_t ldfRootFun(Double_t *x, Double_t *par)
{
  return par[0] * ldffun(x[0]*par[2], par[1]);
}

void ldffitter::prepPlotVars()
{
    
  // ldf fuction for drawing. Define it if it hasn't been defined yet.
  if (!ldfFun)
    {
      ldfFun = new TF1("ldfFun",ldfRootFun,0.0,10e3,3);
      ldfFun->SetParameter(2, 1.2e3); // by default, use [1200m] units.  Needed here so that GetXaxis works
      ldfFun->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
      ldfFun->GetYaxis()->SetTitle("Charge Density, [VEM/m^{2}]");
      ldfFun->SetLineColor(2);
      ldfFun->SetLineWidth(3);
    }

  // Setting the LDF fit parameters
  ldfFun->SetParameter(0, S);
  ldfFun->SetParameter(1, theta);
  ldfFun->SetParameter(2, 1.2e3); // by default, want to use [1200m] units
    
  if (gqvsr)
    gqvsr->Delete();
  gqvsr = new TGraphErrors(nfpts,ltr_dist,rho,0,drho);
  gqvsr->SetTitle("#rho vs dist. from shower axis");
  gqvsr->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  gqvsr->GetYaxis()->SetTitle("Charge Density, [VEM/m^{2}]");

  if (grsdvsr)
    grsdvsr->Delete();
  grsdvsr = new TGraphErrors(nfpts,ltr_dist,rhorsd,0,drho);
  grsdvsr->SetTitle("Residuals vs dist. from shower axis");
  grsdvsr->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  grsdvsr->GetYaxis()->SetTitle("(#rho_{Actual}-#rho_{Expected}), [VEM/m^{2}]");
}

bool ldffitter::Ifit(bool fixCore, bool verbose)
{
  Int_t nfpars;
  // For passing lists of options to Minuit
  Double_t arglist[10];
  Int_t ierflg;
  
  if (gMinuit)
    gMinuit->Delete();
  gMinuit=0;

  // for plane fitting
  if (verbose)
    fprintf(stdout,"Fitting into AGASA LDF\n");
  nfpars=NFPARS;
  gMinuit = new TMinuit(nfpars);

  gMinuit->SetFCN(fcn_ldf);

  if (!verbose)
    gMinuit->SetPrintLevel(-1);

  ndof = napts-nfpars; // # of d.o.f
  if (napts < 1)
    return false;
  ierflg = 0;
  arglist[0] = 1;
  gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
  gMinuit->mnparm(0,"Core X", R[0], 0.1, 0, 0, ierflg);
  gMinuit->mnparm(1,"Core Y", R[1], 0.1, 0, 0, ierflg);
  gMinuit->mnparm(2,"Scale",  S,    0.1, 0, 0, ierflg);
  
  if (fixCore)
    {
      gMinuit->FixParameter(0);
      gMinuit->FixParameter(1);
    }

  // Do the fitting
  gMinuit->SetMaxIterations(LDF_MITR);
  gMinuit->Migrad();

  if (verbose)
    {
      // Print results
      Double_t amin, edm, errdef;
      Int_t nvpar, nparx, icstat;
      gMinuit->mnstat(amin, edm, errdef, nvpar, nparx, icstat);
      gMinuit->mnprin(3, amin);
    }

  // Obtain the fit parameters 
  gMinuit->GetParameter(0, R[0], dR[0]);
  gMinuit->GetParameter(1, R[1], dR[1]);
  gMinuit->GetParameter(2, S, dS);

  // Evaluate the LDF function at 600 and 800m
  S600=S*ldffun(600.0, theta);
  S600_0 = S600 / S600_attenuation_AGASA(theta);

   
  S800=S*ldffun(800.0, theta); // Signal size at 800m from core
  energy = rusdenergy(S800,theta);
  log10en = 18.0 + Log10(energy);
    
  chi2 = gMinuit->fAmin;
  ndof = napts-nfpars;

  if (verbose)
    {
      fprintf(stdout, "theta = %f\n",theta);
      fprintf(stdout, "S600 = %f\n",S600);
      fprintf(stdout, "S600_0 = %f\n",S600_0);
      fprintf(stdout, "S600 attenuation: %f\n",
	      S600_attenuation_AGASA(theta));
      fprintf(stdout, "S800 = %f\n",S800);
      fprintf(stdout, "log10en = %.1f\n",log10en);
      fprintf(stdout, "Energy  = %.1f EeV\n",energy);
      fprintf(stdout, "chi2 = %f\n",chi2);
      fprintf(stdout, "ndof = %d\n",ndof);
      fprintf(stdout, "Number of fit points: %d\n",nfpts);
      fprintf(stdout, "Number of non-zero SDs: %d\n",napts);
    }
  
  return true;  
}

Int_t ldffitter::clean(Double_t deltaChi2, bool fixCore, bool verbose)
{
  Int_t xxyy;
  Int_t i;
  Double_t chi2old, chi2new;
  Int_t nDeletedPts;
  Double_t dChi2;
  Int_t worst_i; // Keep track of the worst data point


  nDeletedPts = 0;
  iexclude = -1;
  Ifit(fixCore, verbose);

  do
    {
      if (nfpts < 1)
	{
	  iexclude = -1;
	  return nDeletedPts;
	}
      // Fit with current data points and save the chi2 value
      iexclude = -1;
      Ifit(fixCore, verbose);

      chi2old=chi2;
      // When chi2 is already smaller than deltaChi2
      if (chi2old < deltaChi2)
	{
	  iexclude = -1;
	  return nDeletedPts;
	}

      dChi2=0.0; // initialize the chi2 difference.
      worst_i = 0; // just to initialize it
      // Find the largest chi2 difference.  This will be the worst point.
      for (i=0; i<nfpts; i++)
	{
	  // Fit w/o the i'th data pint and take the new chi2
	  iexclude = i;
	  Ifit(fixCore, verbose);
	  chi2new=chi2;

	  //       fprintf(stdout, "chi2new=%f\n",chi2new);
	  if ((chi2old-chi2new)>dChi2)
	    {
	      dChi2=(chi2old-chi2new);
	      worst_i = i;
	    }
	}
      if (dChi2 >= deltaChi2)
	{
	  xxyy=((Int_t)(Int_t)Floor(X[worst_i][0]+0.5))*100
	    + ((Int_t)(Int_t)Floor(X[worst_i][1]+0.5));
	  remove_point(worst_i);
	  nDeletedPts ++;
	  if (verbose)
	    fprintf(stdout,"Removed point: %04d\n",xxyy);
	}

    } while (dChi2>=deltaChi2);

  iexclude = -1;
  return nDeletedPts;

}

void ldffitter::remove_point(Int_t ipoint)
{
  nfpts --;
  if (pflag[ipoint]==1)
    napts--;
  for (Int_t i=ipoint; i<nfpts; i++)
    {
      memcpy(&X[i][0], &X[i+1][0], 3*sizeof(Double_t));
      grn_dist[i]=grn_dist[i+1];
      ltr_dist[i]=ltr_dist[i+1];
      rho[i]=rho[i+1];
      drho[i]=drho[i+1];
      rhorsd[i]=rhorsd[i+1];
      pflag[i]=pflag[i+1];
      xxyy_posid[i]=xxyy_posid[i+1];
    }
}

void ldffitter::calc_var()
{
  chi2 = 0.0; // initialize the chi2 calculation
  for (Int_t i = 0; i < nfpts; i++)
    {
      if (i==iexclude)
	continue;
      Double_t d[2] = 
	{
	  X[i][0] - R[0],
	  X[i][1] - R[1]
	};
      Double_t dotp = Sin(d2r*theta)*(d[0]*Cos(d2r*phi)+d[1]*Sin(d2r*phi));
      grn_dist[i] = sqrt(d[0]*d[0]+d[1]*d[1]);
      ltr_dist[i] = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
      rhorsd[i] = (rho[i] - S * ldffun(ltr_dist[i]*1.2e3,theta));
      chi2 += rhorsd[i]*rhorsd[i]/drho[i]/drho[i];
    }   
  chi2 += ((coreXY[0]-R[0])*(coreXY[0]-R[0])+ (coreXY[1]-R[1])
	   *(coreXY[1]-R[1])) / (LDF_COG_dr*LDF_COG_dr);
}

bool ldffitter::get_param(TMinuit* g)
{
  if(!g)
    {
      Error("get_param","minuit pointer must be initialized before calling this method");
      R[0]  = 0.0;
      dR[0] = 0.0;
      R[1]  = 0.0;
      dR[1] = 0.0;
      S     = 0.0;
      dS    = 0.0;
      return false;
    }
  gMinuit->GetParameter(0, R[0], dR[0] );
  gMinuit->GetParameter(1, R[1], dR[1] );
  gMinuit->GetParameter(2, S, dS );
  calc_var(); // re-calculate all variables given the best fit parameters
  return true;
}
int ldffitter::print_points(Bool_t sort_by_r)
{
  Int_t *ldfind = new Int_t[nfpts];
  if(sort_by_r)
    TMath::Sort(nfpts,ltr_dist,ldfind,0);
  else
    for(Int_t i=0; i<nfpts; i++)
      ldfind[i] = i;
  fprintf(stdout,"%s%7s%20s%18s%15s%15s%15s\n",
	  "ldfind","xxyy","grn_dist[1200m]","ltr_dist[1200m]","rho[VEM/m^2]","drho[VEM/m^2]","fit[VEM/m^2]");
  for (Int_t i=0; i<nfpts; i++)
    {
      fprintf(stdout,"%3d%10.04d%14.2f%18.2f%18.1f%15.1f%15.1f\n",
	      ldfind[i],
	      xxyy_posid[ldfind[i]],
	      grn_dist[ldfind[i]],
	      ltr_dist[ldfind[i]],
	      rho[ldfind[i]],
	      drho[ldfind[i]],
	      rho[ldfind[i]]-rhorsd[ldfind[i]]);
    }
  fflush(stdout);
  delete[] ldfind;
  return nfpts;
}
