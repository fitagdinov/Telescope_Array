#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"

#include "TF1.h"
#include "TH1F.h"
#include "TMath.h"

using namespace TMath;




// To facilitate variable transfer b/w tasdmonitor and 1MIP fitter w/o changing 1MIP fitter too much
typedef struct
{
  integer4 hmip[2][RUSDCAL_NMONCHAN];	  /* 1mip histograms (0-lower, 1-upper) */
  integer4 hped[2][RUSDCAL_NMONCHAN / 2]; /* pedestal histogram */ 
} monsimple_struct;


monsimple_struct *monsimple;


TH1F *hmip[2];  // dummy histogram for fitting 1MIP distribution
TF1 *ffit; // dummt variable for the fit function


void prep_rusdcal(tasdmonitor_dst_common *mon, rusdcal_dst_common *cal);
void prepMonInfo(int isd, monsimple_struct *mon, rusdcal_dst_common *cal);
void mipFIT(integer4 isd, rusdcal_dst_common *cal);

int main (int argc, char **argv)
{

  // program arguments
  char inFile[0x400];
  char outFile[0x400]; 

  int imon;
  
  // for dst i/o
  integer4 inUnit,outUnit,inMode,outMode;
  integer4 wantBanks,gotBanks,outBanks;
  integer4 size,rc,event;
  FILE *fl; // dummy

  // read the command line arguments
  if (argc != 3)
    {
      fprintf (stderr, "\nusage: %s [input dst file] [output dst file]\n",argv[0]);
      fprintf (stderr,"(1): input dst file which contains tasdmonitor dst bank\n" );
      fprintf (stderr,"(2): output dst file which will contain both tasdmonitor and rusdcal dst banks\n");
      fprintf (stderr, "\n\n");
      return 2;
    }
  memcpy(inFile,argv[1],strlen(argv[1])+1);
  memcpy(outFile,argv[2],strlen(argv[2])+1); 



  // open the dst files
  size=100;
  

  // input file
  inUnit = 1;
  inMode = MODE_READ_DST;
  wantBanks=newBankList(size);
  gotBanks=newBankList(size);
  addBankList(wantBanks,TASDMONITOR_BANKID);

  // prefer to check file existence separately
  // (dstOpenUnit return value is sometimes wrong for bz2 files)
  if ((fl=fopen(inFile,"r"))==0)
    {
      fprintf (stderr,"can't open %s\n",inFile);
      return 2;
    }
  else
    {
      fclose(fl);
      dstOpenUnit(inUnit,inFile,inMode); 
    }
  

  
  // output file
  outUnit=2;
  outMode=MODE_WRITE_DST;
  outBanks=newBankList(size);
  addBankList(outBanks,TASDMONITOR_BANKID);
  addBankList(outBanks,RUSDCAL_BANKID);
  if ((fl=fopen(outFile,"w"))==0)
    {
      fprintf (stderr,"can't start %s\n",outFile);
      return 2;
    }
  else
    {
      fclose(fl);
      dstOpenUnit(outUnit,outFile,outMode); 
    }


  // needed for 1mip fitter.
  monsimple = new monsimple_struct;
  hmip[0] = new TH1F("hmip0","mip fit (lower)",RUSDCAL_NMONCHAN,-0.5,511.5);
  hmip[1] = new TH1F("hmip1","mip fit (upper)",RUSDCAL_NMONCHAN,-0.5,511.5);
  ffit = 0;
  
  imon = 0;
  // read input file, add rusdcalib, write into a new dst
  while ((rc=eventRead(inUnit, wantBanks, gotBanks, &event)) > 0)
    {
      // prepare rusdcal variables
      prep_rusdcal(&tasdmonitor_,&rusdcal_);
      fprintf (stdout, "MONCYCLE %03d yymmdd=%06d hhmmss=%06d\n",
	       imon,tasdmonitor_.date,tasdmonitor_.time);
      // eventDump(outBanks);
      imon ++;
      fflush(stdout);
      
      if ((rc=eventWrite(outUnit, outBanks, TRUE)) < 0)
	{
	  return 2;
	}
    }
  dstCloseUnit(inUnit);
  dstCloseUnit(outUnit);

  

  

  fprintf (stdout, "\n\nDone\n");
  return 0;
}


void prep_rusdcal(tasdmonitor_dst_common *mon, rusdcal_dst_common *cal)
{

  int isd;
  
  cal->nsds=mon->num_det;  
  
  for (isd=0; isd < cal->nsds; isd++)
    {     
      cal->xxyy[isd] = mon->lid[isd];
      
      // Load 1MIP histogram
      
      // lower
      memcpy(&monsimple->hmip[0][0],&tasdmonitor_.sub[isd].mon.mip2[0],
	     RUSDCAL_NMONCHAN * sizeof(integer4));
      
      // upper
      memcpy(&monsimple->hmip[1][0],&tasdmonitor_.sub[isd].mon.mip1[0],
	     RUSDCAL_NMONCHAN * sizeof(integer4));
      
      // Load pedestal histogram
      
      // lower
      memcpy(&monsimple->hped[0][0],&tasdmonitor_.sub[isd].mon.ped2[0],
	     (RUSDCAL_NMONCHAN/2) * sizeof(integer4));

      // uppper
      memcpy(&monsimple->hped[1][0],&tasdmonitor_.sub[isd].mon.ped1[0],
	     (RUSDCAL_NMONCHAN/2) * sizeof(integer4));

      // Compute peak/half-peak channels and do the 1MIP fitting
      prepMonInfo(isd,monsimple,cal);
      
    }

}


// To calculate the peak values of the histograms and find their half-peak channels.
void prepMonInfo(int isd, monsimple_struct *mon, rusdcal_dst_common *cal)
{
  int j, k, l;

  // First index is for upper/lower
  // 2nd index: 0 for the values, 1 for the channel numbers (in C-notation, just like the firmware)
  int pch_mip[2][2], pch_ped[2][2], lhpch_mip[2][2], lhpch_ped[2][2],
    rhpch_mip[2][2], rhpch_ped[2][2], jl_mip, jl_ped, ju_mip, ju_ped;

  unsigned char dflag[2]; // this flag is 255 if found all half-peak channels

  // Initialize the peak channels and peak values and clean the 1mip histograms
  for (k=0; k<2; k++)
    {
      for (l=0; l<2; l++)
	{
	  pch_mip[k][l] = 0;
	  pch_ped[k][l] = 0;
	}
      hmip[k]->Reset(); // clean the 1MIP histograms used in fitting
    }

  // Find the peak channels for mip & ped.  Avoid using
  // the last channels as they may contain special error flags recorded by DAQ. 
  for (j=0; j<(RUSDCAL_NMONCHAN-1); j++)
    {
      // Loop over upper/lower
      for (k=0; k<2; k++)
	{
	  if (mon->hmip[k][j] >= 0)
	    hmip[k]->SetBinContent(j+1, mon->hmip[k][j]);
	  if (mon->hmip[k][j] > pch_mip[k][0])
	    {
	      pch_mip[k][0] = mon->hmip[k][j]; // the peak value
	      pch_mip[k][1] = j; // the peak channel
	    }
	  // peak channel for ped, keeping in mind that the number of ped chanels is half as large as the number
	  // of channels for mip.
	  if ( (j < (RUSDCAL_NMONCHAN /2 - 1)) && 
	       (mon->hped[k][j] > pch_ped[k][0]))
	    {
	      pch_ped[k][0] = mon->hped[k][j];
	      pch_ped[k][1] = j;
	    }

	}
    }

  // Record the peak channels to DST bank
  for (k=0; k<2; k++)
    {
      cal->pchmip[isd][k] = pch_mip[k][1];
      cal->pchped[isd][k] = pch_ped[k][1];
    }

  // Initialize the half-peak channels, first with the peak channels
  for (k=0; k<2; k++)
    {

      // When mip and ped half-peak channels are found, then these flags will be set to 15.
      dflag[k] = (unsigned char)0;

      for (l=0; l<2; l++)
	{
	  lhpch_mip[k][l] = pch_mip[k][l];
	  rhpch_mip[k][l] = pch_mip[k][l];
	  lhpch_ped[k][l] = pch_ped[k][l];
	  rhpch_ped[k][l] = pch_ped[k][l];
	}
    }

  // Find the half-peak channels for mip and ped.
  for (j=0; j<(RUSDCAL_NMONCHAN-1); j++)
    {

      // No need to continue if found the half-peak channels for mip and ped.
      if (dflag[0] == (unsigned char)15 && dflag[1] == (unsigned char)15)
	break;

      for (k=0; k<2; k++)
	{
	  jl_mip = pch_mip[k][1]-j;
	  ju_mip = pch_mip[k][1]+j;
	  jl_ped = pch_ped[k][1]-j;
	  ju_ped = pch_ped[k][1]+j;

	  // left half-peak for mip
	  if ((lhpch_mip[k][0] > (pch_mip[k][0] / 2)) && (jl_mip >=0 )
	      && (mon->hmip[k][jl_mip] > 0))
	    {
	      lhpch_mip[k][0] = mon->hmip[k][jl_mip];
	      lhpch_mip[k][1] = jl_mip;
	    }
	  else
	    {
	      // set the bit flag that says left-half peak for mip is found
	      dflag[k] |= (unsigned char)1;
	    }

	  // right half-peak for mip
	  if ((rhpch_mip[k][0] > (pch_mip[k][0] / 2)) && 
	      (ju_mip < (RUSDCAL_NMONCHAN-1)) && (mon->hmip[k][ju_mip] > 0))
	    {
	      rhpch_mip[k][0] = mon->hmip[k][ju_mip];
	      rhpch_mip[k][1] = ju_mip;
	    }
	  else
	    {
	      // indicated that the right-half peak for mip is found
	      dflag[k] |= (unsigned char)2;
	    }

	  // left half-peak for ped
	  if ((lhpch_ped[k][0] > (pch_ped[k][0] / 2)) && (jl_ped >=0 )
	      && (mon->hped[k][jl_ped] > 0))
	    {
	      lhpch_ped[k][0] = mon->hped[k][jl_ped];
	      lhpch_ped[k][1] = jl_ped;
	    }
	  else
	    {
	      dflag[k] |= (unsigned char)8;
	    }

	  // right half-peak for ped
	  if ((rhpch_ped[k][0] > (pch_ped[k][0] / 2)) && 
	      (ju_ped < (RUSDCAL_NMONCHAN/2 - 1)) && 
	      (mon->hped[k][jl_ped] > 0))
	    {
	      rhpch_ped[k][0] = mon->hped[k][ju_ped];
	      rhpch_ped[k][1] = ju_ped;
	    }
	  else
	    {
	      dflag[k] |= (unsigned char)16;
	    }
	}
    }

  // Record the half peak channels to DST bank, and fit the 1mip histograms      
  for (k=0; k<2; k++)
    {
      cal->lhpchmip[isd][k] = lhpch_mip[k][1];
      cal->lhpchped[isd][k] = lhpch_ped[k][1];
      cal->rhpchmip[isd][k] = rhpch_mip[k][1];
      cal->rhpchped[isd][k] = rhpch_ped[k][1];
    }

  // Fit 1MIP histograms for upper and lower
  mipFIT(isd, cal);

  // Evaluate best estimate for 1MIP peak, subtract the pedestal
  for (k=0; k<2; k++)
    {
      cal->mip[isd][k]= cal->mftp[isd][k][0]+ 0.5/cal->mftp[isd][k][2]
	* (sqrt(1.0+4.0 *cal->mftp[isd][k][2]*cal->mftp[isd][k][2]
		* cal->mftp[isd][k][1]*cal->mftp[isd][k][1])-1.0)- 1.5
	*(real8)cal->pchped[isd][k];
    }
}


static real8 mipfun(real8 *x, real8 *par)
{
  // Fit parameters:
  // par[0]=Gauss Mean
  // par[1]=Gauss Sigma
  // par[2]=Linear Coefficient
  // par[3]=Scalling Factor (integral)
  return par[3]*(1+par[2]*(x[0]-par[0])) *(Gaus(x[0], par[0], par[1],true));
}


void mipFIT(integer4 isd, rusdcal_dst_common *cal)
{
  // par[0]=Gauss Mean
  // par[1]=Gauss Sigma
  // par[2]=Linear Coefficient
  // par[3]=Scalling Factor (integral)
  real8 fr[2];
  real8 sv[4], pllo[4], plhi[4];
  real8 delta;
  integer4 j, k;

  // go over upper and lower
  for (k=0; k<2; k++)
    {
      // Fit range
      delta = (real8)(cal->pchmip[isd][k]-cal->lhpchmip[isd][k]);
      fr[0] = (real8)cal->pchmip[isd][k]-delta;
      fr[1] = (real8)cal->pchmip[isd][k]+0.7*delta;

      // Starting values
      sv[0] =(real8)cal->pchmip[isd][k]; // estimate for mean of the gaussian
      sv[1] = delta; // estimate for sigma of the gaussian  
      sv[2] = 0.1; // The linear coeficient estimate
      sv[3] = hmip[k]->Integral(1, 598); // the last two channels contain garbage

      // Lower limits
      pllo[0] = (real8)cal->pchmip[isd][k]-2.0*delta;
      pllo[1] = 0.5*sv[1];
      pllo[2] = 2e-2;
      pllo[3] = 0.5*sv[3];

      // Upper Limits
      plhi[0] = (real8)cal->pchmip[isd][k]+2.0*delta;
      plhi[1] = 2.0*sv[1];
      plhi[2] = 10.0;
      plhi[3] = 1.5*sv[3];

      // Initialize the fit function
      ffit = new TF1("mipfun",mipfun,fr[0],fr[1],4);
      ffit->SetParameters(sv);
      for (j=0; j<4; j++)
	ffit->SetParLimits(j, pllo[j], plhi[j]);

      // Fit
      if (hmip[k]->Fit(ffit, "RB0Q") == -1)
	{
	  fprintf(stderr,"mipFIT: xxyy=%04d layer=%d nentries=%d",
		  cal->xxyy[isd],k,(int)hmip[k]->Integral());
	}

      // Get chi2 and ndof
      cal->mftchi2[isd][k] = ffit->GetChisquare();
      cal->mftndof[isd][k] = ffit->GetNDF();

      // Get fit parameters and their errors
      for (j=0; j<4; j++)
	{
	  cal->mftp[isd][k][j] = ffit->GetParameter(j);
	  cal->mftpe[isd][k][j] = ffit->GetParError(j);
	}
      // Discard the fit function
      delete ffit;
    }
}
