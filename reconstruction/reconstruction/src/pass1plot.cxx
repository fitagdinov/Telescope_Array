//
// Main SD event display routines
//


#include <fstream>
#include "TObjArray.h"
#include "TChainElement.h"
#include "pass1plot.h"


using namespace std;


static sdxyzclf_class clfcoor;


ClassImp(pass1plot)


pass1plot::pass1plot(const char *listfile) : 
MD_PMT_QE(0.278)
{
  
  
  fdplane     = 0; // pointer to brplane or lrplane branch
  fdprofile   = 0; // poter to brprofile or fdprofile branch
  
  // thse variables calculated inside of this class
  md_pcgf     = new md_pcgf_class;
  
  // Initialize graphs
  for (Int_t k=0; k<3; k++)
    {
      gTvsUall[k] = 0;
      gTvsUsclust[k] = 0;
      gTvsUclust[k] = 0;
      
      gTvsRall[k] = 0;
      gTvsRsclust[k] = 0;
      gTvsRclust[k] = 0;
      
      gTvsVall[k] = 0;
      gTvsVsclust[k] = 0;
      gTvsVclust[k] = 0;
      
      gQvsSall[k] = 0;
      gQvsSsclust[k] = 0;
      gQvsSclust[k] = 0;
      
      if (k<2)
	hfadc[k] = 0;
    }
  
  
  // initialize the acceptable tree names in the files
  tree_names.clear();
  tree_names.push_back("taTree");
  tree_names.push_back("pass1tree");
  
  
  // exit the event display if can't properly chain the input files contained in
  // the list file
  if(!(pass1tree = chain_input_root_files(listfile)))
    exit(2);

  // Initialize the event time stamp to a default value
  set_event_time_stamp(true);
  
  // clear out the vector that holds all existent dst class branches
  dst_branches.clear();

 
 
  
  // IMPLEMENT THE DST BRANCH VARIABLES
#ifndef _PASS1PLOT_IMPLEMENTATION_
#define _PASS1PLOT_IMPLEMENTATION_
#endif
#include "pass1plot_dst_branch_handler.h"
 
  
  eventsRead = pass1tree->GetEntries();
    
  init_hist();
  fprintf(stdout,"Found %d events\n", eventsRead);
    
}

/* Nothing needed for the destructor */
pass1plot::~pass1plot()
{
}


TChain* pass1plot::chain_input_root_files(const char* listfile)
{
  TChain* chain = 0;       // initialize
  ifstream fin(listfile);  // open the input list file
  if(!fin)
    {
      SDIO::printMessage(stderr,"pass1plot:chain_input_root_files", "ERROR: can't open list file '%s' for reading", 
			 listfile);
      return 0;
    }
  TString fname = "";
  while (fname.ReadLine(fin))
    {
      TFile* f = new TFile(fname);
      if (f->IsZombie())
        continue; // ignore files that can't be opened for reading
      // if chain chain hasn't been allocated yet
      if (!chain)
        {
          // search for recognized root tree names in the file
          for (vector<TString>::iterator it = tree_names.begin(); it != tree_names.end(); it++)
            {
              const TString& tree_name = (*it);
              if(!f->Get(tree_name))
                continue;
              if(f->Get(tree_name)->InheritsFrom("TTree"))
                {
                  // successfully identified the name of the root trees in the file
                  chain = new TChain(tree_name);
                  break;
                }
            }
          // if successfully identified ROOT tree for the first time, then close
          // the file, add it to the chain, and move on to other files in the list
          if(chain)
            {
              f->Close();
              if(!chain->Add(fname))
                {
                  SDIO::printMessage(stderr,0,"ERROR: file %s contains TTree named '%s' but it could not be added to the chain",
                                      fname.Data(), chain->GetName());
                  return 0;
                }
              continue;
            }
          // otherwise display a message saying that the file does not work
          else
            {
              SDIO::printMessage(stderr,"pass1plot:chain_input_root_files",
                                  "WARNING: no root tree with acceptable object name found in '%s'\n",fname.Data());
              SDIO::printMessage(stderr,0,"acceptable root tree names are:");
              for (vector<TString>::iterator it = tree_names.begin(); it != tree_names.end(); it++)
                SDIO::printMessage(stderr,0,"'%s'",it->Data());
              continue;
            }
        }
      // this isn't the 1st file in the chain so make sure that the root tree has the same name 
      // as the one in the 1st successfully chained file
      if(f->Get(chain->GetName()))
        {
          if(f->Get(chain->GetName())->InheritsFrom("TTree"))
            {
              f->Close();
              if(!chain->Add(fname))
                {
                  SDIO::printMessage(stderr,0,"ERROR: file %s contains TTree named '%s' but it could not be added to the chain",
                                      fname.Data(), chain->GetName());
                  return 0;
                }
              continue;
            }
        }
      // if could't add the file to the chain because it doesn't contain the root tree that is named correctly
      SDIO::printMessage(stderr,0,"error: '%s' doesn't contain a root tree named '%s' ",fname.Data(),chain->GetName());
      if(chain->GetListOfFiles())
        {
          TIter next(chain->GetListOfFiles());
          TChainElement *chEl = (TChainElement*)next();
          if(chEl)
            SDIO::printMessage(stderr,0,"which first appeared in '%s'",chEl->GetTitle());
        }
      SDIO::printMessage(stderr,0,"Notice: ROOT trees with the same object name must exist in every input file given");
    }
  
  // close the input list file
  fin.close();
  
  // return the successfully made chain
  return chain;
}



Int_t pass1plot::GetEntry(Long64_t entry, Int_t getall)
{
  if(!pass1tree)
    {
      fprintf(stderr,"Error: pass1tree hasn't been initialized!\n");
      return 0;
    }
  Int_t ret_flag=pass1tree->GetEntry(entry,getall);
  set_event_time_stamp();
  return ret_flag;
}


/* Initialize histograms: assign memory blocks to pointers, then
   make sure these histograms are cleaned up */
void pass1plot::init_hist()
{
    
  hNpart[0] = new TH1F("hNpart0","# of particles vs time (Lower)",128,-0.5,127.5);
  hNpart[1] = new TH1F("hNpart1","# of particles vs time (Upper)",128,-0.5,127.5);
  hNfadc[0] = new TH1F("hNfadc0","fadc-pedestals (Lower)",128,-0.5,127.5);
  hNfadc[1] = new TH1F("hNfadc1","fadc-pedestals (Upper)",128,-0.5,127.5);
  hResp[0] = new TH1F("hResp0","1 #mu response (Lower)",128,-0.5,127.5);
  hResp[1] = new TH1F("hResp1","1 #mu response (Upper)",128,-0.5,127.5);
  clean_event_hist();

  // 1 mu response shape, obtained from looking at many 1mu fadc traces.
  muresp_shape[0] = 0.09044301;
  muresp_shape[1] = 0.2011518;
  muresp_shape[2] = 0.2357062;
  muresp_shape[3] = 0.1958141;
  muresp_shape[4] = 0.1346454;
  muresp_shape[5] = 0.07871757;
  muresp_shape[6] = 0.03778135;
  muresp_shape[7] = 0.01320628;
  muresp_shape[8] = 0.003927172;
  muresp_shape[9] = 0.001456995;
  muresp_shape[10] = 0.0007651797;
  muresp_shape[11] = 0.0004421589;

  sNpart = new TSpectrum(); // # for de-convolving fadc traces  
  hTdiff1 = new TH1F ("hTdiff1",
		      "Time difference of hits in the same counter",
		      240,-12.0,12.0);  
  hQVsdT = new TH2F ("hQVsdT",
		     "Q_{excluded} vs time relative to chosen hit",
		     240,-12.0,12.0,200,0.0,20.0);
  hdQVsdT = new TH2F ("hdQVsdT",
		      "Q_{excluded} - Q_{chosen} vs time relative to chosen hit",
		      240,-12.0,12.0,400,-20.0,20.0);
  hQrVsdT = new TH2F ("hQrVsdT",
		      "(Q_{excluded} / Q_{chosen} - 1) vs time relative to chosen hit",
		      240,-12.0,12.0,400,-1.0,20.0);
  hTdiff2VsR = new TH2F ("hTdiff2VsR",
			 "Time difference of hits for counters separated by 1200m vs dist from core",
			 50,0.0,5.0,240,-12.0,12.0);
  hTdiff3VsR = new TH2F ("hTdiff3VsR",
			 "Time difference of hits for counters separated by sqrt(2) * 1200m vs dist from core",
			 50,0.0,5.0,240,-12.0,12.0);

}

void pass1plot::init_varbnum_hist()
{
  Int_t k;
  Char_t hName[20], hTitle[124];
  for (k = 0; k < 3; k++)
    {

      // Pulse Height Histograms
      if (hCharge[k])
	hCharge[k]->Delete();
      sprintf(hName, "hCharge%d", k);
      if (k == 0)
	sprintf(hTitle, "CHARGE (LOWER)");
      if (k == 1)
	sprintf(hTitle, "CHARGE (UPPER)");
      if (k == 2)
	sprintf(hTitle, "CHARGE (BOTH)");
      hCharge[k] = new TH1F (hName, hTitle, rufptn->nhits,
			     -0.5, ((Double_t) rufptn->nhits - 0.5));
      hCharge[k]->GetYaxis ()->SetNdivisions(4);
      hCharge[k]->GetYaxis ()->SetLabelSize(0.1);
      hCharge[k]->GetXaxis ()->SetNdivisions(10);
      hCharge[k]->GetXaxis ()->SetLabelSize(0.1);

      // relative time histograms
      if (hErelTime[k])
	hErelTime[k]->Delete();
      sprintf(hName, "hErelTime%d", k);
      if (k == 0)
	sprintf(hTitle, "Relative times (LOWER)");
      if (k == 1)
	sprintf(hTitle, "Relative times (UPPER)");
      if (k == 2)
	sprintf(hTitle, "Relative times (BOTH)");
      hErelTime[k] = new TH1F (hName, hTitle, rufptn->nhits, -0.5,
			       (Double_t) rufptn->nhits - 0.5);
      hErelTime[k]->SetFillColor(20);

      if (k < 2)
	{
	  // VEM histograms
	  if (hVEM[k])
	    hVEM[k]->Delete();
	  sprintf(hName, "hVEM%d", k);
	  if (k == 0)
	    sprintf(hTitle, "VEM (LOWER)");
	  if (k == 1)
	    sprintf(hTitle, "VEM (UPPER)");

	  hVEM[k] = new TH1F (hName, hTitle, rufptn->nhits, -0.5,
			      (Double_t)rufptn->nhits - 0.5);

	  // Monitoring Ped histograms
	  if (hPed[k])
	    hPed[k]->Delete();
	  sprintf(hName, "hPed%d", k);
	  if (k == 0)
	    sprintf(hTitle, "PED (LOWER)");
	  if (k == 1)
	    sprintf(hTitle, "PED (UPPER)");
	  hPed[k] = new TH1F (hName, hTitle, rufptn->nhits, -0.5,
			      (Double_t)rufptn->nhits - 0.5);

	  // FADC Ped histograms
	  if (hFadcPed[k])
	    hFadcPed[k]->Delete();
	  sprintf(hName, "hFadcPed%d", k);
	  if (k == 0)
	    sprintf(hTitle, "PED (LOWER)");
	  if (k == 1)
	    sprintf(hTitle, "PED (UPPER)");
	  hFadcPed[k] = new TH1F (hName, hTitle, rufptn->nhits, -0.5,
				  (Double_t)rufptn->nhits - 0.5);
	}

    }

  // some cosmetics
  hCharge[2]->SetFillColor(12);
  hCharge[1]->SetFillColor(14);
  hCharge[0]->SetFillColor(16);
}

/* This method resets the individual event histograms */
void pass1plot::clean_event_hist()
{
  Int_t k;
    
  for (k=0; k<3; k++)
    {

      tEarliest[k] = 0.0;
      lcharge[k] = 0.0;
      qtot[k] = 0.0;

      // time vs long axis dist
      // all
      if (gTvsUall[k] != 0)
	{
	  gTvsUall[k] -> Delete();
	  gTvsUall[k] = 0;
	}
      // space cluster
      if (gTvsUsclust[k] != 0)
	{
	  gTvsUsclust[k] -> Delete();
	  gTvsUsclust[k] = 0;
	}
      // cluster
      if (gTvsUclust[k] != 0)
	{
	  gTvsUclust[k] -> Delete();
	  gTvsUclust[k] = 0;
	}

      // time vs dist from core all
      if (gTvsUall[k] != 0)
	{
	  gTvsRall[k] -> Delete();
	  gTvsRall[k] = 0;
	}
      // space cluster
      if (gTvsUsclust[k] != 0)
	{
	  gTvsRsclust[k] -> Delete();
	  gTvsRsclust[k] = 0;
	}
      // cluster
      if (gTvsUclust[k] != 0)
	{
	  gTvsRclust[k] -> Delete();
	  gTvsRclust[k] = 0;
	}

      // time vs short axis dist
      // all
      if (gTvsVall[k] != 0)
	{
	  gTvsVall[k] -> Delete();
	  gTvsVall[k] = 0;
	}
      // cluster
      if (gTvsVclust[k] != 0)
	{
	  gTvsVclust[k] -> Delete();
	  gTvsVclust[k] = 0;
	}

      // charge vs dist from core all
      if (gQvsSall[k] != 0)
	{
	  gQvsSall[k] -> Delete();
	  gQvsSall[k] = 0;
	}
      //space cluster
      if (gQvsSsclust[k] != 0)
	{
	  gQvsSsclust[k] -> Delete();
	  gQvsSsclust[k] = 0;
	}
      //cluster
      if (gQvsSclust[k] != 0)
	{
	  gQvsSclust[k] -> Delete();
	  gQvsSclust[k] = 0;
	}

    }

}

// Histograms FADC for individual events
void pass1plot::event_hist()
{
  Int_t i, k, n;
  Int_t xy[2];
  Double_t x [RUFPTNMH], dx [RUFPTNMH], y [RUFPTNMH], dy [RUFPTNMH];

  clean_event_hist(); // clean fixed binsize histograms
  init_varbnum_hist(); // re-intialize variable bin size histograms

  for (k=0; k < 3; k++)
    {
      if (k<2)
	tEarliest[k] = rufptn->tearliest[k];
      if (k==2)
	tEarliest[k] = (rufptn->tearliest[0]+rufptn->tearliest[1])/2.0;
    }

    

  // Now, go over each hit.
  for (i = 0; i < rufptn->nhits; i++)
    {
	
      xycoor(rufptn->xxyy[i], xy);

      for (k=0; k<3; k++)
	{
	  if (k<2)
	    {
	      charge[i][k] = rufptn->pulsa[i][k];
	      relTime[i][k] = rufptn->reltime[i][k];
	      chargeErr[i][k] = rufptn->pulsaerr[i][k];
	      timeErr[i][k] = rufptn->timeerr[i][k];
	    }
	  else
	    {
	      charge[i][k] = (rufptn->pulsa[i][0]+rufptn->pulsa[i][1])/2.0;
	      chargeErr[i][k] = sqrt(rufptn->pulsaerr[i][0]
				     *rufptn->pulsaerr[i][0]+ rufptn->pulsaerr[i][1]
				     *rufptn->pulsaerr[i][1])/2.0;
	      relTime[i][k] = (rufptn->reltime[i][0]+rufptn->reltime[i][1])
		/ 2.0;
	      timeErr[i][k] = sqrt(rufptn->timeerr[i][0]
				   *rufptn->timeerr[i][0]+ rufptn->timeerr[i][1]
				   *rufptn->timeerr[i][1])/2.0;
	    }

	  if (charge[i][k] > lcharge[k])
	    lcharge[k] = charge[i][k];
	    
	  // Fill the charge histograms
	  hCharge[k]->SetBinContent(i + 1, charge[i][k]);
	  hErelTime[k]->SetBinContent(i + 1, relTime[i][k]);

	  if (k<2)
	    {
	      // fill the FADC/VEM histograms
	      hVEM[k] -> SetBinContent(i + 1, rufptn->vem[i][k]);
	      hVEM[k] -> SetBinError(i + 1, rufptn->vemerr[i][k]);

	      // fill the pedestal histograms

	      hPed[k]->SetBinContent(i+1, rufptn->ped[i][k]);
	      hPed[k]->SetBinError(i+1, rufptn->pederr[i][k]);

	    }

	}

    }

  for (k=0; k < 3; k++)
    {

      // t vs u graph
      // all counters
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 1)
	    continue;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  xycoor(rufptn->xxyy[i], xy);
	  x[n] = (rufptn->tyro_u[k][0] * ((real8) xy[0]
					  - rufptn->tyro_xymoments[k][0]) + rufptn->tyro_u[k][1]
		  * ((real8) xy[1] - rufptn->tyro_xymoments[k][1]));
	  dx[n] = 0.0;
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsUall[k] = new TGraphErrors(n,x,y,dx,dy);
      //counters in space  cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 2)
	    continue;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  xycoor(rufptn->xxyy[i], xy);
	  x[n] = (rufptn->tyro_u[k][0] * ((real8) xy[0]
					  - rufptn->tyro_xymoments[k][0]) + rufptn->tyro_u[k][1]
		  * ((real8) xy[1] - rufptn->tyro_xymoments[k][1]));
	  dx[n] = 0.0;
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsUsclust[k] = new TGraphErrors(n,x,y,dx,dy);
      //counters in S-T cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 4)
	    continue;
	  y[n] = relTime[i][k];
	  dy[n] = 0.31; // set error on time equals 0.31 counter separation unit
	  xycoor(rufptn->xxyy[i], xy);
	  x[n] = (rufptn->tyro_u[k][0] * ((real8) xy[0]
					  - rufptn->tyro_xymoments[k][0]) + rufptn->tyro_u[k][1]
		  * ((real8) xy[1] - rufptn->tyro_xymoments[k][1]));
	  dx[n] = 0.0;
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsUclust[k] = new TGraphErrors(n,x,y,dx,dy);
      gTvsUclust[k]->Fit("pol1", "F,0,Q");
      gTvsUclust[k]->GetFunction("pol1")->ResetBit((1<<9));
      gTvsUclust[k]->GetFunction("pol1")->SetLineWidth(3);
      gTvsUclust[k]->GetFunction("pol1")->SetLineColor(2);

      // t vs R graph
      // all counters
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 1)
	    continue;
	  x[n] = rufptn->tyro_cdist[k][i];
	  dx[n] = 0.0;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsRall[k] = new TGraphErrors(n,x,y,dx,dy);

      //counters in space cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 2)
	    continue;
	  x[n] = rufptn->tyro_cdist[k][i];
	  dx[n] = 0.0;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsRsclust[k] = new TGraphErrors(n,x,y,dx,dy);
      //counters in S-T cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 4)
	    continue;
	  x[n] = rufptn->tyro_cdist[k][i];
	  dx[n] = 0.0;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsRclust[k] = new TGraphErrors(n,x,y,dx,dy);

      // t vs v graph
      // all counters
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 1)
	    continue;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  xycoor(rufptn->xxyy[i], xy);
	  x[n] = (rufptn->tyro_v[k][0] * ((real8) xy[0]
					  - rufptn->tyro_xymoments[k][0]) + rufptn->tyro_v[k][1]
		  * ((real8) xy[1] - rufptn->tyro_xymoments[k][1]));
	  dx[n] = 0.0;
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsVall[k] = new TGraphErrors(n,x,y,dx,dy);
      //counters in space cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 2)
	    continue;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  xycoor(rusdraw->xxyy[i], xy);
	  x[n] = (rufptn->tyro_v[k][0] * ((real8) xy[0]
					  - rufptn->tyro_xymoments[k][0]) + rufptn->tyro_v[k][1]
		  * ((real8) xy[1] - rufptn->tyro_xymoments[k][1]));
	  dx[n] = 0.0;
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsVsclust[k] = new TGraphErrors(n,x,y,dx,dy);
      //counters in S-T cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 4)
	    continue;
	  y[n] = relTime[i][k];
	  dy[n] = timeErr[i][k];
	  xycoor(rusdraw->xxyy[i], xy);
	  x[n] = (rufptn->tyro_v[k][0] * ((real8) xy[0]
					  - rufptn->tyro_xymoments[k][0]) + rufptn->tyro_v[k][1]
		  * ((real8) xy[1] - rufptn->tyro_xymoments[k][1]));
	  dx[n] = 0.0;
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gTvsVclust[k] = new TGraphErrors(n,x,y,dx,dy);

      // charge vs dist from core graph
      // all counters
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 1)
	    continue;
	  // x[n] = rufptn->tyro_cdist[k][i];
	  x[n] = svalue(i);
	  dx[n] = 0.0;
	  y[n] = charge[i][k];
	  dy[n] = chargeErr[i][k];
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gQvsSall[k] = new TGraphErrors(n,x,y,dx,dy);
      //space cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 2)
	    continue;
	  //x[n] = rufptn->tyro_cdist[k][i];
	  x[n] = svalue(i);
	  dx[n] = 0.0;
	  y[n] = charge[i][k];
	  dy[n] = chargeErr[i][k];
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gQvsSsclust[k] = new TGraphErrors(n,x,y,dx,dy);
      //S-T cluster
      n = 0;
      for (i = 0; i < rufptn->nhits; i++)
	{
	  if (rufptn->isgood[i] < 4)
	    continue;
	  // x[n] = rufptn->tyro_cdist[k][i];
	  x[n] = svalue(i);
	  dx[n] = 0.0;
	  y[n] = charge[i][k];
	  dy[n] = chargeErr[i][k];
	  n++;
	}
      if (n < 1)
	n=1; // safety
      gQvsSclust[k] = new TGraphErrors(n,x,y,dx,dy);

    } // for(k=0; k<3; ...


} // end of event_hist


bool pass1plot::fadc_hist(Int_t ihit)
{

  Int_t i, k, j;
  Int_t xy[2];
  char hTitle[125];
  Int_t nbins;
  // array of left-edge values for variable-size bin histograms
  Float_t *leftedgeX = 0;

  Int_t iwf; // 1st waveform in the multi-fold hit

  if ( (ihit > (rufptn->nhits-1)) || ihit < 0)
    {
      fprintf(stdout, "ihit must be in %d - %d range\n",
	      0,(rufptn->nhits-1));
      return false;
    }

  iwf = rufptn -> wfindex [ihit]; // 1st wf index in the hit
  xycoor(rufptn -> xxyy[ihit], xy); // need counter position just for the histogram title

  // calculate the array of gaps in units of 20ns (corresponds to
  // fadc channesl
  ngaps_plot = rufptn -> nfold[ihit]-1;
  nbins = 128; // are always there from the 1st fadc trace
  if (ngaps_plot > 0)
    {
      gapsize_plot.resize(ngaps_plot);
      for (i=0; i<ngaps_plot; i++)
	{
	  // gap size b/w the end of the last waveform and next waveform in the multi-fold hit
	  // here we measure the gap size in units of 20ns = 1 FADC bin.
	  gapsize_plot [i] = ( (Double_t)rusdraw->clkcnt[iwf+i+1]
			       / (Double_t)rusdraw->mclkcnt[iwf+i+1]
			       - (Double_t)rusdraw->clkcnt[iwf+i]
			       / (Double_t)rusdraw->mclkcnt[iwf+i] ) * 50e6 - 128.0;
	  fprintf(stdout,"gapsize (%d) = %e x 20 nS = %f uS\n",
		  i,gapsize_plot[i],0.02*gapsize_plot[i]);
	  if (gapsize_plot[i] < 0)
	    {
	      gapsize_plot[i]=1e-4;
	      fprintf(stdout, "gapsize %d set to %e x 20 nS = %f uS\n",
		      i,gapsize_plot[i],0.02*gapsize_plot[i]);
	    }
	  nbins += 129; // 1 gap bin plus 128 fadc bins from next fadc trace
	}
    }

  // array of left edges for fadc histograms.
  leftedgeX = new Float_t[nbins+1];
  j = 0;
  leftedgeX[0] = -0.5; // left-most bin edge in the fadc trace
  for (i=1; i < (nbins+1); i++)
    {
      // have left edge of the bin next to the gap-bin
      if (i>128 && ((i-129) % 129 == 0) && j < ngaps_plot)
	{
	  leftedgeX[i] = leftedgeX[i-1]+gapsize_plot[j];
	  j++;
	}
      else
	{
	  leftedgeX[i] = leftedgeX[i-1]+1.0;
	}
    }

  // clean up the old histograms
  for (k=0; k<2; k++)
    if (hfadc[k])
      hfadc[k]->Delete();
  hfadc[0] = new TH1F ("hfadc0", " ", nbins, leftedgeX); // lower
  hfadc[1] = new TH1F ("hfadc1", " ", nbins, leftedgeX); // upper

  // Adjust the histogram titles to display the muon numbers
  // Histogram for upper and lower counters only, don't do for both
  for (k=0; k<2; k++)
    {
      if (k==0)
	sprintf(hTitle, "FADC_{L}(%02d%02d), N_{#mu}=%.1f#pm%.1f", xy[0],
		xy[1], rufptn->pulsa[ihit][k], rufptn->pulsaerr[ihit][k]);
      if (k==1)
	sprintf(hTitle, "FADC_{U}(%02d%02d), N_{#mu}=%.1f#pm%.1f", xy[0],
		xy[1], rufptn->pulsa[ihit][k], rufptn->pulsaerr[ihit][k]);
      hfadc[k] -> SetTitle(hTitle);
    }

  // Fill the fadc histograms

  j = 0;
  for (i=0; i < nbins; i++)
    {
      // we are on the gap-bin, don't fill it
      // just increase the wf counter
      if (i>127 && ((i-128) % 129 == 0) && j < ngaps_plot)
	{
	  j++;
	  for (k=0; k<0; k++)
	    hfadc[k] -> SetBinContent(i+1, 0.0);
	}
      else
	{
	  for (k=0; k<2; k++)
	    hfadc[k]->SetBinContent(i+1, rusdraw->fadc[iwf+j][k][i-j*129]);
	}
    }

  nfadcb = nbins; // to return the number of combined fadc bins


  delete leftedgeX;
  return true;
}

bool pass1plot::npart_hist(Int_t ihit)
{
  if ( (ihit > (rufptn->nhits-1)) || ihit < 0)
    {
      fprintf(stdout, "ihit must be in %d - %d range\n",
	      0,(rufptn->nhits-1));
      return false;
    }

  if (rufptn->isgood[ihit]<1)
    {
      fprintf(stdout,"Counter %04d was not working\n",rufptn->xxyy[ihit]);
      return false;
    }

  // People who wrote TSpectrum class did not ensure backward
  // compatiblity: in the older versions of ROOT the Deconvolution method
  // of the class used Float_t type but for the newer versions
  // it uses only Double_t type.
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,0,0)
#   define TSPECTRUM_DECONVOLUTION_TYPE Double_t
#else
#   define TSPECTRUM_DECONVOLUTION_TYPE Float_t
#endif
  TSPECTRUM_DECONVOLUTION_TYPE* source = new TSPECTRUM_DECONVOLUTION_TYPE[128];
  TSPECTRUM_DECONVOLUTION_TYPE* response = new TSPECTRUM_DECONVOLUTION_TYPE[128];

  Int_t iwf = rufptn->wfindex[ihit]; // Get the index of the first waveform that went into the hit


  for (Int_t k=0; k<2; k++)
    {
      hNpart[k] -> Reset();
      hNfadc[k] -> Reset();
      hResp[k]->Reset();

      for (Int_t ic=0; ic<128; ic++)
	{
	  source[ic] = (TSPECTRUM_DECONVOLUTION_TYPE)(rusdraw->fadc[iwf][k][ic]
				 - rufptn->ped[ihit][k]);
	  if (source[ic] < 5*rufptn->pederr[ihit][k])
	    source[ic] = 0.0;
	  hNfadc[k]->SetBinContent(ic+1, source[ic]);
	  if (ic<12)
	    response[ic] = (TSPECTRUM_DECONVOLUTION_TYPE)(muresp_shape[ic]);
	  else
	    response[ic] = 0.0;
	  hResp[k]->SetBinContent(ic+1, response[ic]);
	}
      Int_t nbins = 128;
      sNpart->Deconvolution(source, response, nbins, 100000, 1, 1.0);
      for (Int_t ic=0; ic<126; ic++)
	hNpart[k]->SetBinContent(ic+1, source[ic+2]);
    }

  delete[] source;
  delete[] response;
  return true;
}

/* To focus on a given event in a data sample */
bool pass1plot::lookat(Int_t eventNumber)
{
  Int_t i, k;
  Int_t eNumber;
  eNumber = eventNumber;
  if (eventsRead < 1)
    {
      fprintf(stderr, "there are no events in the tree\n");
      return false;
    }
  if ((eNumber < (-1)) || (eNumber > (eventsRead - 1)))
    {
      fprintf(stderr,"Event number must be in 0-%d range, since there are %d events\n",
	      eventsRead - 1, eventsRead);
      fprintf(stderr,"If event nunmber set to -1, then will look at the current event again\n");
      return false;
    }
  if (eNumber==-1)
    eNumber=pass1tree->GetReadEvent();
  GetEntry(eNumber);
    
  if (!(have_rusdraw) || (rusdraw->yymmdd == 0))
    {
      fprintf(stderr,"rusdraw branch is either absent or empty for this event\n");
      return false;
    }
  fprintf(stdout,"\n\n\n");
  fprintf(stdout,"*************************************************\n");
  fprintf(stdout,"*                                                \n");
  fprintf(stdout,"*                                                \n");
  fprintf(stdout,"* SET ON EVENT:%20.04d\n", eNumber);
  fprintf(stdout,"* DST FILE EVENT NUMBER:%11.04d\n", rusdraw->event_num);
  fprintf(stdout,"* TRIGGER ID(s):%17s=%06d%5s=%06d%5s=%06d\n", "BR",
	 rusdraw->trig_id[0], "LR", rusdraw->trig_id[1], "SK",
	 rusdraw->trig_id[2]);
  fprintf(stdout,"* TOWER(s):");
  switch (rusdraw->site)
    {
    case 0:
      fprintf(stdout,"%23s", "BR\n");
      break;
    case 1:
      fprintf(stdout,"%23s", "LR\n");
      break;
    case 2:
      fprintf(stdout,"%23s", "SK\n");
      break;
    case 3:
      fprintf(stdout,"%23s", "BRLR\n");
      break;
    case 4:
      fprintf(stdout,"%23s", "BRSK\n");
      break;
    case 5:
      fprintf(stdout,"%23s", "LRSK\n");
      break;
    case 6:
      fprintf(stdout,"%23s", "BRLRSK\n");
      break;
    default:
      fprintf(stdout,"%23s", "??\n");
      break;
    }
  fprintf(stdout,"* RUN ID(s): %20s=%06d%5s=%06d%5s=%06d\n", "BR",
	 rusdraw->run_id[0], "LR", rusdraw->run_id[1], "SK", rusdraw->run_id[2]);
  fprintf(stdout,"* MON. CYCLE: %23s=%04d %8s=%04d\n", "YYMMDD", rusdraw->monyymmdd,
	 "HHMMSS", rusdraw->monhhmmss);
  fprintf(stdout,"*                                                \n");
  fprintf(stdout,"*                                                \n");
  fprintf(stdout,"*************************************************\n");
  fprintf(stdout,"year=%d month=%d day=%d hour=%d minute=%d second=%d\n",
	 rusdraw->yymmdd / 10000, (rusdraw->yymmdd % 10000 - rusdraw->yymmdd
				   % 100) / 100, rusdraw->yymmdd % 100, rusdraw->hhmmss / 10000,
	 (rusdraw->hhmmss % 10000 - rusdraw->hhmmss % 100) / 100,
	 rusdraw->hhmmss % 100);
  fprintf(stdout,"%d waveforms\n", rusdraw->nofwf);
  if (rusdraw->nofwf < 1)
    {
      fprintf(stderr,"( MC ?) EVENT DID NOT TRIGGER\n");
      return false;
    }
  // if tasdcalibev branch is present, fill the tasdcalibev index information
  // array
  if (have_tasdcalibev)
    fill_tasdcalibev_wf_indices();
    
  if (!have_rufptn)
    {
      fprintf(stderr,"rufptn branch required for plotting SD events\n");
      return false;
    }
    
  if(rufptn->nstclust < 1)
    {
      fprintf(stderr, "No SDs in space-time cluster found\n");
      return false;
    }
  fprintf(stdout,"Space cluster (%d counters):\n", rufptn->nsclust);
  for (i = 0; i < rufptn->nhits; i++)
    {
      if (rufptn->isgood[i] < 2)
	continue;
      fprintf(stdout,"%04d ", rufptn->xxyy[i]);
    }
  fprintf(stdout,"\n");
  fprintf(stdout,"Space-time cluster: (%d counters):\n", rufptn->nstclust);
  for (i = 0; i < rufptn->nhits; i++)
    {
      if (rufptn->isgood[i] < 4)
	continue;
      fprintf(stdout,"%04d ", rufptn->xxyy[i]);
    }
  fprintf(stdout,"\n");

  // Fill in FADC trace histograms for all counters that were hit
  event_hist();

  // Find the sum of the pulse heights in mu-equivalents

  for (k=0; k<2; k++)
    qtot[k] = rufptn->qtot[k];
  qtot[2] = (qtot[0]+qtot[1])/2.0;
  fprintf(stdout,"CHARGE SUMS IN SPACE-TIME CLUSTER:\n");
  fprintf(stdout,"LOWER: %.2f\n", qtot[0]);
  fprintf(stdout,"UPPER: %.2f\n", qtot[1]);
  fprintf(stdout,"BOTH:  %.2f\n", qtot[2]);

  return true;
}

/* To focus on a next event in a data sample with
   largest number of counters in a ST-cluster greater or
   equal to nmax. eNumber is from where to start
   searching for such event */
bool pass1plot::findCluster(Int_t eNumber, Int_t nmin)
{
  if ((eNumber + 1) >= eventsRead)
    {
      fprintf(stdout,"Already at the end of the event list\n");
      return false;
    }
  Int_t i = (eNumber + 1);
  GetEntry(i);
  while (rufptn->nstclust < nmin && i < eventsRead)
    {
      i++;
      GetEntry(i);
    }
  if (rufptn->nstclust >= nmin)
    return lookat(i);

  fprintf(stdout,"Can't find events with nmin>= %d in %d-%d event id range\n", nmin,
	 eNumber, (eventsRead-1));
  return false;
}

bool pass1plot::findCluster(Int_t nmin)
{
  Int_t i = pass1tree->GetReadEvent(); // current event within the tree
  return findCluster(i, nmin);
}

bool pass1plot::findTrig(Int_t trig_id, Int_t site)
{
  Int_t eNumber;
  Int_t i = (Int_t)pass1tree->GetReadEvent();
  eNumber=i;
  if (i >= (eventsRead-1))
    {
      fprintf(stdout,"Already at the end of the event list\n");
      return false;
    }
  while (i < eventsRead)
    {
      GetEntry(i);
      if ((rusdraw->trig_id[site]==trig_id) && (site==rusdraw->site))
	return true;
      i++;
    }
  printf(
	 "Can't find events with trig_id = %d, site=%d in %d-%d event id range\n",
	 trig_id, site, eNumber, (eventsRead-1));
  return false;
}

bool pass1plot::findTrig(Int_t eNumber, Int_t trig_id, Int_t site)
{
  if (eNumber < 0 || eNumber >(eventsRead-1))
    {
      fprintf(stderr,"Starting event number must be in %d-%d range\n",0,(eventsRead-1));
      return false;
    }
  GetEntry(eNumber);
  return findTrig(trig_id, site);
}

// find out if the detector appears in space cluster
bool pass1plot::inSclust(Int_t xxyy)
{
  Int_t i;
  for (i=0; i<rufptn->nhits; i++)
    {
      if ((xxyy==rufptn->xxyy[i]) && (rufptn->isgood[i] >= 2))
	return true;
    }
  return false;
}

// loops over all evetns, analyzes the multi-fold hits
void pass1plot::analyze_Mf_hits(Bool_t fsclust)
{
  Int_t i, j, k;
  Int_t na; // > 0 for multi-fold hits.
  Int_t na_l;
  Double_t qrat;

  if (hNfold)
    hNfold ->Delete();
  if (hQrat)
    hQrat->Delete();

  hNfold = new TH1F ("hNfold","Events with multiple waveforms", 10,0.5,10.5);
  hQrat = new TH1F ("hQrat", "Ratio of adjacent pulse areas, earlier/later",50,-5.0,5.0);
  hQrat->GetXaxis()->SetTitle("log_{10}(Q_{i}/Q_{i+1})");

  for (i=0; i<eventsRead; i++)
    {
      GetEntry(i);
      for (j=0; j<rufptn->nhits; j++)
	{
	  // use only the space cluster if the flag is set
	  if (fsclust && (rufptn->isgood[j] < 2))
	    continue;
	  na = 0; // count how many additional wfms there are for the counter
	  for (k=j+1; k<rufptn->nhits; k++)
	    {
	      if (rufptn->xxyy[k]==rufptn->xxyy[j])
		{
		  na++;
		  qrat = (rufptn->pulsa[k-1][0]+rufptn->pulsa[k-1][1])
		    / (rufptn->pulsa[k][0]+rufptn->pulsa[k][1]);
		  hQrat->Fill(log10(qrat));
		}
	    }
	  j+=na; // move on to next counter hit
	}

      na_l = 0;
      for (j=0; j<rusdraw->nofwf; j++)
	{
	  // space cluster cut if the flag is set
	  if (fsclust && (!inSclust(rusdraw->xxyy[j])))
	    continue;
	  na=0;
	  for (k=j+1; k<rusdraw->nofwf; k++)
	    {
	      if (rusdraw->wf_id[k]==0)
		break;
	      na++;
	    }
	  if (na_l < na)
	    na_l = na;
	  j+=na;
	}

      for (j=0; j<=na_l; j++)
	{
	  hNfold->Fill((Double_t)(j+1));
	}

    }

}

double pass1plot::get_sd_secfrac()
{
  double tref=0.5*(rufptn->tearliest[0]+rufptn->tearliest[1]);
  tref -= TMath::Floor(tref);
  return tref;
}

// This routine uses PROTON energy estimation table
Double_t pass1plot::get_sdenergy(Double_t s800, Double_t theta)
{ return rusdenergy(s800,theta); }

// This routine uses IRON energy estimation table
Double_t pass1plot::get_sdenergy_iron(double s800, double theta)
{  return rusdenergy_iron(s800,theta); }



void pass1plot::histSignal()
{

  Int_t i, j, k, l;
  Int_t iwf;
  Int_t lph;
  Int_t nent;
  if (hQscat)
    hQscat->Delete();
  if (pQscat)
    pQscat->Delete();
  if (hQupQloRat)
    hQupQloRat->Delete();

  if (hQupQloRatScat)
    hQupQloRatScat->Delete();
  if (pQupQloRatScat)
    pQupQloRatScat->Delete();

  if (hTscat)
    hTscat->Delete();

  if (pTscat)
    pTscat->Delete();

  if (hLargePheight)
    hLargePheight->Delete();

  if (h1MIPResp)
    h1MIPResp->Delete();

  hQscat = new TH2F("hQscat","Q_{Upper} vs Q_{Lower}",200,0.0,200.0,200,0.0,200.0);
  hQscat ->GetXaxis()->SetTitle("Q_{Lower}, [VEM]");
  hQscat ->GetYaxis()->SetTitle("Q_{Upper}, [VEM]");
  pQscat = new TProfile("pQscat","Q_{Upper} vs Q_{Lower}",200,0.0,200.0,0.0,200.0,"S");
  pQscat ->GetXaxis()->SetTitle("Q_{Lower}, [VEM]");
  pQscat ->GetYaxis()->SetTitle("Q_{Upper}, [VEM]");

  hQupQloRat = new TH1F("hQupQloRat","Q_{Upper} divided by Q_{Lower}",
			400,0.0,4.0);
  hQupQloRat->GetXaxis()->SetTitle("Q_{Upper}/Q_{Lower}");

  hQupQloRatScat = new TH2F("hQupQloRatScat","Q_{Upper}/Q_{Lower} vs Q_{Lower}",
			    200,0.0,200.0,400,0.0,4.0);
  hQupQloRatScat ->GetXaxis()->SetTitle("Q_{Lower}, [VEM]");
  hQupQloRatScat ->GetYaxis()->SetTitle("#frac{Q_{Upper}}{Q_{Lower}}");
  pQupQloRatScat = new TProfile("pQupQloRatScat","Q_{Upper}/Q_{Lower} vs Q_{Lower}",
				100,0.0,200.0,0.0,4.0,"S");
  pQupQloRatScat ->GetXaxis()->SetTitle("Q_{Lower}, [VEM]");
  pQupQloRatScat ->GetYaxis()->SetTitle("Q_{Upper}/Q_{Lower}");

  hTscat = new TH2F("hTscat","T_{Upper} vs T_{Lower}",100,0.0,10.0,100,0.0,10.0);
  hTscat ->GetXaxis()->SetTitle("T_{Lower}, [1200m]");
  hTscat ->GetYaxis()->SetTitle("T_{Upper}, [1200m]");

  pTscat = new TProfile("pTscat","T_{Upper} vs T_{Lower}",10,0.0,10.0,0.0,10.0,"S");
  pTscat ->GetXaxis()->SetTitle("T_{Lower}, [1200m]");
  pTscat ->GetYaxis()->SetTitle("T_{Upper}, [1200m]");

  hLargePheight = new TH1F("hLargePheight","Largest pulse height (FADC counts)",100,0,0);

  h1MIPResp = new TH1F("h1MIPResp","1#mu Response Function",128,-0.5,127.5);

  nent = 0;
  for (i=0; i<eventsRead; i++)
    {
      GetEntry(i);
      for (j=0; j < rufptn->nhits; j++)
	{
	  if (rufptn->isgood[j] < 4)
	    continue;

	  hQscat->Fill(rufptn->pulsa[j][0], rufptn->pulsa[j][1]);
	  pQscat->Fill(rufptn->pulsa[j][0], rufptn->pulsa[j][1]);

	  hQupQloRat->Fill(rufptn->pulsa[j][1] / rufptn->pulsa[j][0]);
	  hQupQloRatScat->Fill(rufptn->pulsa[j][0], rufptn->pulsa[j][1]
			       / rufptn->pulsa[j][0]);
	  pQupQloRatScat->Fill(rufptn->pulsa[j][0], rufptn->pulsa[j][1]
			       / rufptn->pulsa[j][0]);

	  hTscat->Fill(rufptn->reltime[j][0], rufptn->reltime[j][1]);
	  pTscat->Fill(rufptn->reltime[j][0], rufptn->reltime[j][1]);
	}

      lph = 0;
      for (j=0; j<rusdraw->nofwf; j++)
	{
	  for (l=0; l<128; l++)
	    {
	      for (k=0; k<2; k++)
		{
		  if (lph<rusdraw->fadc[j][k][l])
		    lph = rusdraw->fadc[j][k][l];
		}
	    }
	}
      hLargePheight->Fill((Float_t)lph);

      for (j=0; j<rufptn->nhits; j++)
	{
	  if ( (rufptn->isgood[j]==1) && ((rufptn->pulsa[j][0] < 3.0)
					  && (rufptn->pulsa[j][1]) < 3.0))
	    {

	      iwf=rufptn->wfindex[j];
	      for (k=0; k<2; k++)
		{
		  for (l=rufptn->sstart[j][k]; l<=rufptn->sstop[j][k]; l++)
		    {
		      h1MIPResp->AddBinContent( (l-rufptn->sstart[j][k]+1),
						((Double_t)rusdraw->fadc[iwf][k][l]
						 -rufptn->ped[j][k]));
		    }
		  nent++;
		}
	    }
	}

      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)i/(Double_t)(eventsRead-1)*100.0,'%');
      fflush(stdout);

    }

  h1MIPResp->Scale(1.0/(Double_t)nent);

  fprintf(stdout,"\n");

}

void pass1plot::fill_tasdcalibev_wf_indices()
{
  int iwf;
  int j;
  for (iwf=0; iwf < rusdraw->nofwf; iwf++)
    {
      rusdraw2tasdcalibev[iwf] = -1;
      // here we are using the fact that 
      // rusdraw can only have fewer waveforms than
      // tasdcalibev ( corrupted waveforms are not added
      // to rusdraw bank )
      for (j=iwf; j<tasdcalibev->numTrgwf; j++)
	{
	  if ( 
	      (rusdraw->xxyy[iwf] == (int)tasdcalibev->sub[j].lid) &&
	      (rusdraw->wf_id[iwf] == (int)tasdcalibev->sub[j].wfId) && 
	      (rusdraw->clkcnt[iwf] == (int)tasdcalibev->sub[j].clock)
	       )
	    rusdraw2tasdcalibev[iwf] = j;
	}
      if (rusdraw2tasdcalibev[iwf] == -1)
	fprintf(stderr,"Failed to find tasdcalib waveform iwf = %d\n",iwf);
    } 
}

Int_t pass1plot::get_tasdcalibev_iwf(Int_t rusdraw_iwf)
{
  Int_t tasdcalibev_iwf = 0;
  if (!have_tasdcalibev)
    {
      fprintf(stderr,"Must have tasdcalibev branch to obtain tasdcalibev_iwf\n");
      return 0;
    }
  if (rusdraw_iwf < 0 || rusdraw_iwf >= rusdraw->nofwf)
    {
      fprintf(stderr,"rusdraw_iwf must be in 0 to %d range\n",rusdraw->nofwf-1);
      return 0;
    }
  if ((tasdcalibev_iwf=rusdraw2tasdcalibev[rusdraw_iwf]) == -1)
    {
      fprintf(stderr,"Failed to indentify tasdcalibev_iwf for rusdraw_iwf=%d\n",rusdraw_iwf);
      return 0;
    }
  return tasdcalibev_iwf;
}


// Get counter xyz coordinates in CLF frame in [1200m] units
// on any given date
bool pass1plot::get_xyz(int yymmdd, int xxyy, double *xyz)
{ return clfcoor.get_xyz(yymmdd,xxyy,xyz); }

Double_t pass1plot::svalue(Int_t ihit)
{
  Int_t i;
  Int_t xy[2];
  Double_t d[2];
  Double_t dotp;
  xycoor(rufptn->xxyy[ihit], xy);
  for (i=0; i<2; i++)
    d[i] = (Double_t)xy[i] - rufptn->tyro_xymoments[2][i];
  dotp = rufptn->tyro_tfitpars[2][1]*(rufptn->tyro_u[2][0]*d[0]
				      + rufptn->tyro_u[2][1]*d[1]); // Dot product of distance from core and shower axis vector
  return sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
}

Int_t pass1plot::get_event_surroundedness()
{
  if(!have_rusdraw || ! have_rusdgeom || !have_bsdinfo)
    return -1;
  rusdraw->loadToDST();
  rusdgeom->loadToDST();
  bsdinfo->loadToDST();
  return clfcoor.get_event_surroundedness(&rusdraw_,&rusdgeom_,&bsdinfo_);
}


void pass1plot::Help()
{
  fprintf(stdout,"\n\n================QUICK MANUAL==============================\n\n");
  fprintf(stdout,"For now, see comments in the code. Good luck! :)))\n");
  fprintf(stdout,"\n================QUICK MANUAL===============================\n\n");
}
