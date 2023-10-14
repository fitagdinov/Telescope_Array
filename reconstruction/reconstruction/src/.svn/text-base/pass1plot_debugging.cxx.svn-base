#include "pass1plot.h"


using namespace std;

//
// Routines that were mostly used for debugging of the SD analysis
// and that are not useful for the main SD event display
//

// To recalculate the core without the i'th counter
bool pass1plot::recalcCore(Int_t iexclude, Double_t *coreXY)

{
  // Calculate the original core position (coreXY0)
  for (Int_t j = 0; j < 2; j++)
    coreXY[j] = 0.0;
  Double_t Q = 0.0;
  for (Int_t i = 0; i < rufptn->nhits; i++)
    {
      if (i == iexclude)
	continue;
      if (rufptn->isgood[i] < 4)
	continue;
      Double_t xy[2] = 
	{
	  rufptn->xyzclf[i][0] - RUFPTN_ORIGIN_X_CLF,
	  xy[1] = rufptn->xyzclf[i][1] - RUFPTN_ORIGIN_Y_CLF
	};
      Double_t q = (rufptn->pulsa[i][0]+rufptn->pulsa[i][1])/2.0;
      Q += q;
      for (Int_t j = 0; j < 2; j++)
	coreXY[j] += q * xy[j]; 
    }
  if (Q<1e-3)
    return false;
  for (Int_t j = 0; j < 2; j++)
    coreXY[j] /= Q;
  return true;
}

// Compute new core location with given cuts:
// rmin: minimum distance from the Old core
// minOneOverQ: minimum value of 1/charge
// nrem - number of SDs removed by the cuts
bool pass1plot::calcNewCore(Double_t rmin, Double_t minOneOverQ,
			    Double_t *oldCoreXY, Double_t *newCoreXY, Int_t *nremoved)
{
  Int_t i, j, n;
  Double_t xy[2];
  Double_t r, q, Q;

  for (j=0; j<2; j++)
    newCoreXY[j] = 0.0;

  // Calculating new core with given rmin, qmin.


  n = 0; // To count how many SDs are removed by cuts on rmin and (1/Q) min
  Q = 0.0;
  for (i=0; i<rufptn->nhits; i++)
    {
      if (rufptn->isgood[i]<4)
	continue;

      xy[0] = rufptn->xyzclf[i][0] - RUFPTN_ORIGIN_X_CLF;
      xy[1] = rufptn->xyzclf[i][1] - RUFPTN_ORIGIN_Y_CLF;

      r = sqrt( (xy[0]-oldCoreXY[0])*(xy[0]-oldCoreXY[0]) + (xy[1]
							     -oldCoreXY[1])*(xy[1]-oldCoreXY[1]));
      q = (rufptn->pulsa[i][0]+rufptn->pulsa[i][1])/2.0;

      if (1.0/q < minOneOverQ)
	{
	  n++;
	  continue;
	}

      if (r < rmin)
	{
	  n++;
	  continue;
	}
      Q += q;
      for (j=0; j<2; j++)
	newCoreXY[j] += q * xy[j];

    }

  // Can't calculate the core with no total charge
  if (Q<1e-3)
    {
      (*nremoved) = n;
      return false;
    }

  for (j=0; j<2; j++)
    newCoreXY[j] /= Q;

  (*nremoved) = n;

  return true;

}
void pass1plot::histDcore()
{
  Int_t i, j, k, nremoved;
  Double_t rmin;
  Double_t minOneOverQ;
  Double_t oldCoreXY[2];
  Double_t newCoreXY[2];
  Double_t dr2;
  Int_t nbr2, nbq;

  if (hDcoreR2vsRmin)
    hDcoreR2vsRmin->Delete();
  if (pDcoreR2vsRmin)
    pDcoreR2vsRmin->Delete();
  if (hNremVsRmin)
    hNremVsRmin->Delete();
  if (pNremVsRmin)
    pNremVsRmin->Delete();

  if (hDcoreR2vsOneOverQmin)
    hDcoreR2vsOneOverQmin->Delete();
  if (pDcoreR2vsOneOverQmin)
    pDcoreR2vsOneOverQmin->Delete();
  if (hNremVsOneOverQmin)
    hNremVsOneOverQmin->Delete();
  if (pNremVsOneOverQmin)
    pNremVsOneOverQmin->Delete();

  hDcoreR2vsRmin = new TH2F("hDcoreR2vsRmin","#deltaR vs R_{min}",
			    40,0.0,4.0,20,0.0,2.0);
  pDcoreR2vsRmin = new TProfile("pDcoreR2vsRmin","#deltaR vs R_{min}",
				(Int_t)4e3,0.0,4.0,0.0,2.0);

  hNremVsRmin = new TH2F("hNremVsRmin","Counters Removed vs R_{min}",
			 40,0.0,4.0, 21,-0.5,20.5);
  pNremVsRmin = new TProfile("pNremVsRmin","Counters Removed vs R_{min}",
			     (Int_t)4e3,0.0,4.0,-0.5,20.5);

  hDcoreR2vsOneOverQmin = new TH2F("hDcoreR2vsOneOverQmin",
				   "#deltaR vs (#frac{1}{Q})_{min}",
				   20,0.0,1.0,20,0.0,2.0);

  pDcoreR2vsOneOverQmin = new TProfile("pDcoreR2vsOneOverQmin",
				       "#deltaR vs (#frac{1}{Q})_{min}",(Int_t)1e3,0.0,1.0,0.0,2.0);

  hNremVsOneOverQmin = new TH2F("hNremVsOneOverQmin",
				"Number of remaining counters vs (#frac{1}{Q})_{min}",
				20,0.0,1.0,21,-0.5,20.5);

  pNremVsOneOverQmin = new TProfile("pNremVsOneOverQmin",
				    "Number of remaining counters vs (#frac{1}{Q})_{min}",
				    (Int_t)1e3,0.0,1.0,-0.5,20.5);

  nbr2 = pDcoreR2vsRmin->GetNbinsX();
  nbq = pDcoreR2vsOneOverQmin->GetNbinsX();

  fprintf(stdout,"Computing change in core position\n");

  // Loops over events
  for (i=0; i<eventsRead; i++)
    {
      GetEntry(i);

      if (!recalcCore(1e5, oldCoreXY))
	continue;

      // Loop over all possible minimum r
      for (j=0; j<nbr2; j++)
	{
	  rmin=pDcoreR2vsRmin->GetBinCenter(j+1);
	  if (calcNewCore(rmin, 0.0, oldCoreXY, newCoreXY, &nremoved))
	    {
	      dr2=0.0;
	      for (k=0; k<2; k++)
		dr2 += (newCoreXY[k]-oldCoreXY[k]) *(newCoreXY[k]
						     -oldCoreXY[k]);

	      hDcoreR2vsRmin->Fill(rmin, sqrt(dr2));
	      pDcoreR2vsRmin->Fill(rmin, sqrt(dr2));

	    }
	  hNremVsRmin->Fill(rmin, (Double_t)(rufptn->nstclust-nremoved));
	  pNremVsRmin->Fill(rmin, (Double_t)(rufptn->nstclust-nremoved));

	}

      // Loop over all possible minimum 1/q
      for (j=0; j<nbq; j++)
	{
	  minOneOverQ = pDcoreR2vsOneOverQmin->GetBinCenter(j+1);
	  if (calcNewCore(0.0, minOneOverQ, oldCoreXY, newCoreXY, &nremoved))
	    {
	      dr2=0.0;
	      for (k=0; k<2; k++)
		dr2 += (newCoreXY[k]-oldCoreXY[k]) *(newCoreXY[k]
						     -oldCoreXY[k]);

	      hDcoreR2vsOneOverQmin->Fill(minOneOverQ, sqrt(dr2));
	      pDcoreR2vsOneOverQmin->Fill(minOneOverQ, sqrt(dr2));

	    }
	  hNremVsOneOverQmin->Fill(minOneOverQ, (Double_t)(rufptn->nstclust
							   -nremoved));
	  pNremVsOneOverQmin->Fill(minOneOverQ, (Double_t)(rufptn->nstclust
							   -nremoved));
	}

      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)i/(Double_t)(eventsRead-1)*100.0,'%');
      fflush(stdout);

    }
  fprintf(stdout,"\n");

}

bool pass1plot::isSaturated(Int_t ihit)
{
  Int_t j, k;
  Double_t binCont;
  Double_t mipval;

  if ( (rufptn->pulsa[ihit][0] < 80.0) && (rufptn->pulsa[ihit][1] < 80.0))
    return false;

  // If the hit is sufficiently large, then histogram the # of particles vs time
  // for the 1st FADC window for the counter
  npart_hist(ihit);

  for (j=1; j<=128; j++)
    {
      for (k=0; k<2; k++)
	{
	  binCont = hNpart[k]->GetBinContent(j);
	  mipval = rufptn->vem[ihit][k]/TMath::Cos(TMath::DegToRad()*35.0); // Use MIP, not VEM here
	  if (binCont/mipval >= 40.0)
	    return true;
	}
    }

  return false;
}

// To compute the time delay of ihit
static void compTdelay(Double_t Rho, Double_t R, Double_t *td, Double_t *ts)
{
  Double_t a;
  a = 2.6;
  (*td) = a * TMath::Power( (1.0 + R / 30.0), 1.5) * TMath::Power(Rho, -0.5)/4e3;
  (*ts) = a * TMath::Power( (1.0 + R / 30.0), 1.5) * TMath::Power(Rho, -0.3)/4e3;
}

void pass1plot::histSat()
{

  Int_t ievent;
  Int_t ihit;
  Int_t j;

  Double_t coreXY[2];

  Double_t d[2];
  Double_t dr;
  Double_t q, rdist;
  Double_t td, ts, dt;

  if (hDcoreRnoSat)
    hDcoreRnoSat->Delete();

  if (hDcoreRnoSatVsR)
    hDcoreRnoSatVsR->Delete();

  if (pDcoreRnoSatVsR)
    pDcoreRnoSatVsR->Delete();

  hDcoreRnoSat = new TH1F("hDcoreRnoSat",

			  "#deltaR when saturated counters are removed",20,0.0,1.5);
  hDcoreRnoSatVsR = new TH2F("hDcoreRnoSatVsR",
			     "#deltaR when saturated counters are removed",40,0.0,4.0,20,0.0,1.5);
  pDcoreRnoSatVsR = new TProfile("pDcoreRnoSatVsR",
				 "#deltaR when saturated counters are removed",40,0.0,4.0,0.0,1.5,"S");

  hDcoreRnoSat->GetXaxis()->SetTitle("#deltaR, [1200m]");

  hDcoreRnoSatVsR->GetXaxis()->SetTitle("Distance of saturated counter from core, [1200m]");
  hDcoreRnoSatVsR->GetYaxis()->SetTitle("#deltaR, [1200m]");

  pDcoreRnoSatVsR->GetXaxis()->SetTitle("Distance of saturated counter from core, [1200m]");
  pDcoreRnoSatVsR->GetYaxis()->SetTitle("#deltaR, [1200m]");

  if (hTdSat)
    hTdSat->Delete();

  if (hTsSat)
    hTdSat->Delete();

  if (hTrSat)
    hTrSat->Delete();

  hTdSat = new TH1F ("hTdSat","Time dealy of saturated counters",20,0.0,2.0);
  hTsSat = new TH1F ("hTsSat","Time fluctuation of saturated counters",20,0.0,2.0);
  hTrSat = new TH1F ("hTrSat","Time resolution of saturated counters",20,0.0,2.0);

  hTdSat->GetXaxis()->SetTitle("Time delay, [1200m]");
  hTsSat->GetXaxis()->SetTitle("Time fluctuation, [1200m]");
  hTrSat->GetXaxis()->SetTitle("Time resolution, [1200m]");

  for (ievent=0; ievent<eventsRead; ievent++)
    {
      GetEntry(ievent);
      // Do for events with at least 7 counters in the cluster
      if (rufptn->nstclust < 7)
	continue;

      for (ihit=0; ihit<rufptn->nhits; ihit++)

	{
	  // Use hits in space-time cluster only
	  if (rufptn->isgood[ihit]<4)
	    continue;

	  //            if(isSaturated(ihit))
	  if (true)
	    {

	      // Distance from original core of the saturated counter
	      rdist=rufptn->tyro_cdist[2][ihit];

	      // Charge of the saturated counter
	      q = (rufptn->pulsa[ihit][0]+rufptn->pulsa[ihit][1])/2.0;

	      // By how much the core moves when the saturated counter is removed
	      recalcCore(ihit, coreXY);
	      for (j=0; j<2; j++)
		{
		  d[j] = coreXY[j]-rufptn->tyro_xymoments[2][j];
		}
	      dr = sqrt(d[0]*d[0]+d[1]*d[1]);

	      // Fill histograms w/o the saturated counter
	      hDcoreRnoSat->Fill(dr);
	      hDcoreRnoSatVsR->Fill(rdist, dr);
	      pDcoreRnoSatVsR->Fill(rdist, dr);

	      // Fill histograms for the saturated counter


	      compTdelay(q/3.0, rdist*1200.0, &td, &ts);

	      //                if ((rdist > 1.0) && (dr > 0.7))
	      //                  {
	      //                    fprintf(stdout,"EVENT: %d\n",ievent);
	      //                    return;
	      //                  }
	      //
	      dt = 0.5*sqrt(rufptn->timeerr[ihit][0]*rufptn->timeerr[ihit][0]
			    + rufptn->timeerr[ihit][1]*rufptn->timeerr[ihit][1]);

	      hTdSat->Fill(td);
	      hTsSat->Fill(ts);
	      hTrSat->Fill(dt);
	    }

	}

      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)ievent/(Double_t)(eventsRead-1)*100.0,'%');
      fflush(stdout);

    }

  fprintf(stdout,"\n");

}
// loops over all evetns, tries different core calc. techniques
void pass1plot::tryCoreCalc()
{
  Int_t i, j, k;
  Double_t xy[2];
  Double_t cdiffxy[2];
  Double_t cdiffr2;
  Double_t c1[2], c2[2];
  Double_t w1, w2;
  Double_t q1, q2;

  if (hCoreDiffX)
    hCoreDiffX->Delete();

  if (hCoreDiffY)
    hCoreDiffY->Delete();
  if (hCoreDiffR)
    hCoreDiffR->Delete();

  hCoreDiffX = new TH1F("hCoreDiffX","Core X Difference (using vem minus using sqrt(vem))",
			100,-5.0,5.0);
  hCoreDiffY = new TH1F("hCoreDiffY","Core Y Difference (using vem minus using sqrt(vem))",
			100,-5.0,5.0);
  hCoreDiffR = new TH1F("hCoreDiffR","Magnitude of the XY Difference",
			100,0.0,5.0);

  for (i=0; i<eventsRead; i++)
    {
      GetEntry(i);

      w1 = 0.0;
      w2 = 0.0;
      q1 = 0.0;
      q2 = 0.0;
      cdiffr2 = 0.0;
      for (k=0; k<2; k++)
	{
	  c1[k] = 0.0;
	  c2[k] = 0.0;
	  cdiffxy[k] = 0.0;
	}

      for (j=0; j<rufptn->nhits; j++)
	{
	  if (rufptn_.isgood[j] < 4)
	    continue;
	  xy[0] = rufptn->xyzclf[j][0] - RUFPTN_ORIGIN_X_CLF;
	  xy[1] = rufptn->xyzclf[j][1] - RUFPTN_ORIGIN_Y_CLF;
	  q1 = (rufptn->pulsa[j][0]+rufptn->pulsa[j][1])/2.0;
	  q2 = sqrt(q1);

	  for (k=0; k<2; k++)
	    {
	      c1[k] += q1 * xy[k];
	      c2[k] += q2 * xy[k];
	    }
	  w1 += q1;
	  w2 += q2;
	}

      for (k=0; k<2; k++)
	{
	  c1[k] /= w1;
	  c2[k] /= w2;
	  cdiffxy[k] = c1[k]-c2[k];
	  cdiffr2 += cdiffxy[k]*cdiffxy[k];
	}
      hCoreDiffX -> Fill(cdiffxy[0]);
      hCoreDiffY -> Fill(cdiffxy[1]);
      hCoreDiffR -> Fill(sqrt(cdiffr2));
    }

}

// To recalculate the core with cut on maximum charge
bool pass1plot::recalcCore(Double_t qmax, Double_t *coreXY)

{
  Int_t i, j;
  Double_t xy[2];
  Double_t q, Q;
  // Calculate the original core position (coreXY0)

  for (j=0; j<2; j++)
    coreXY[j] = 0.0;

  Q=0.0;
  for (i=0; i<rufptn->nhits; i++)
    {
      if (rufptn->isgood[i]<4)
	continue;
      xy[0] = rufptn->xyzclf[i][0] - RUFPTN_ORIGIN_X_CLF;
      xy[1] = rufptn->xyzclf[i][1] - RUFPTN_ORIGIN_Y_CLF;
      q = (rufptn->pulsa[i][0]+rufptn->pulsa[i][1])/2.0;

      if (q>qmax)
	continue;
      Q += q;
      for (j=0; j<2; j++)
	coreXY[j] += q * xy[j];

    }

  if (Q<1e-3)
    return false;

  for (j=0; j<2; j++)
    coreXY[j] /= Q;

  return true;

}

bool pass1plot::areSadjacent(Int_t ih1, Int_t ih2, Double_t *r)
{
  Int_t xy1[2], xy2[2];
  xycoor(rufptn->xxyy[ih1], xy1);
  xycoor(rufptn->xxyy[ih2], xy2);
  (*r) = sqrt( (Double_t)( (xy1[0]-xy2[0])*(xy1[0]-xy2[0])+(xy1[1]-xy2[1])
			   *(xy1[1] - xy2[1]) ) );

  return ((((xy1[0] - xy2[0]) * (xy1[0] - xy2[0]) + (xy1[1] - xy2[1])
	    * (xy1[1] - xy2[1])) <= 2) ? true : false);
}

// To check validity of space-time pattern recognition
void pass1plot::chk_precog()
{
  Int_t ievent;
  Int_t ihit;
  Int_t i, j;
  Int_t ih1, ih2;
  Int_t nsc; // number of hits in space cluster
  Int_t sclust[0x100]; // hits in space-cluster

  Double_t r, t1, t2, dt, dq, q1, q2, qr, qf, rcore;

  // Loop over all events
  for (ievent=0; ievent<eventsRead; ievent++)
    {
      GetEntry(ievent);

      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)ievent/(Double_t)(eventsRead-1)*100.0,'%');
      fflush(stdout);

      // Gather hits in space cluster
      nsc = 0;
      for (ihit=0; ihit < rufptn->nhits; ihit++)
	{
	  if (rufptn->isgood[ihit] < 2)
	    continue;
	  sclust[nsc] = ihit;
	  nsc ++;
	}

      if (rufptn->nsclust != nsc)
	{
	  fprintf(stderr, "ALARM! Internal error\n");
	}

      // Go over hits in space cluster and consider each pair
      // of spatially adjacent neighbors once
      for (i = 0; i < nsc; i ++)
	{
	  ih1 = sclust[i];
	  for (j=(i+1); j<nsc; j++)
	    {
	      ih2 = sclust[j];
	      if (areSadjacent(ih1, ih2, &r))

		{
		  t1 = 0.5 * (rufptn->reltime[ih1][0]
			      + rufptn->reltime[ih1][1]);
		  t2 = 0.5 * (rufptn->reltime[ih2][0]
			      + rufptn->reltime[ih2][1]);

		  q1 = 0.5*(rufptn->pulsa[ih1][0]+rufptn->pulsa[ih1][1]);
		  q2 = 0.5*(rufptn->pulsa[ih2][0]+rufptn->pulsa[ih2][1]);

		  // Average distance from core for the 2 hits
		  rcore=0.5*(rufptn->tyro_cdist[2][ih1]
			     +rufptn->tyro_cdist[2][ih2]);

		  if (r < 0.5)
		    {
		      if (rufptn->xxyy[ih1] != rufptn->xxyy[ih2])
			continue;
		      if (rufptn->isgood[ih1] >= 3)
			{

			  dt = t2-t1;
			  dq = q2-q1;
			  qr = q2/q1;
			  qf = q2;
			}
		      else if (rufptn->isgood[ih2] >= 3)
			{
			  dt = t1-t2;
			  dq = q1-q2;
			  qr = q1/q2;
			  qf = q1;
			}
		      else
			{
			  continue;
			}

		      hQVsdT->Fill(dt, qf);
		      hdQVsdT->Fill(dt, dq);
		      hQrVsdT->Fill(dt, qr-1.0);
		      hTdiff1->Fill(dt);

		    }
		  else if (r > 0.5 && r < 1.3)
		    {
		      hTdiff2VsR->Fill(rcore, t2-t1);
		    }
		  else
		    {
		      hTdiff3VsR->Fill(rcore, t2-t1);
		    }

		}
	    }

	}

      //looping over all events
    }

  fprintf(stdout,"\n");

}
