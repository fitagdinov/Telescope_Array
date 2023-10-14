#include <vector>


using namespace std;
// wLayer - to use what layer?  
// wLayer=0 - lower, wLayer=1 - upper, wLayer=2 - both
bool plotEvent(int wLayer, Double_t cx=-1.0, Double_t cy=-1.0,
	       Int_t nx=11, Int_t ny=11)
{
  
  if(curEdMode != ED_SD_MODE)
    sd_mode();
  
  Double_t cxy[2] = {cx, cy}; // xy coordinates of the center
  // if negative values given then make the event display center
  // coincide with the core position of the event
  if(cx<1.0 || cy <1.0) 
    {
      cxy[0] = p1.rufptn->tyro_xymoments[wLayer][0];
      cxy[1] = p1.rufptn->tyro_xymoments[wLayer][1];
    }

  else 
    {
      cxy[0] = cx;
      cxy[1] = cy;
    }
   
  fprintf (stdout, "ALL HITS:\n");
  drawEvent(wLayer,hEVENT,c1,cxy,nx,ny);
  
  fprintf (stdout, "SPACE-TIME CLUSTER: \n");
  vector<Int_t> iXXYYs;
  for(Int_t i=0;i<p1.rufptn->nhits;i++)
    {
      if (p1.rufptn->isgood[i] < 4) continue;
      iXXYYs.push_back(i);
    }
  if(iXXYYs.size())
    drawEvent(wLayer,hCLUSTER,c2,(Int_t)iXXYYs.size(),&iXXYYs[0],cxy,nx,ny);

 
  
  printf("1st moments: <x>=%f <y>=%f\n",  
	 p1.rufptn->tyro_xymoments[wLayer][0],p1.rufptn->tyro_xymoments[wLayer][1]);
  printf("2nd moments about (<x>,<y>):\n");
  printf("| <xx>   <xy> |     %f   %f\n", 
	 p1.rufptn->tyro_xymoments[wLayer][2],p1.rufptn->tyro_xymoments[wLayer][3]);
  printf("|             |  =\n");
  printf("| <xy>   <yy> |     %f   %f\n", 
	 p1.rufptn->tyro_xymoments[wLayer][3],p1.rufptn->tyro_xymoments[wLayer][4]);
  printf("Principal moments: %f %f\n",    
	 p1.rufptn->tyro_xypmoments[wLayer][0],p1.rufptn->tyro_xypmoments[wLayer][1]);
  printf("Long axis: (%f %f)\n",          
	 p1.rufptn->tyro_u[wLayer][0],p1.rufptn->tyro_u[wLayer][1]);
  printf("Short axis: (%f %f)\n",         
	 p1.rufptn->tyro_v[wLayer][0],p1.rufptn->tyro_v[wLayer][1]);
  
  printf ("Tyro event direction: PHI = %.2f, THETA =  %.2f (DEG)\n", 
	  p1.rufptn->tyro_phi[wLayer],p1.rufptn->tyro_theta[wLayer]);
  
  
  // Plot the pulse height histograms in the following order:
  // Both counters (top), upper (middle), lower (bottom)
  c4->cd();
  for(Int_t i=1;i<4;i++){ 
    c4->cd(i); 
    p1.hCharge[3-i]->Draw();
  }
  
  // Plot the T vs U graphs
  c3->cd(1);  
  gPad->Clear();
  gPad->SetLogx(0);
  gPad->SetLogy(0);
  // all counters, in black  
  p1.gTvsUall[wLayer]->SetMarkerColor(1); 
  p1.gTvsUall[wLayer]->SetMarkerStyle(20); 
  p1.gTvsUall[wLayer]->SetMarkerSize(1.2);
  p1.gTvsUall[wLayer]->SetTitle("T vs U (Tyro)");
  p1.gTvsUall[wLayer]->GetYaxis()->SetTitle("T, [1200m]");
  p1.gTvsUall[wLayer]->GetXaxis()->SetTitle("U, [1200m]");
  p1.gTvsUall[wLayer] -> Draw("AP");
  // space cluster, in green  
  p1.gTvsUsclust[wLayer]->SetMarkerColor(3); 
  p1.gTvsUsclust[wLayer]->SetMarkerStyle(20); 
  p1.gTvsUsclust[wLayer]->SetMarkerSize(1.2);
  p1.gTvsUsclust[wLayer]-> Draw("P");
  // cluster, in blue
  c3->cd(1);
  p1.gTvsUclust[wLayer]->SetMarkerColor(4); 
  p1.gTvsUclust[wLayer]->SetMarkerStyle(20); 
  p1.gTvsUclust[wLayer]->SetMarkerSize(1.2);
  p1.gTvsUclust[wLayer] -> Draw("P");


  // Plot T vs R graph
  c5->cd(1);  
  gPad->Clear();
  // all counters, in black  
  p1.gTvsRall[wLayer]->SetMarkerColor(1); 
  p1.gTvsRall[wLayer]->SetMarkerStyle(20); 
  p1.gTvsRall[wLayer]->SetMarkerSize(1.2);
  p1.gTvsRall[wLayer]->SetTitle("T vs R (Tyro)");
  p1.gTvsRall[wLayer]->GetXaxis()->
    SetTitle("Distance from core in ground plane, [1200m]");
  p1.gTvsRall[wLayer]->GetYaxis()->SetTitle("T, [1200m]");
  
  p1.gTvsRall[wLayer] -> Draw("AP");


  // space cluster, in green
  p1.gTvsRsclust[wLayer]->SetMarkerColor(3); 
  p1.gTvsRsclust[wLayer]->SetMarkerStyle(20); 
  p1.gTvsRsclust[wLayer]->SetMarkerSize(1.2);
  p1.gTvsRsclust[wLayer] -> Draw("P");
  
  // plane cluster, in blue
  p1.gTvsRclust[wLayer]->SetMarkerColor(4); 
  p1.gTvsRclust[wLayer]->SetMarkerStyle(20); 
  p1.gTvsRclust[wLayer]->SetMarkerSize(1.2);
  p1.gTvsRclust[wLayer] -> Draw("P");


  // Plot the T vs V graphs
//   c3->cd(2);  
//   gPad->Clear();
//   // all counters, in black  
//   p1.gTvsVall[wLayer]->SetMarkerColor(1); 
//   p1.gTvsVall[wLayer]->SetMarkerStyle(20); 
//   p1.gTvsVall[wLayer]->SetMarkerSize(1.2);
//   p1.gTvsVall[wLayer]->GetYaxis()->SetTitle("T, 1200m");
//   p1.gTvsVall[wLayer]->GetXaxis()->SetTitle("V, 1200m");
//   p1.gTvsVall[wLayer] -> Draw("AP");
//   // space cluster, in green
//   p1.gTvsVsclust[wLayer]->SetMarkerColor(3); 
//   p1.gTvsVsclust[wLayer]->SetMarkerStyle(20); 
//   p1.gTvsVsclust[wLayer]->SetMarkerSize(1.2);
//   p1.gTvsVsclust[wLayer] -> Draw("P");
//   // cluster, in blue
//   p1.gTvsVclust[wLayer]->SetMarkerColor(4); 
//   p1.gTvsVclust[wLayer]->SetMarkerStyle(20); 
//   p1.gTvsVclust[wLayer]->SetMarkerSize(1.2);
//   p1.gTvsVclust[wLayer] -> Draw("P");




  // Plot Charge vs Dist. from Shower Axis graph, on Log-Log scale
  c5->cd(2);
  gPad->SetLogy();
  gPad->Clear();
  // all counters (black)
  p1.gQvsSall[wLayer]->SetMarkerColor(1); 
  p1.gQvsSall[wLayer]->SetMarkerStyle(20); 
  p1.gQvsSall[wLayer]->SetMarkerSize(1.2);
  p1.gQvsSall[wLayer]->SetTitle("Charge vs S (Tyro)");
  p1.gQvsSall[wLayer]->GetXaxis()->
    SetTitle("Distance from shower axis, [1200m]");
  p1.gQvsSall[wLayer]->GetYaxis()->SetTitle("Charge, [VEM]");
  p1.gQvsSall[wLayer]->Draw("AP");
  // space cluster (green)
  p1.gQvsSsclust[wLayer]->SetMarkerColor(3); 
  p1.gQvsSsclust[wLayer]->SetMarkerStyle(20); 
  p1.gQvsSsclust[wLayer]->SetMarkerSize(1.2);
  p1.gQvsSsclust[wLayer]->Draw("P");
  // S-T cluster (blue)
  p1.gQvsSclust[wLayer]->SetMarkerColor(4); 
  p1.gQvsSclust[wLayer]->SetMarkerStyle(20); 
  p1.gQvsSclust[wLayer]->SetMarkerSize(1.2);
  p1.gQvsSclust[wLayer]->Draw("P");




  geomFit();
  ldfFit(false);

  return true;
  
}



bool plotEventLower(int eNumber=-1, 
		    Double_t cx=-1.0, 
		    Double_t cy=-1.0,
		    Int_t nx=11, 
		    Int_t ny=11)
{
  if (!p1.lookat(eNumber)) return false;
  return plotEvent(0,cx,cy,nx,ny);
}


bool plotEventUpper(int eNumber=-1, 
		    Double_t cx=-1.0, 
		    Double_t cy=-1.0,
		    Int_t nx=11, 
		    Int_t ny=11)
{
  if (!p1.lookat(eNumber)) return false;
  return plotEvent(1,cx,cy,nx,ny);
}


bool plotEventBoth(int eNumber=-1, 
		   Double_t cx=-1.0, 
		   Double_t cy=-1.0,
		   Int_t nx=11, 
		   Int_t ny=11)
{
  if (!p1.lookat(eNumber)) return false;
  return plotEvent(2,cx,cy,nx,ny);
}



// Find the largest cluster and plot the event.
// istart - start event ID
// n - # of counters in the cluster must be at least n 
// Qmin - sum of charges (averaged over
// upper and lower counters) must be at least Qmin
// One can then execute plotEventUpper(.. to see
// what's in the upper counter, etc
bool findCluster(int istart,int nmin,double Qmin){
  Int_t i = istart;
  while(true){ 
    if(p1.findCluster(i,nmin)){
      if(p1.qtot[2]>=Qmin) return plotEvent(2);
      i=p1.pass1tree->GetReadEvent();
    }
    else return false;
  }
}
bool findCluster(Int_t nmin, Double_t Qmin)
{ return findCluster(p1.pass1tree->GetReadEvent(),nmin,Qmin); }

bool findCluster(Int_t istart,Int_t nmin) 
{ return findCluster(istart,nmin,0.0); }

bool findCluster(Int_t nmin) 
{ if(p1.findCluster(nmin)) return plotEvent(2); return false; }

bool nextEvent()
{ return findCluster(0);}

// this moves to a next event after a given starting position
// return true if was able to do so and false otherwise
bool moveToNextEvent(Int_t ievent_start, Int_t nstclust_min)
{
  // return false if already at the end of the tree
  if(ievent_start == p1.GetEntries() - 1)
    return false;
  if(findCluster(ievent_start,nstclust_min))
    return true;
  // findCluster returns false if it found event with no SDs in the space-time cluste.  If, however,
  // user runs moveToNextEvent with nstclust_min = 0, then the outcome of the search is a success
  // and should be returned as such
  if(!nstclust_min && (p1.GetReadEvent() <= p1.GetEntries() - 1) && p1.GetReadEvent() != -1)
    return true;
  // return false otherwise
  return false;
}

bool findTrig (Int_t trig_id, Int_t site)
{
  if (!p1.findTrig(trig_id,site)) return false;
  // -1 to plot the event that we are currently on
  return plotEventBoth(-1);
}
bool findTrig (Int_t eNumber, Int_t trig_id, Int_t site)
{
  if (!p1.findTrig(eNumber,trig_id,site)) return false;
  return plotEventBoth(-1);
}

bool eventStartDate(Int_t yymmdd)
{
  Int_t ientry;
  Int_t nentries;
  Bool_t foundEvent;
  nentries=p1.GetEntries();
  
  foundEvent=false;
  for ( ientry=0; ientry < nentries; ientry++)
    {
      p1.GetEntry(ientry);
      if (yymmdd <= p1.rusdraw->yymmdd)
	{
	  foundEvent=true;
	  break;
	}
    }
  if (foundEvent)
    {
      return plotEventBoth(-1);
    }
  else
    {
      fprintf (stdout,"Can't find events occuring on/after yymmdd=%06d\n",
	       yymmdd);
      return false;
    }
}





// Some hints on how to use the plotting routines
void plotEvent(){
  printf("plotEvent(int wLayer)\n");
  printf("wLayer=0,1,2: draw using lower/both/both counters\n");
  printf("Where x,y are the coordinates of the center of the plot\n");
  printf ("Or run plotEvent(wLayer,x,y,cx,cy)\n");
  printf("Where nx, ny are the (integer) ED sizes in x and y\n");
}



void autoplayEvents(Int_t istart=0, Int_t nstclust=3, Double_t nsec = 5.0)
{
  // stop if the entry either doesn't exist (GetEntry returns 0) or there's an I/O
  // error (GetEntry returns -1)
  if(p1.GetEntry(istart) <= 0)
    return;
  // continue plotting events until not possible or ENTER has been pressed
  do {
    if(!moveToNextEvent((Int_t)p1.GetReadEvent(),nstclust))
      break;
    fprintf (stdout, "ievent = %d\n",(Int_t)p1.GetReadEvent());
    fprintf(stdout,"press <ENTER> to stop\n");
    fflush(stdout);
    updateAll();
  } while(p1.continue_activity(nsec));
}

void playEvents(Int_t istart=0, Int_t nstclust=3)
{
  // stop if the entry either doesn't exist (GetEntry returns 0) or there's an I/O
  // error (GetEntry returns -1)
  if(p1.GetEntry(istart) <= 0)
    return;
  // continue plotting events until not possible or <Q> and then <ENTER> were pressed
  for(TString response = ""; response != "q"; )
    {
      if(!moveToNextEvent((Int_t)p1.GetReadEvent(),nstclust))
        break;
      fprintf(stdout,"ievent = %d\n",(Int_t)p1.GetReadEvent());
      fprintf(stdout,"press <ENTER> to continue viewing\n");
      fprintf(stdout,"              or <Q> + <ENTER> to stop\n");
      fflush(stdout);
      updateAll();
      while(!p1.have_stdin());
      response=TString((Char_t)getchar());
    }
}


bool findEvent(Int_t yyyymmdd, Double_t hhmmssf, Double_t epsilon=1e-3,
	       Bool_t loop_over_all=false,
	       Bool_t plot_event=true)
{
  Int_t i,ilo,iup;
  Int_t nevents;
  Int_t icur;
  Int_t istart;
  Int_t hh,mm,ss, iyyyymmdd;
  Double_t fsec,ifsec;
  Int_t hhmmss;
  Int_t jd_need;  // julian day that corresponds to the date
  Int_t jd_start; // julian day from which to start the search
  Int_t jd_lo; // earliest possible jd in case events are time-sorted
  Int_t jd_hi; // latest possible jd in case events are time-sorted
  Int_t jd;
  // Save the current event number
  icur = (Int_t)p1.pass1tree->GetReadEvent();
  hhmmss=(Int_t)TMath::Floor(hhmmssf);
  hh=(hhmmss)/10000;
  mm=(hhmmss%10000)/100;
  ss=(hhmmss)%100;
  fsec=(Double_t)(hh*3600+mm*60+ss)+(hhmmssf-(Double_t)hhmmss);
  // to loop over the events
  nevents  = (Int_t)p1.GetEntries();
  
  // if events are sorted by date then one doesn't need to loop over all events
  // instead, one can do a binary search for the relevant date and then
  // search that date only
  istart=0;
  if(!loop_over_all)
    {
      if(yyyymmdd < 2000*10000)
	{
	  yyyymmdd += 2000*10000;
	  fprintf(stderr,"note: assuming the date is %08d\n", yyyymmdd);
	}
      jd_need = p1.greg2jd((yyyymmdd/10000),((yyyymmdd%10000)/100),(yyyymmdd%100));
      // jd start is one day before the event that we need.
      jd_start = jd_need-1;
      // find the earliest possible jd.  if this is <= jd_start then proceed scanning the events
      // starting at i=0.
      p1.GetEntry(0);
      iyyyymmdd=(p1.rusdraw->yymmdd)+20000000;
      jd_lo=p1.greg2jd((iyyyymmdd/10000),((iyyyymmdd%10000)/100),(iyyyymmdd%100));
      p1.GetEntry(nevents-1);
      iyyyymmdd=(p1.rusdraw->yymmdd)+20000000;
      jd_hi=p1.greg2jd((iyyyymmdd/10000),((iyyyymmdd%10000)/100),(iyyyymmdd%100));      
      if (jd_need < jd_lo || jd_need > jd_hi)
	{
	  Int_t yr,mo,da;
	  p1.jd2greg((double)jd_lo,&yr,&mo,&da);
	  Int_t yyyymmdd_start = yr*10000+mo*100+da;
	  p1.jd2greg((double)jd_hi,&yr,&mo,&da);
	  Int_t yyyymmdd_end = yr*10000+mo*100+da;

	  fprintf(stderr,"notice: event date must be in ");
	  fprintf(stderr,"%d(1st event)-%d(last event) range; ",yyyymmdd_start,yyyymmdd_end);
	  fprintf(stderr,"if events are not sorted, use 'loop_over_all' flag\n");
	  
	}
      if (jd_start>jd_lo)
	{
	  ilo=0;
	  iup=nevents-1;
	  while(true)
	    {
	      i=(iup+ilo)/2;
	      p1.GetEntry(i);
	      iyyyymmdd=(p1.rusdraw->yymmdd)+20000000;
	      jd=p1.greg2jd((iyyyymmdd/10000),((iyyyymmdd%10000)/100),(iyyyymmdd%100));
	      
	      if(jd>jd_start)
		iup=i;
	      else if(jd<jd_start && (iup-ilo)>1)
		ilo=i;
	      else
		{
		  istart=i;
		  break;
		}
	      if(ilo==iup)
		{
		  fprintf(stderr,"Binary search for the day before %08d failed, starting at the beginning\n",
			  yyyymmdd);
		  break;
		}
	    }
	}
    }
  for(i=istart;i<nevents;i++)
    {
      p1.GetEntry(i);
      iyyyymmdd=(p1.rusdraw->yymmdd)+20000000;
      if(iyyyymmdd != yyyymmdd)
	{
	  if(!loop_over_all && iyyyymmdd > yyyymmdd)
	    break;
	  continue;
	}
      hh=(p1.rusdraw->hhmmss)/10000;
      mm=(p1.rusdraw->hhmmss%10000)/100;
      ss=(p1.rusdraw->hhmmss)%100;
      ifsec=(Double_t)(hh*3600+mm*60+ss)+((Double_t)p1.rusdraw->usec)/1.0e6;
      if(iyyyymmdd==yyyymmdd && fabs(ifsec-fsec)<epsilon)
	{
	  // plot the event that has been found if the option is specified
	  if(plot_event)
	    {
	      fprintf(stdout,"i=%d matches\n",i);
	      return plotEventBoth(i);
	    }
	  fprintf(stdout,"i=%d matches\n",i);
	  // otherwise, restore the current event
	  p1.GetEntry(icur);
	  return true;
	}
    }
  // Restore the current event
  fprintf(stdout, "No matches\n");
  p1.GetEntry(icur);
  return false;
}
