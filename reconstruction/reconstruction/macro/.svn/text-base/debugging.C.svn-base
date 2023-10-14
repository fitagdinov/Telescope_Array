#define secIn1200m 4.0027691424e-6 // Seconds in a [1200m] unit
#define RUSDRAW_BR 0
#define RUSDRAW_LR 1
#define RUSDRAW_SK 2
#define RUSDRAW_BRLR 3
#define RUSDRAW_BRSK 4
#define RUSDRAW_LRSK 5
#define RUSDRAW_BRLRSK 6

void dummy()
{
}


void plot_bsdinfo()
{

  if(!p1.have_bsdinfo)
    {
      fprintf(stderr,"warning: must have bsdinfo bank!\n");
      return;
    }

  for (Int_t i=0; i<p1.bsdinfo->nbsds; i++)
    {
      Int_t xxyy = p1.bsdinfo->xxyy[i];
      c1->cd();
      pass1plot_drawmark(Double_t(xxyy/100),Double_t(xxyy%100),1.0,21,1);
      c2->cd();
      pass1plot_drawmark(Double_t(xxyy/100),Double_t(xxyy%100),1.0,21,1);
    }
  
  for (Int_t i=0; i<p1.bsdinfo->nsdsout; i++)
    {
      Int_t xxyy = p1.bsdinfo->xxyyout[i];
      c1->cd();
      pass1plot_drawmark(Double_t(xxyy/100),Double_t(xxyy%100),1.0,25,1);
      c2->cd();
      pass1plot_drawmark(Double_t(xxyy/100),Double_t(xxyy%100),1.0,25,1);
    }

}


bool findMultWf(Int_t nmin, Int_t istart=-1)
{

  Int_t i,j;
  Int_t starti,endi,nent;
  if(istart==-1) 
    {
      starti = (Int_t)p1.pass1tree->GetReadEvent();
    }
  endi = p1.eventsRead - 1;
  if (starti == endi) 
    {
      fprintf (stderr, "At the end of the event list\n");
      return false;
    }
  if (starti < 0 || starti > endi) 
    {
      fprintf (stderr, "istart must be in 0 - %d range\n",endi);
      return false;
    }
  for (i = (starti+1); i <= endi; i++)
    {
      p1.lookat(i);
      for (j=0; j<p1.rufptn->nhits; j++)
	{
	  if (p1.rufptn->nfold[j] >= nmin) return plotEventBoth(-1);
	} 
    }
  fprintf (stdout, "Didn't find any >= %d-fold hits\n",nmin);
  return false;
}





void find_usec()
{
  int i;
  Double_t diff[2];
  Int_t iwf;
  Double_t t0,t1;
  t0 = ((double)p1.rusdraw->usec)/1e6;
  diff[0] = 10.0;
  iwf = 0;
  for (i=0; i<p1.rusdraw->nofwf; i++)
    { 
      t1 = ((double)p1.rusdraw->clkcnt[i] / (double)p1.rusdraw->mclkcnt[i]);
      diff[1] = TMath::Abs(t1-t0);

      if (diff[1] < diff[0])
	{
	  diff[0] = diff[1];
	  iwf = i;
	}
      
    }

  printf ("WF: %d\n",iwf);


}

Double_t calc_usec(int iwf)
{
  Double_t usec;
  if (iwf < 0 || iwf > (p1.rusdraw->nofwf-1)) return -1;
  usec = 
    ((double)p1.rusdraw->clkcnt[iwf]/(double)p1.rusdraw->mclkcnt[iwf]) * 1e6;
  return usec;
}



void print_QvsR()

{


  Int_t hits_by_R[NWFMAX];
  

  Int_t i,j,k;
  Int_t npts;

  npts = 0;
  for (i=0; i<p1.rufptn->nhits;i++)
    {
      if(p1.rufptn->isgood[i] < 4) 
	continue;
      hits_by_R[npts] = i;
      npts++;
    }
  for (i=0; i<npts; i++)
    {
      for (j=(i+1); j<npts; j++)
	{
	  if (p1.rufptn->tyro_cdist[2][hits_by_R[j]] 
	      < p1.rufptn->tyro_cdist[2][hits_by_R[i]])
	    {
	      k = hits_by_R[i];
	      hits_by_R[i] = hits_by_R[j]; 
	      hits_by_R[j] = k;
	    }
	}
    }

  fprintf(stdout,"%s%10s%10s%10s\n","HIT","XXYY","R","Q");
  for (i=0; i<npts; i++)
    {
      j=hits_by_R[i];
      fprintf(stdout,"%02d%11.04d%10.2f%10.2f\n",
	      j,p1.rufptn->xxyy[j],
	      p1.rufptn->tyro_cdist[2][j],p1.charge[j][2]);
    }
  
}






// To write a root tree with geom. fit parameters
void writeGeomFitRt(char *rootFile, char *treeName)
{
 
  Int_t i,j,l;
  
  Int_t tower_id;
  Int_t trig_id;
  Int_t yymmdd;
  Int_t nevents;
  Int_t eventsWritten;
  Double_t theta,phi,xcore,ycore,t0;

  
  TFile *fl;
  fl = new TFile(rootFile,"recreate","Geom. var. root-tree file");
  TTree *t;
  t = new TTree(treeName,"Geom. variables");
  t->Branch("tower_id",&tower_id,"tower_id/I");
  t->Branch("trig_id",&trig_id,"trig_id/I");
  t->Branch("yymmdd",&yymmdd,"yymmdd/I");
  t->Branch("theta",&theta,"theta/D");
  t->Branch("phi",&phi,"phi/D");
  t->Branch("xcore",&xcore,"xcore/D");
  t->Branch("ycore",&ycore,"ycore/D");
  t->Branch("t0",&t0,"t0/D");  
   
  nevents = (Int_t)p1.GetEntries();
  eventsWritten = 0;
  fprintf(stdout,"nevents = %d\n",nevents);
  fprintf(stdout,"Filling geometry tree ...\n");
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);

      tower_id = p1.rusdraw->site;
      trig_id = p1.rusdraw->trig_id[p1.rusdraw->site];
      yymmdd = p1.rusdraw->yymmdd;
      
      theta = p1.rusdgeom->theta[2];
      phi = p1.rusdgeom->phi[2];
      xcore = p1.rusdgeom->xcore[2];
      ycore = p1.rusdgeom->ycore[2];
      t0 = p1.rusdgeom->t0[2];
      
      

      // Fill the tree for the event
      t->Fill();
      eventsWritten++;
      fprintf(stdout,"Completed: %.0f%c\r", (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout); 
    }
  fprintf(stdout,"\n");
   
  fprintf(stdout,"%d events written\n",eventsWritten);
  
  t->Write();
  fl->Close();
    
}


bool findMtEvent( Int_t tower_id = 3 )
{
  Int_t i;
  Int_t nevents;
  Int_t icur;
  if (tower_id > 6)
    return false;
  // Save the current event number
  icur = (Int_t)p1.pass1tree->GetReadEvent();
  // Loop over events
  nevents  = (Int_t)p1.GetEntries();
  for(i=0;i<nevents;i++)
    {
      p1.GetEntry(i);
      if(tower_id==p1.rusdraw->site)
	{
	  fprintf(stdout,"i=%d matches\n",i);
	}
    }
  // Restore the current event
  p1.GetEntry(icur);
  return true;
}

void pass1plot_pScat(TString hname,TString pname)
{
  TH2F *hist = (TH2F*)gROOT->FindObject(hname);
  TProfile *prof = (TProfile*)gROOT->FindObject(pname);
  prof->SetLineColor(2);
  hist->Draw("box");
  prof->Draw("same");  
}


void nextMCEvent()
{


  findCluster(4); 
  plotEventBoth(-1,-1.,-1.,29,29);
  c1->cd();
  pass1plot_drawmark(p1.rusdmc1->xcore,p1.rusdmc1->ycore,1.,21,2);
  
  
}


void findHeEvents(Double_t log10enMin=19.9)
{

  Double_t log10en;
  Double_t gfchi2pdof;
  Double_t ldfchi2pdof;

  FILE *fp=fopen("test.txt","w");
  fprintf(fp,"%s %15s %17s %17s %17s\n",
	  "Event number","log10(E/eV)","theta [Degree]",
	  "G.F. chi2/dof", "LDF chi2/dof");
  
  for(Int_t ievent=0; ievent<p1.eventsRead;ievent++)
    {
      p1.GetEntry(ievent);
      log10en=18.0 + TMath::Log10(p1.rufldf->energy[0]);

      gfchi2pdof  = ((p1.rusdgeom->ndof[1]>0) ? (p1.rusdgeom->chi2[1]/(Double_t)p1.rusdgeom->ndof[1]) : (p1.rusdgeom->chi2[1]));
      ldfchi2pdof = ((p1.rufldf->ndof[0]>0) ? (p1.rufldf->chi2[0]/(Double_t)p1.rufldf->ndof[0]) : (p1.rufldf->chi2[0]));
      
//       if(gfchi2pdof > 4.0)
// 	continue;
//       if (ldfchi2pdof > 4.0)
// 	continue;
//       if (p1.rusdgeom->theta[1] > 45.0)
// 	continue;
//       if (p1.rufldf->bdist < 1.0 || p1.rufldf->tdist < 1.0)
// 	continue;
      
      if (log10en > log10enMin)
	{
	  fprintf (fp, "%9d %15.2f %15.2f %18.2f %17.2f\n",
		   ievent,log10en,p1.rusdgeom->theta[1],gfchi2pdof,ldfchi2pdof);
	}
    }
  fclose(fp);
  
}


// bool printEventInfo(char *outfile=0)
// {  
//   Int_t xxyy;
//   Double_t detX,detY;
//   Int_t    fadcUpper,fadcLower;
//   Double_t vemLower,vemUpper;
//   Double_t onsetTime;
//   if ( ! outfile ) 
//     {
//       fprintf (stderr, "Specify the output file !\n");
//       return false;
//     }
//   FILE *fp;
//   fp = fopen(outfile,"w");
//   if ( fp ==0)
//     {
//       fprintf(stderr, "Can't open file %s\n",outfile);
//       return false;
//     }
//   Int_t isig;
//   for (isig=0; isig < p1.rufptn->nhits; isig++)
//     {
      
//       // Choose counters which are a part of the event
//       if (p1.rufptn->isgood[isig] < 4)
// 	continue;
//       xxyy = p1.rufptn->xxyy[isig];
//       detX = 1.2 * p1.rufptn->xyzclf[isig][0];
//       detY = 1.2 * p1.rufptn->xyzclf[isig][1];
//       fadcUpper=(Int_t)Floor(p1.rufptn->fadcpa[isig][0]+0.5);
//       fadcLower=(Int_t)Floor(p1.rufptn->fadcpa[isig][1]+0.5);
//       vemUpper=p1.rufptn->pulsa[isig][0];
//       vemLower=p1.rufptn->pulsa[isig][1];
      
//       onsetTime=
// 	0.5*(p1.rusdgeom->tearliest[0]+p1.rufptn->tearliest[1])
// 	+
// 	secIn1200m*0.5*(p1.rufptn->reltime[isig][0]+p1.rufptn->reltime[isig][1]);
//       fprintf (fp,"%04d %.3f %.3f %d %d %.2f %.2f %.9f\n",
// 	       xxyy,detX,detY,fadcUpper,fadcLower,vemUpper,
// 	       vemLower,onsetTime);
      
//     }
//   fclose(fp);
//   return true;
// }

bool printEventInfo(char *outfile=0)
{  
  Int_t xxyy;
  Double_t detX,detY;
  Int_t    fadcUpper,fadcLower;
  Double_t vemLower,vemUpper;
  Double_t onsetTime;
  FILE *fp;
  if ( ! outfile ) 
    {
      fprintf (stderr, "Specify the output file !\n");
      fp = stdout;
    }
  else
    {
      fp = fopen(outfile,"w");
      if ( fp ==0)
	{
	  fprintf(stderr, "Can't open file %s\n",outfile);
	  return false;
	}
    }
  Int_t isig;
  for (isig=0; isig < p1.rufptn->nhits; isig++)
    {
      
      // Choose counters which are a part of the event
      if (p1.rufptn->isgood[isig] < 4)
	continue;
      xxyy = p1.rufptn->xxyy[isig];
      detX = 1.2 * p1.rufptn->xyzclf[isig][0];
      detY = 1.2 * p1.rufptn->xyzclf[isig][1];
      fadcUpper=(Int_t)Floor(p1.rufptn->fadcpa[isig][0]+0.5);
      fadcLower=(Int_t)Floor(p1.rufptn->fadcpa[isig][1]+0.5);
      vemUpper=p1.rufptn->pulsa[isig][0];
      vemLower=p1.rufptn->pulsa[isig][1];
      
      onsetTime=
	0.5*(p1.rufptn->tearliest[0]+p1.rufptn->tearliest[1])
	+
	secIn1200m*0.5*(p1.rufptn->reltime[isig][0]+p1.rufptn->reltime[isig][1]);
      fprintf (fp,"%04d %.3f %.3f %d %d %.2f %.2f %.9f\n",
	       xxyy,detX,detY,fadcUpper,fadcLower,vemUpper,
	       vemLower,onsetTime);
      
    }
  fclose(fp);
  return true;
}


void print_rusdgeom( bool long_output = false, char *asciifile = 0)
{
  
  FILE *fp;
  Int_t i,j;


  if (asciifile == 0)
    {
      fp = stdout;
    }
  else
    {
      if ( (fp = fopen (asciifile, "w")) == 0)
	{
	  cerr << "Can't start " << asciifile << endl;
	  return;
	}
    }
  fprintf (fp, "%s :\n","rusdgeom");
  fprintf(fp,"nsds=%d tearliest=%.2f\n",p1.rusdgeom->nsds,p1.rusdgeom->tearliest);

  fprintf(fp,
	  "Plane fit  xcore=%.2f+/-%.2f ycore=%.2f+/-%.2f t0=%.2f+/-%.2f theta=%.2f+/-%.2f phi=%.2f+/-%.2f chi2=%.2f ndof=%d\n",
	  p1.rusdgeom->xcore[0],p1.rusdgeom->dxcore[0],
	  p1.rusdgeom->ycore[0],p1.rusdgeom->dycore[0],p1.rusdgeom->t0[0],
	  p1.rusdgeom->dt0[0],p1.rusdgeom->theta[0],p1.rusdgeom->dtheta[0],
	  p1.rusdgeom->phi[0],p1.rusdgeom->dphi[0],
	  p1.rusdgeom->chi2[0],p1.rusdgeom->ndof[0]);
  fprintf(fp,
	  "Modified Linsley fit  xcore=%.2f+/-%.2f ycore=%.2f+/-%.2f t0=%.2f+/-%.2f theta=%.2f+/-%.2f phi=%.2f+/-%.2f chi2=%.2f ndof=%d\n",
	  p1.rusdgeom->xcore[1],p1.rusdgeom->dxcore[1],p1.rusdgeom->ycore[1],p1.rusdgeom->dycore[1],
	  p1.rusdgeom->t0[1],p1.rusdgeom->dt0[1],p1.rusdgeom->theta[1],p1.rusdgeom->dtheta[1],
	  p1.rusdgeom->phi[1],p1.rusdgeom->dphi[1],p1.rusdgeom->chi2[1],p1.rusdgeom->ndof[1]);

  fprintf(fp,
	  "Mod. Lin. fit w curv.  xcore=%.2f+/-%.2f ycore=%.2f+/-%.2f t0=%.2f+/-%.2f theta=%.2f+/-%.2f phi=%.2f+/-%.2f a=%.2f+/-%.2f chi2=%.2f ndof=%d\n",
	  p1.rusdgeom->xcore[2],p1.rusdgeom->dxcore[2],p1.rusdgeom->ycore[2],p1.rusdgeom->dycore[2],
	  p1.rusdgeom->t0[2],p1.rusdgeom->dt0[2],p1.rusdgeom->theta[2],p1.rusdgeom->dtheta[2],
	  p1.rusdgeom->phi[2],p1.rusdgeom->dphi[2],p1.rusdgeom->a,p1.rusdgeom->da,
	  p1.rusdgeom->chi2[2],p1.rusdgeom->ndof[2]);
  
  
  fprintf(fp,"%s%18s%17s%16s%10s%8s\n",
	  "xxyy","pulsa,[VEM]","sdtime,[1200m]","sdterr,[1200m]","sdirufptn","igsd");
  for(i=0;i<p1.rusdgeom->nsds;i++)
    {
      fprintf(fp,"%04d%15f%15f%15f%11d%12d\n",
	      p1.rusdgeom->xxyy[i],p1.rusdgeom->pulsa[i],p1.rusdgeom->sdtime[i],
	      p1.rusdgeom->sdterr[i],p1.rusdgeom->sdirufptn[i],p1.rusdgeom->igsd[i]);
    }
  

  if (long_output)
    { 
      
      fprintf(fp,"\n%s%18s%17s%16s%10s%8s\n",
	      "xxyy","sdsigq,[VEM]","sdsigt,[1200m]","sdsigte,[1200m]","irufptn","igsig");
      for(i=0;i<p1.rusdgeom->nsds;i++)
	{
	  for(j=0;j<p1.rusdgeom->nsig[i];j++)
	    {
	      fprintf(fp,"%04d%15f%15f%15f%11d%12d\n",
		      p1.rusdgeom->xxyy[i],p1.rusdgeom->sdsigq[i][j],p1.rusdgeom->sdsigt[i][j],
		      p1.rusdgeom->sdsigte[i][j],p1.rusdgeom->irufptn[i][j],p1.rusdgeom->igsig[i][j]);
	    }
	  
	}
    }
  if (fp != stdout ) 
    fclose(fp);
}

bool print_sdtrgbk(FILE *fp = stdout)
{
  int xy[3][2];
  int isd,jsd,ksd;
  int isig,jsig,ksig;
   if (!p1.have_sdtrgbk)
    {
      fprintf(stderr,"branch 'sdtrgbk' not found\n");
      return false;
    }
  fprintf(fp, "igevent %d",     (int)p1.sdtrgbk->igevent);
  fprintf(fp," n_bad_ped %d",   p1.sdtrgbk->n_bad_ped);
  fprintf(fp," trigp %d\n",     p1.sdtrgbk->trigp);
  if (p1.sdtrgbk->igevent == 0)
    {
      fprintf(fp, "\nDID NOT TRIGGER\n");
    }
  else
    {
      
      fprintf(fp, "dec_ped: %d\n",p1.sdtrgbk->dec_ped);
      fprintf(fp, "inc_ped: %d\n",p1.sdtrgbk->inc_ped);
      
      // Informatino on level-2 trigger SDs and level-2 signals
      isd=p1.sdtrgbk->il2sd[0];
      jsd=p1.sdtrgbk->il2sd[1];
      ksd=p1.sdtrgbk->il2sd[2];
      isig=p1.sdtrgbk->il2sd_sig[0];
      jsig=p1.sdtrgbk->il2sd_sig[1];
      ksig=p1.sdtrgbk->il2sd_sig[2];
      
      fprintf(fp,
	      "TRIG SDs: %04d %04d %04d\n",
	      p1.sdtrgbk->xxyy[isd],
	      p1.sdtrgbk->xxyy[jsd],
	      p1.sdtrgbk->xxyy[ksd]
	      );
      fprintf(fp,"TRIG SD Q (Ql, Qu): (%d,%d) (%d,%d) (%d,%d)\n",
	      p1.sdtrgbk->q[isd][isig][0],p1.sdtrgbk->q[isd][isig][1],
	      p1.sdtrgbk->q[jsd][jsig][0],p1.sdtrgbk->q[jsd][jsig][1],
	      p1.sdtrgbk->q[ksd][ksig][0],p1.sdtrgbk->q[ksd][ksig][1]
	      );
      
      xy[0][0]=p1.sdtrgbk->xxyy[isd]/100;
      xy[0][1]=p1.sdtrgbk->xxyy[isd]%100;
      xy[1][0]=p1.sdtrgbk->xxyy[jsd]/100;
      xy[1][1]=p1.sdtrgbk->xxyy[jsd]%100;
      xy[2][0]=p1.sdtrgbk->xxyy[ksd]/100;
      xy[2][1]=p1.sdtrgbk->xxyy[ksd]%100;
      
      int mstyle = 29;
      double msize = 2.0;
      short int mcolor= 1;
      c1->cd();
      pass1plot_drawmark(xy[0][0],xy[0][1], msize, mstyle, mcolor);
      pass1plot_drawmark(xy[1][0],xy[1][1], msize, mstyle, mcolor);
      pass1plot_drawmark(xy[2][0],xy[2][1], msize, mstyle, mcolor);
      
      fprintf(fp,
	      "FADC Time Slices (20nS): %3d %3d %3d\n",
	      p1.sdtrgbk->ich[isd][isig],
	      p1.sdtrgbk->ich[jsd][jsig],
	      p1.sdtrgbk->ich[ksd][ksig]
	      );
      fprintf(fp,
	      "SECF(uS): %.2f %.2f %.2f\n",
	      p1.sdtrgbk->secf[isd][isig] * 1e6,
	      p1.sdtrgbk->secf[jsd][jsig] * 1e6,
	      p1.sdtrgbk->secf[ksd][ksig] * 1e6
	      );
    }
  return true;
}

void print_rusdraw(FILE *fp = stdout)
{
  Int_t i, j, k;
  Int_t yr, mo, day, hr, min, sec, usec, xy[2];
  fprintf (fp, "%s :\n","rusdraw");
  yr = p1.rusdraw->yymmdd / 10000;
  mo = (p1.rusdraw->yymmdd / 100) % 100;
  day = p1.rusdraw->yymmdd % 100;
  hr = p1.rusdraw->hhmmss / 10000;
  min = (p1.rusdraw->hhmmss / 100) % 100;
  sec = p1.rusdraw->hhmmss % 100;
  usec = p1.rusdraw->usec;
  fprintf (fp,"event_num %d event_code %d site ",
	   p1.rusdraw->event_num,p1.rusdraw->event_code);
  switch(p1.rusdraw->site)
    {
    case RUSDRAW_BR:
      fprintf(fp,"BR ");
      break;
    case RUSDRAW_LR:
      fprintf(fp,"LR ");
      break;
    case RUSDRAW_SK:
      fprintf(fp,"SK ");
      break;
    case RUSDRAW_BRLR:
      fprintf(fp,"BRLR ");
      break;
    case RUSDRAW_BRSK:
      fprintf(fp,"BRSK ");
      break;
    case RUSDRAW_LRSK:
      fprintf(fp,"LRSK ");
      break;
    case RUSDRAW_BRLRSK:
      fprintf(fp,"BRLRSK ");
      break;
    default:
      fprintf(fp,"%d ",p1.rusdraw->site);
      break;
    }
  fprintf(fp,"run_id: BR=%d LR=%d SK=%d trig_id: BR=%d LR=%d SK=%d\n",
	  p1.rusdraw->run_id[0],p1.rusdraw->run_id[1],p1.rusdraw->run_id[2],
	  p1.rusdraw->trig_id[0],p1.rusdraw->trig_id[1],p1.rusdraw->trig_id[2]);
  fprintf (fp,"errcode %d date %.02d/%.02d/%.02d %02d:%02d:%02d.%06d nofwf %d monyymmdd %06d monhhmmss %06d\n",
	   p1.rusdraw->errcode,mo, day, yr, hr, min,sec, p1.rusdraw->usec,
	   p1.rusdraw->nofwf,p1.rusdraw->monyymmdd,p1.rusdraw->monhhmmss);
  

  fprintf(fp,"%s\n",
	  "wf# wf_id  X   Y    clkcnt     mclkcnt   fadcti(lower,upper)  fadcav      pchmip        pchped      nfadcpermip     mftchi2      mftndof");
  for(i=0;i<p1.rusdraw->nofwf;i++)
    {
      xy[0] = p1.rusdraw->xxyy[i]/100;
      xy[1] = p1.rusdraw->xxyy[i]%100;
      fprintf(fp,"%02d %5.02d %4d %3d %10d %10d %8d %8d %5d %4d %6d %7d %5d %5d %8.1f %6.1f %6.1f %6.1f %5d %4d\n",
	      i,p1.rusdraw->wf_id[i],
	      xy[0],xy[1],p1.rusdraw->clkcnt[i],
	      p1.rusdraw->mclkcnt[i],p1.rusdraw->fadcti[i][0],p1.rusdraw->fadcti[i][1],
	      p1.rusdraw->fadcav[i][0],p1.rusdraw->fadcav[i][1],
	      p1.rusdraw->pchmip[i][0],p1.rusdraw->pchmip[i][1],
	      p1.rusdraw->pchped[i][0],p1.rusdraw->pchped[i][1],
	      p1.rusdraw->mip[i][0],p1.rusdraw->mip[i][1],
	      p1.rusdraw->mftchi2[i][0],p1.rusdraw->mftchi2[i][1],
	      p1.rusdraw->mftndof[i][0],p1.rusdraw->mftndof[i][1]);
    }
}


void print_rufptn(FILE *fp = stdout)
{
  Int_t i;
  fprintf (fp, "%s :\n","rufptn");
  fprintf(fp, 
	  "nhits %d nsclust %d nstclust %d nborder %d core_x %f core_y %f t0 %.9f \n",
	  p1.rufptn->nhits,p1.rufptn->nsclust,p1.rufptn->nstclust,p1.rufptn->nborder,
	  p1.rufptn->tyro_xymoments[2][0],
	  p1.rufptn->tyro_xymoments[2][1],
	  0.5*(p1.rufptn->tearliest[0]+p1.rufptn->tearliest[1])+p1.rufptn->tyro_tfitpars[2][0]*secIn1200m);
  
  
  fprintf(fp, "%s%8s%14s%15s%15s%18s%18s\n",
	  "#","XXYY","Q upper","Q lower","T upper","T lower","isgood");
  for(i=0; i<p1.rufptn->nhits; i++)
    {
      fprintf(fp,"%02d%7.4d%15f%15f%22.9f%18.9f%7d\n",
	      i,p1.rufptn->xxyy[i],p1.rufptn->pulsa[i][0],p1.rufptn->pulsa[i][1],
	      p1.rufptn->tearliest[0]+(4.0028e-6)*p1.rufptn->reltime[i][0],
	      p1.rufptn->tearliest[1]+(4.0028e-6)*p1.rufptn->reltime[i][1],
	      p1.rufptn->isgood[i]);
    }
  
}

void print_rusdmc(FILE *fp = stdout)
{
  const double RADDEG = TMath::RadToDeg();
  fprintf (fp, "%s :\n","rusdmc");
  fprintf (fp, "Event Number: %d\n", p1.rusdmc->event_num);
  fprintf (fp, "Corsika Particle ID: %d\n", p1.rusdmc->parttype);
  fprintf (fp, "Total Energy of Primary Particle: %g EeV\n", p1.rusdmc->energy);
  fprintf (fp, "Height of First Interaction: %g km\n", p1.rusdmc->height/1.e5);
  fprintf (fp, "Zenith Angle of Primary Particle Direction: %g Degrees\n", 
	   p1.rusdmc->theta*RADDEG);
  fprintf (fp, "Azimuth Angle of Primary Particle Direction: %g Degrees (N of E)\n",
	   p1.rusdmc->phi*RADDEG);
  fprintf (fp, "Counter ID Number for Counter Closest to Core: %d\n", 
	   p1.rusdmc->corecounter);
  fprintf (fp, "Position of the core in CLF reference frame: (%g,%g,%g) m\n",
	   p1.rusdmc->corexyz[0]/100., p1.rusdmc->corexyz[1]/100., p1.rusdmc->corexyz[2]/100.);
  fprintf (fp, "Time of shower front passing through core position: %d x 20 nsec\n",
	   p1.rusdmc->tc);
}

void get_event_time()
{
  fprintf(stdout,"number: %06d date: %06d time: %06d.%06d\n",
	  (Int_t)p1.pass1tree->GetReadEvent(),
	  p1.rusdraw->yymmdd,
	  p1.rusdraw->hhmmss,
	  p1.rusdraw->usec);
}

void find_bad_clkcnt()
{
  int nevents=p1.GetEntries();
  int iwf;
  int n_early;
  int n_late;
  for (int ievent=0; ievent<nevents; ievent++)
    {
      p1.GetEntry(ievent);
      n_early=0;
      n_late=0;
      for (iwf=0; iwf<p1.rusdraw->nofwf; iwf++)
	{
	  if(( ((double)p1.rusdraw->clkcnt[iwf]/(double)p1.rusdraw->mclkcnt[iwf])*1e6 - 
	       p1.rusdraw->usec) < -200.0)
	    {
	      fprintf(stdout,"event %d iwf=%03d too early\n",ievent,iwf);
	      n_early ++;
	    }
	  if((((double)p1.rusdraw->clkcnt[iwf]/(double)p1.rusdraw->mclkcnt[iwf])*1e6 - 
	      p1.rusdraw->usec) > 200.0)
	    {
	      fprintf(stdout,"event %d iwf=%03d too late\n",ievent,iwf);
	      n_late++;
	    }
	}
      if(n_early || n_late)
	fprintf(stdout,"bad_clkcnt: event %d date=%06d time=%06d.%06d n_early=%d n_late=%d\n",
		ievent,p1.rusdraw->yymmdd,p1.rusdraw->hhmmss,p1.rusdraw->usec,n_early,n_late);
    }
}

bool find_no_largest()
{
  Int_t curevent = (Int_t)p1.pass1tree->GetReadEvent();
  Int_t nevents  = (Int_t)p1.GetEntries();
  Int_t ievent = curevent+1;
  Int_t isd;
  Double_t vem_largest;
  Int_t igsd,xxyy,xx,yy;

  ievent = curevent+1;
  if(ievent >= nevents)
    {
      fprintf(stdout, "At the end of the list\n");
      return false;
    }
  while(ievent < nevents)
    {
      p1.GetEntry(ievent);
      vem_largest = 0.0;
      igsd = 0.0;
      for (Int_t isd=0; isd < p1.rusdgeom->nsds; isd++)
	{
	  if(p1.rusdgeom->pulsa[isd] > vem_largest)
	    {
	      vem_largest = p1.rusdgeom->pulsa[isd];
	      igsd = p1.rusdgeom->igsd[isd];
	      xxyy = p1.rusdgeom->xxyy[isd];
	    }
	}
      if(igsd < 2)
	{
	  fprintf(stdout, "ievent = %d\n", ievent);
	  plotEventBoth(ievent);
	  xx = xxyy/100;
	  yy = xxyy%100;
	  pfadc(xx,yy);
	  return true;
	}
      ievent++;
    }
  return false;
}

void plot_rusdmc1()
{
  int mstyle = 29;
  double msize = 2.0;
  short int mcolor= 2;
  if(!p1.haveMC)
    return;
  c2->cd();
  pass1plot_drawmark(p1.rusdmc1->xcore,p1.rusdmc1->ycore, msize, mstyle, mcolor);
}

void plot_rufldf()
{
  int mstyle = 29;
  double msize = 2.0;
  short int mcolor= 1;
  if(!p1.have_rufldf)
    return;
  c2->cd();
  pass1plot_drawmark(p1.rufldf->xcore[0],p1.rufldf->ycore[0], msize, mstyle, mcolor);
}
