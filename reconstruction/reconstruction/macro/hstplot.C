
void wait_nsec(Double_t n_wait_sec=1.0)
{
  TStopwatch t;
  t.Start(kTRUE);
  while (t.RealTime() < n_wait_sec) t.Continue();
}

bool hstplot(char *hName = 0, TString plotfile="", FILE *fp = 0)
{
  TObject *ohst;
  TH1  *h1;
  Int_t ix,iy,nx,ny;
  bool saveplot;
  TPaveStats *stats;
  Double_t x1ndc,x2ndc,y1ndc,y2ndc;

  
  saveplot = (plotfile.Length() > 0 ? true : false);
  
  if (!hName)
    {
      cerr << "(1) hName: name of the histogram stored in the file" << endl;
      cerr << "(2) plotfile: (optional) save plot into a file" << endl;
      cerr << "(3) fp: ascii dump into the file descriptor" << endl;
      return false;
    }
  
  ohst = hstfl->Get(hName);
  if (ohst==0)
    {
      cerr << hName << " not found in the histogram file " << endl;
      return false;
    }  
  c1->cd();

  if(ohst->InheritsFrom("TH1"))
    {
      h1=(TH1 *) ohst;
      h1->SetLineWidth(2);
      h1->Draw();
      c1->Modified();
      c1->Update();
      if (fp)
	{
	  nx = h1->GetNbinsX();
	  fprintf(fp,"BINX:\tBCONT\n");
	  for (ix=1; ix<=nx; ix++)
	    {
	      fprintf(fp,"%.2f\t%5.5e\n",
		      h1->GetBinCenter(ix),
		      (int)h1->GetBinContent(ix)
		      );
	    }
	}
      stats=(TPaveStats *)h1->FindObject("stats");
      stats->SetOptStat(111110);
      stats->SetOptFit(0);
      x1ndc=stats->GetX1NDC();
      y1ndc=stats->GetY1NDC();
      x2ndc=stats->GetX2NDC();
      y2ndc=stats->GetY2NDC();
      h1->Draw();
      c1->Modified();
      c1->Update();
      if (saveplot)
	c1->SaveAs(plotfile);
    }
  else
    {
      cerr << "Type " << ohst->ClassName() << " is not supported" << endl;
      return false;
    }
  return true;
}

bool hstplot_and_save(char *hName=0, 
		      TString plotfile="", 
		      TString asciifile="")
{
  FILE *fp = 0;
  TString shellcmd;
  bool retflag = false;
  if (hName == 0)
    {
      cerr << "(1) hName: name of the histogram stored in the file" << endl;
      cerr << "(2) plotfile: (optional) save plot into a file" << endl;
      cerr << "(3) asciifile: ascii dump" << endl;
    }
  if (asciifile.Length() == 0)
    fp = 0;
  else
    fp = fopen(asciifile.Data(),"w");
  retflag=hstplot(hName,plotfile,fp);
  if (!retflag && fp)
    {
      fclose(fp);
      fp = 0;
      shellcmd = "rm -f ";
      shellcmd += asciifile;
      gSystem->Exec(shellcmd);
    }
  if (fp)
    fclose(fp);
  return retflag;
}

void mk_hist_plots(Int_t ihist_start = 0, 
		   Double_t n_wait_sec = 1.0, 
		   TString outbname="", 
		   TString fExt=".gif")
{
  
  Bool_t writeFile;
  TString hName,fName1,fName2;
  Int_t ihist,icut,ilayer;
  
  writeFile = (outbname.Length() > 0 ? true : false);
  
  if (ihist_start < 0 || ihist_start >= NDTMCHISTWCUTS)
    {
      cerr << "Warning: ihist_start should be in " << 0 << " to " 
	   << NDTMCHISTWCUTS-1 << " range" << endl;
    }

  // Main data/mc histograms with cut levels
  for (ihist = ihist_start; ihist < NDTMCHISTWCUTS; ihist ++ )
    {
      for ( icut = 0; icut < NCUTLEVELS; icut++)
	{
	  hName = dtmchist_wcuts[ihist];
	  hName += icut;

	  fprintf (stdout, 
		   "ihist = %d icut = %d hName = '%s'\n",
		   ihist,icut,hName.Data());
	  
	  if(writeFile)
	    {
	      fName1 = outbname;
	      fName1 += "_";
	      fName1 += hName;
	      fName1 += fExt;
	      fName2 = outbname;
	      fName2 += "_";
	      fName2 += hName;
	      fName2 += ".txt";
	      hstplot_and_save(hName,fName1,fName2);
	    }
	  else
	    hstplot(hName);
	  
	  wait_nsec(n_wait_sec);
	  
	} 
    }

  // Calibration data/mc histograms
  for (ihist = 0; ihist < NDTMCHISTCALIB; ihist++ )
    {
      for (ilayer=0; ilayer < 2; ilayer++)
	{
	  hName = dtmchist_calib[ihist];
	  hName += ilayer;
	  fprintf (stdout, "ihist = %d ilayer = %d hName = '%s'\n",
		   ihist,ilayer,hName.Data());
	  if(writeFile)
	    {
	      fName1 = outbname;
	      fName1 += "_";
	      fName1 += hName;
	      fName1 += fExt;
	      fName2 = outbname;
	      fName2 += "_";
	      fName2 += hName;
	      fName2 += ".txt";
	      hstplot_and_save(hName,fName1,fName2);
	    }
	  else
	    hstplot(hName);
	  
	  wait_nsec(n_wait_sec);
	}
    }
}
