using namespace TMath;

void sdps3fit()
{

  Double_t u[2], v[2], coreXY[2];
  Double_t xmin,xmax,ymin,ymax;

  TGraphErrors* g;

  if(!sdps3fitter_ptr)
    {
      fprintf(stderr,"error: sdps3fitter class not loaded\n");
      return;
    }
  sdps3fitter* sdps3 = (sdps3fitter* )sdps3fitter_ptr;
  
  if(!sdps3->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom,p1.rufldf))
    {
      cerr << "sdps3fit: loadVariables failed" << endl;
      return;
    }
  sdps3->Ifit();

   

  // Draw event arrow
  coreXY[0] = sdps3->fit.R[0];
  coreXY[1] = sdps3->fit.R[1];
  u[0] = cos(DegToRad()*sdps3->fit.phi);
  u[1] = sin(DegToRad()*sdps3->fit.phi);
  v[0] = -u[1];
  v[1] = u[0];
  c2->cd();
  pass1plot_drawArrow (coreXY, u, 0.05, 6.0, 4, 1);
  pass1plot_drawLine  (coreXY, v, 3.0, 4, 1);


  c5->cd(2);
  gPad->Clear();
  gPad->SetLogx();
  gPad->SetLogy();
  sdps3->compPlots();


  g = sdps3->gLdf_sdps3fitter;
  g->SetMarkerStyle(20);
  g->SetMarkerSize(1.0);
  g->SetLineWidth(2);
  xmin=g->GetXaxis()->GetXmin();
  xmax=g->GetXaxis()->GetXmax();
  ymin=sdps3->fLdf_sdps3fitter->Eval(xmax);
  ymax=sdps3->fLdf_sdps3fitter->Eval(xmin);
  g->GetYaxis()->SetRangeUser(ymin,ymax);
  g->Draw("a,e1p");
  sdps3->fLdf_sdps3fitter->Draw("same");
}



void print_sdps3_sdinfo()
{
  if(!sdps3fitter_ptr)
    {
      fprintf(stderr,"error: sdps3fitter class not loaded\n");
      return;
    }
  sdps3fitter *sdps3 = (sdps3fitter* )sdps3fitter_ptr;
  
  sdps3fitter_fitvar* fit;
  fit = &sdps3->fit; 
  Int_t isd;
  

  fprintf (stdout, 
	   "%s %8s %8s %10s %5s %10s %7s\n", 
	   "xxyy", 
	   "s[1200m]", 
	   "t[1200m]", 
	   "rho[VEM/m^2]",
	   "trsd",
	   "rhorsd",
	   "sdflag"
	   );
  
  for (isd = 0; isd < fit->nsd; isd++)
    {
      fprintf (stdout, 
  	       "%04d %6.2f %8.2f %8.2f %11.2f %9.2f %5d\n",
  	       fit->sdxxyy[isd], 
  	       fit->sdsdist[isd],
  	       fit->sdtim[isd],
  	       fit->sdrho[isd],
	       (fit->sdtim[isd]-fit->sdtimexp[isd]),
	       (fit->sdrho[isd]-fit->sdrhoexp[isd]),
	       fit->sdflag[isd]
  	       );
    }
  
}

void print_sdps3_satsdinfo( char *asciiname = 0 )
{
  Int_t ievent,nevents;
  Int_t isd;
  bool have_sat;
  FILE *fp;
  
  if(!sdps3fitter_ptr)
    {
      fprintf(stderr,"error: sdps3fitter class not loaded\n");
      return;
    }
  sdps3fitter* sdps3 = (sdps3fitter* )sdps3fitter_ptr;
  
  sdps3fitter_fitvar* fit = &sdps3->fit;
  if (asciiname)
    {
      if ( !(fp = fopen (asciiname,"w")) )
	{
	  cerr "Can't start " << asciiname << endl;
	  return;
	}
    }
  else 
    {
      fp = stdout;
    }
  nevents=p1.GetEntries();
  for (ievent=0; ievent < nevents; ievent ++)
    {
      p1.GetEntry(ievent);
      have_sat = false;
      for (isd=0; isd < p1.rusdgeom->nsds; isd++)
	{
	  if (p1.rusdgeom->igsd[isd] == 3)
	    {
	      have_sat = true;
	      break;
	    }
	} 
      if (have_sat)
	{
	  sdps3->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom,p1.rufldf);
	  sdps3->Ifit();
	  for (isd = 0; isd < fit->nsd; isd++)
	    {
	      if (fit->sdflag[isd] == 1)
		{
		  fprintf (fp, "%f %f %f %f\n", 
			   fit->theta, fit->sdsdist[isd], fit->sdrho[isd], fit->sdrhoexp[isd]);
		}
	    }
	}
    }
  if (asciiname)
    fclose(fp);
}
