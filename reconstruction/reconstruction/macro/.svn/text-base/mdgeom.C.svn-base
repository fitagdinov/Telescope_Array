
bool mdfit(bool fix_mc_geom = false, TCanvas* canv = 0)
{
  if(!mdgeomfitter_ptr)
    {
      fprintf(stderr,"error: mdgeomfitter class not loaded\n");
      return false;
    }
  mdgeomfitter* mdgeom = (mdgeomfitter* )mdgeomfitter_ptr;
 
  Int_t event_code = 0; // data or MC flag
  
  // checks
  if (!p1.have_hraw1 && !p1.have_mcraw)
    {
      fprintf(stderr,"error: mdfit: must have either hraw1 or mcraw branch\n");
      return false;
    }
  if(!p1.have_stpln)
    {
      fprintf(stderr,"errror: mdfit: must have stpln branch\n");
      return false;
    }
  if(fix_mc_geom && !p1.have_mc04)
    {
      fprintf(stderr,"error: mdfit: fix_mc_geom option requires presence of mc04 branch\n");
      return false;
    }
  
  // DATA or MC ? 
  if(p1.have_mc04)
    event_code = 0;
  else
    event_code = 1;

  // more checks
  if (event_code == 1 && !p1.have_hraw1)
    {
      fprintf(stderr,"error: mdfit: real data doesn't have hraw1 branch\n");
      return false;
    }
  if(event_code == 0 && !p1.have_mcraw)
    {
      fprintf(stderr,"error: mdfit: MC doesn't have mcraw branch\n");
      return false;
    }
  
  if(event_code == 0)
    {
      // if this is MC, fill out the hraw1 class variable
      if(!p1.get_hraw1())
	return false;
      if(!mdgeom->load_variables(p1.hraw1,p1.stpln,p1.mc04))
	return false;
    }
  else
    {
      if(!mdgeom->load_variables(p1.hraw1,p1.stpln,0))
	return false;
    }
  mdgeom->Ifit(fix_mc_geom);

  if(canv)
    canv->cd();
  
  TGraphErrors* gpts = mdgeom->GetDataPoints();
  TGraph*       gfit = mdgeom->GetFitTime();
  TGraph*       gtan = mdgeom->GetTanTime();
  
  gfit->SetLineColor(2);
  gfit->SetLineWidth(2);
  
  gtan->SetLineColor(4);
  gtan->SetLineWidth(2);
  
  gpts->Draw("a,e1p");
  gfit->Draw("l");
  gtan->Draw("l");

  mdgeomfitter_fitvar* fit = mdgeom->fit;
  TPaveStats *ptstats = new TPaveStats(0.62,0.795,0.98,0.995,"brNDC");
  ptstats->SetName("stats");
  ptstats->SetBorderSize(2);
  ptstats->SetFillColor(0);
  ptstats->SetTextAlign(12);
  TString s;
  s.Form("#chi^{2} / ndf = %.2f / %d",fit->chi2,fit->ndof);
  TText *text = ptstats->AddText(s);
  s.Form("T_{0} = %.2e #pm %.2e",fit->t0,fit->dt0);
  text = ptstats->AddText(s);
  s.Form("R_{P} = %.2e #pm %.2e",fit->rp,fit->drp);
  text = ptstats->AddText(s);
  s.Form("#Psi  = %.2e #pm %.2e",fit->psi,fit->dpsi);
  text = ptstats->AddText(s);
  ptstats->SetOptStat(0);
  ptstats->SetOptFit(100101);
  ptstats->Draw();
  
  // TLegend *leg = new TLegend(0.1,0.7,0.25,0.9,"");
  // leg->SetFillColor(0);
  // leg->SetBorderSize(1);
  // leg->AddEntry(gSgnl[FitNumber],"Sig","p");
  // leg->Draw("");
  
}


void writeMdGeomTree(char *outfile="mdgeom.root")
{
  
  // t->Branch("tbr",&tbr,"tbr/D");
  
  Int_t    event_code;          // 1 = data, 0 = MC
  Int_t    jday;                // MD date stamp
  Int_t    jsec;                // MD second stamp
  Int_t    msec;                // MD mili second stamp
  Int_t    ievent;              // event number in the tree with full event information
  Double_t mcenergy = 0.001;    // MC energy in EeV
  Double_t mctheta = 0;         // thrown value of zenith angle, [degree]
  Double_t mcphi   = 0;         // thrown value of azimuthal angle, [degree]
  Double_t mcpsi   = 0;         // thrown value of PSI, degree
  Double_t mcrp    = 0;         // thrown value of Rp, meters
  Double_t mcn[3]  = {0,0,0,};  // thrown shower detector plane normal unit vector
  Int_t    ntb;                 // number of tubes ( only good tubes used )
  Double_t tkl;                 // event track length, [degree]
  Double_t ctm;                 // event crossing time, [uS]
  Double_t phpt;                // average number of photons per good tube
  Double_t paltav;              // npe-averaged altitude in SDP for good tubes, [degree]
  Double_t npetot;              // total npe for good tubes
  Double_t npepdeg;             // npe per degree of track length
  Double_t tref;                // reference time [uS] with respect to GPS
  Double_t n[3];                // shower-detector plane normal unit vector
  Double_t v[TAFD_MAXTUBES][3]; // tube pointing direction, unit vector
  Double_t palt[TAFD_MAXTUBES]; // tube altitude in shower detector plane, [degree]
  Double_t pazm[TAFD_MAXTUBES]; // tube azimuth in the shower detector plane, [degree]
  Double_t npe[TAFD_MAXTUBES];  // tube npe    
  Double_t t[TAFD_MAXTUBES];    // tube time, [uS]
  Double_t dt[TAFD_MAXTUBES];   // error on tube time, [uS]
  Double_t tha[TAFD_MAXTUBES];  // channel A trigger threshold, [mV]
  Double_t thb[TAFD_MAXTUBES];  // channel B trigger threshold, [mV]
  
  TFile *fMdGeom = new TFile(outfile,"recreate");
  if (fMdGeom->IsZombie()) return;
  
  TTree *tMdGeom = new TTree("tMdGeom","");
  
  // data or MC ? 
  tMdGeom->Branch("event_code",&event_code,"event_code/I");

  tMdGeom->Branch("jday",&jday,"jday/I");
  tMdGeom->Branch("jsec",&jsec,"jsec/I");
  tMdGeom->Branch("msec",&msec,"msec/I");
  tMdGeom->Branch("ievent",&ievent,"ievent/I");
  
  // MC thrown variables (if this is data, then all thrown values are zero)
  tMdGeom->Branch("mcenergy",&mcenergy,"mcenergy/D");
  tMdGeom->Branch("mctheta",&mctheta,"mctheta/D");
  tMdGeom->Branch("mcphi",&mcphi,"mcphi/D");
  tMdGeom->Branch("mcpsi",&mcpsi,"mcpsi/D");
  tMdGeom->Branch("mcrp",&mcrp,"mcrp/D");
  tMdGeom->Branch("mcn",mcn,"mcn[3]/D");
  
  // Reconstruction variables
  tMdGeom->Branch("ntb",&ntb,"ntb/I");
  tMdGeom->Branch("tkl",&tkl,"tkl/D");
  tMdGeom->Branch("ctm",&ctm,"ctm/D");
  tMdGeom->Branch("phpt",&phpt,"phpt/D");
  tMdGeom->Branch("paltav",&paltav,"paltav/D");
  tMdGeom->Branch("npetot",&npetot,"npetot/D");
  tMdGeom->Branch("npepdeg",&npepdeg,"npepdeg/D");

  tMdGeom->Branch("tref",&tref,"tref/D");
  tMdGeom->Branch("n",n,"n[3]/D");
  tMdGeom->Branch("v",v,"v[ntb][3]/D");
  tMdGeom->Branch("palt",palt,"palt[ntb]/D");
  tMdGeom->Branch("pazm",pazm,"pazm[ntb]/D");
  tMdGeom->Branch("npe",npe,"npe[ntb]/D");
  tMdGeom->Branch("t",t,"t[ntb]/D");
  tMdGeom->Branch("dt",dt,"dt[ntb]/D");
  tMdGeom->Branch("tha",tha,"tha[ntb]/D");
  tMdGeom->Branch("thb",thb,"thb[ntb]/D");
  
  // checks
  if (!p1.have_hraw1 && !p1.have_mcraw)
    {
      fprintf(stderr,"error: writeMdGeomTree: must have either hraw1 or mcraw branch\n");
      return;
    }
  if(!p1.have_stpln)
    {
      fprintf(stderr,"errror: writeMdGeomTree: must have stpln branch\n");
      return;
    }
  // DATA or MC ? 
  if(p1.have_mc04)
    event_code = 0;
  else
    event_code = 1;
  // more checks
  if (event_code == 1 && !p1.have_hraw1)
    {
      fprintf(stderr,"error: real data doesn't have hraw1 branch\n");
      return;
    }
  if(event_code == 0 && !p1.have_mcraw)
    {
      fprintf(stderr,"error: MC doesn't have mcraw branch\n");
      return;
    }
  
  
  Int_t nevents = p1.GetEntries();
  Int_t events_written = 0;
  for (ievent=0; ievent<nevents; ievent++)
    {
      p1.GetEntry(ievent);
      if(event_code==0)
	{
	  if(!p1.get_hraw1())
	    return;
	}
      
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)ievent/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);
      
      // MD event time stamp
      jday = p1.hraw1->jday;
      jsec = p1.hraw1->jsec;
      msec = p1.hraw1->msec;
      
      
      // thrown values, if this is MC
      if (event_code == 0)
	{
	  mcenergy = p1.mc04->energy / 1e18;
	  for (int ix=0; ix<3; ix++)
	    mcn[ix] = p1.mc04->shwn[2][ix];
	  mctheta = p1.mc04->theta;
	  mcphi   = ATan2(-p1.mc04->uthat[1],-p1.mc04->uthat[0]) * RadToDeg();
	  if(mcphi < 0.0) mcphi += 360.0;
	  
	  // MC psi angle has to be re-calculated following the convention
	  // that it's an angle b/w the shower axis and the core unit vector
	  // with core unit vector lying in MD XY plane
	  mcpsi = ACos((mcn[1]*p1.mc04->uthat[0]-mcn[0]*p1.mc04->uthat[1])/
		       sqrt(1.0-mcn[2]*mcn[2])) * RadToDeg();
	  
	  mcrp = 0.0;
	  for (int ix=0; ix<3; ix++)
	    mcrp += p1.mc04->rpvec[2][ix] * p1.mc04->rpvec[2][ix];
	  mcrp = sqrt(mcrp);
	}
      else
	{
	  mctheta = 0.0;
	  mcphi   = 0.0;
	  mcpsi   = 0.0;
	  mcrp    = 0.0;
	  for (int ix=0; ix<3; ix++)
	    mcn[ix] = 0.0;
	}      
      tkl  = p1.stpln->tracklength[2];
      ctm  = p1.stpln->crossingtime[2];
      phpt = p1.stpln->ph_per_gtube[2];
      for (int ix=0; ix<3; ix++)
	n[ix] = p1.stpln->n_ampwt[2][ix];
      ntb = 0;
      tref = 1e20.0;
      paltav = 0.0;
      npetot = 0.0;
      for (int itube=0; itube<p1.stpln->ntube; itube++)
	{
	  if(p1.stpln->ig[itube] < 1)
	    continue;
	  
	  // tube pointing direction
	  p1.get_fd_tube_pd(2,p1.hraw1->tubemir[itube],p1.hraw1->tube[itube],v[ntb]);
	  
	  // tube altitude and azimuth in the shower detector plane
	  get_alt_azm_in_SDP(n,v[ntb],&palt[ntb],&pazm[ntb]);
	
	  // tube number of photo electrons
	  npe[ntb] = p1.MD_PMT_QE * p1.hraw1->prxf[itube];
	  
	  // tube time in uS
	  t[ntb] = ((double)p1.hraw1->mirtime_ns[0])/1e3 + p1.hraw1->thcal1[itube];
	  if(t[ntb] < tref) tref = t[ntb];
	  
	  // (preliminary) uncertainty on tube time
	  dt[ntb] = 0.055 + 0.225 / sqrt(npe[ntb]);
	  
	  // save the A and B threshold values
	  tha[ntb] = p1.hraw1->tha[itube];
	  thb[ntb] = p1.hraw1->thb[itube];
	  
	  npetot += npe[ntb]; // total npe
	  paltav += npe[ntb] * palt[ntb]; // npe-averaged altitude in the shower detector plane
	  
	  // count the good tubes
	  ntb ++;
	}
      
      // npe-averaged altitude in the shower detector plane
      if (npetot > 1e-3)
	paltav /= npetot;
      else
	paltav = 1e3;
      
      npepdeg = npetot / p1.stpln->tracklength[2]; 
      
      // subtract the common tube reference time
      for (int itube=0; itube < ntb; itube++)
	t[itube] -= tref;
      

      ///////////////////// CUTS (BELOW) ////////////////
      
      if (ntb < 20)   continue; // number of good tubes
      if (ctm < 20.0) continue; // crossing time
      if (tkl < 10.0) continue; // track length
      if (Abs(paltav) > 0.19) continue; // npe avaraged altitude in SDP
      if (RadToDeg()*ACos(Abs(n[2])) < 20.0) continue; // SDP angle
      ///////////////////// CUTS (ABOVE) ////////////////

      
      tMdGeom->Fill(); // fill the tree
      events_written ++;
    }
  tMdGeom->Write();
  fMdGeom->Close();
  fprintf(stdout,"\n");
  TString s="??";
  
  if(event_code==0)
    s="MC";
  if(event_code==1)
    s="DATA";
  
  fprintf(stdout,"%s: events_read = %d events_written = %d\n", 
	  s.Data(), nevents, events_written);
}

mdgeomfitter* get_mdgeom()
{
  return (mdgeomfitter_ptr ? (mdgeomfitter*)mdgeomfitter_ptr : 0);
}
