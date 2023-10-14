void gldfFit(bool loadVariables=true)
{
  if(loadVariables)
    {
      if(!gldf->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom,
			      p1.rufldf))
	return false;
    }
  if(!gldf->Ifit(true))
    return false;
  
  Double_t u[2],v[2];
  Int_t i,l;
  short lcolor;
  Double_t q,cdist,ltd,lts;
  Double_t 
    x[NWFMAX],
    dx[NWFMAX],
    y[NWFMAX],
    dy[NWFMAX];
  Int_t n;
  Int_t xy[2];
  
  gldf->compVars();
  


  // Long and short axes
  u[0] = TMath::Cos(TMath::DegToRad()*gldf->phi);
  u[1] = TMath::Sin(TMath::DegToRad()*gldf->phi);
  v[0] = -u[1];
  v[1] = u[0];



  // Plot time delay vs. distance from core in ground plane
  c6->cd(1);
  gldf->gTrsdVsR->SetTitle("#DeltaT vs R");
  gldf->gTrsdVsR->GetXaxis()->SetTitle("Distance from core in ground plane, [1200m]");
  gldf->gTrsdVsR->GetYaxis()->SetTitle("#DeltaT, [1200m]");
  gldf->gTrsdVsR->SetMarkerStyle(20);
  gldf->gTrsdVsR->SetMarkerSize(0.7);
  gldf->gTrsdVsR->Draw("a1p");
  // Plot time delay vs. distance from core in shower front plane
  c6->cd(2);
  gldf->gTrsdVsS->SetTitle("#DeltaT vs S");
  gldf->gTrsdVsS->GetYaxis()->SetTitle("#DeltaT, [1200m]");
  gldf->gTrsdVsS->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  gldf->gTrsdVsS->SetMarkerStyle(20);
  gldf->gTrsdVsS->SetMarkerSize(0.7);
  gldf->gTrsdVsS->Draw("a1p");
  
  
  c5->cd(1);
  gPad->SetLogy();
  gldf->gRhoVsS->SetTitle("Charge Density vs S");
  gldf->gRhoVsS->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  gldf->gRhoVsS->GetYaxis()->SetTitle("Charge Density, [VEM/m^{2}]");
  gldf->gRhoVsS->SetMarkerStyle(20);
  gldf->gRhoVsS->SetMarkerSize(0.7);
  gldf->gRhoVsS->Draw("a1p");
  gldf->ldfFun->Draw("same");

  c5->cd(2);
  gPad->SetLogy(false);
  gldf->gRhoRsdVsS->SetTitle("Charge Density Residual vs S");
  gldf->gRhoRsdVsS->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  gldf->gRhoRsdVsS->GetYaxis()->SetTitle("Residual, [VEM/m^{2}]");
  gldf->gRhoRsdVsS->SetMarkerStyle(20);
  gldf->gRhoRsdVsS->SetMarkerSize(0.7);
  gldf->gRhoRsdVsS->Draw("a1p");
  

  c2->cd();
  pass1plot_drawmark(gldf->R[0],gldf->R[1],2.0);
  
  lcolor = 4;
  
  pass1plot_drawArrow (gldf->R, u, 0.05, 6.0, 2, lcolor);
  pass1plot_drawLine  (gldf->R, v, 3.0, 2, lcolor);

  
  gldf->gTvsU->Fit("pol1","F,0,Q");
  gldf->gTvsU->GetFunction("pol1")->ResetBit((1<<9));
  gldf->gTvsU->GetFunction("pol1")->SetLineWidth(1);
  gldf->gTvsU->GetFunction("pol1")->SetLineColor(2);


  c3->cd(2);  
  gPad->Clear();

  gldf->gTvsU->SetMarkerStyle(20); 
  gldf->gTvsU->SetMarkerSize(0.5);
  gldf->gTvsU->SetTitle("T vs U'");
  gldf->gTvsU->GetYaxis()->SetTitle("T, [1200m]");
  gldf->gTvsU->GetXaxis()->SetTitle("U',[1200m]");
  gldf->gTvsU -> Draw("AP");
  
}


void writeGldfRt(Int_t ngsds =7, Double_t dchi2 = 4., 
		 const char *rootFile = "gldfdata.root")
{
  Int_t i,j,l;

  Int_t    nsds;         // Number of points to fit
  Int_t    ntfsds;       // Number of SDs in time fit
  Int_t    nldfsds;      // Number of non-zero SDs in LDF fit 
  Int_t    xxyy[NWFMAX]; // Counte LIDs
  Int_t    pflag[NWFMAX];// 0: zero charge counter put in, 1: time fit only counter, 2: ldf and time fit counter
  Double_t X[NWFMAX][3]; // Position of each counter, [1200m]
  Double_t t[NWFMAX];    // Time of each counter, [1200m]
  Double_t dt[NWFMAX];   // Time resolution for each counter
  Double_t rho[NWFMAX];  // Density (VEM/m^2)
  Double_t drho[NWFMAX]; // Error on charge density
  Double_t sR[2];        // Starting value for core XY position
  Double_t stheta;       // Starting value for zenith angle
  Double_t sphi;         // Starting value for azimuthal angle
  Double_t st0;          // Starting value for time when the core hits the ground
  Double_t sS;           // Starting value for scale parameter
  
  
  Double_t S,dS;         // LDF scale parameter


  Int_t nevents;
  Int_t eventsWritten;


  ///////  For filling geometry histograms as we fill the root tree ////////
  Double_t theta,phi;
  Double_t mctheta,mcphi; // MC theta and phi
  Double_t dtheta,dphi;
  Double_t chi2,chi2pdof;
  Int_t ndof;
  
  gStyle->SetOptStat(1);
  hTheta->Reset();
  hPhi->Reset();
  hThetaResVsTheta->Reset();
  hPhiResVsTheta->Reset();
  pThetaResVsTheta->Reset();
  pPhiResVsTheta->Reset();
  hThetaResVsN->Reset();
  hPhiResVsN->Reset();
  pThetaResVsN->Reset();
  pPhiResVsN->Reset();
  hThetaRes->Reset();
  hPhiRes->Reset();
  hChi2->Reset();
  hChi2pDof->Reset();
  hNdof->Reset();


  
  //////////// For filling geom. root tree ///////////////////////
  TFile *fl;
  fl = new TFile(rootFile,"recreate","Geom. & LDF var. root-tree file");
  TTree *gldfTree;
  gldfTree = new TTree("gldfTree","Geom-LDF variables");
  
  gldfTree->Branch("nsds",   &nsds,    "nsds/I");
  gldfTree->Branch("ntfsds", &ntfsds,  "ntfsds/I");
  gldfTree->Branch("nldfsds",&nldfsds, "nldfsds/I");
  gldfTree->Branch("pflag",  &pflag,   "pflag[nsds]/I");
  gldfTree->Branch("xxyy",   xxyy,     "xxyy/I");
  gldfTree->Branch("X",      X,        "X[nsds][3]/D");
  gldfTree->Branch("t",      t,        "t[nsds]/D");
  gldfTree->Branch("dt",     dt,       "dt[nsds]/D");
  gldfTree->Branch("rho",    rho,      "rho[nsds]/D");
  gldfTree->Branch("drho",   drho,     "drho[nsds]/D");
  gldfTree->Branch("sR",     sR,       "sR[2]/D");
  gldfTree->Branch("stheta", &stheta,  "stheta/D");
  gldfTree->Branch("sphi",   &sphi,    "sphi/D");
  gldfTree->Branch("st0",    &st0,     "st0/D");
  gldfTree->Branch("sS",     &sS,      "sS/D");
 
  gldfTree->Branch("mctheta",&mctheta, "mctheta/D");
  gldfTree->Branch("mcphi",  &mcphi,   "mcphi/D");
  
  mctheta = 0.0;
  mcphi   = 0.0;
  
  nevents = (Int_t)p1.GetEntries();
  eventsWritten = 0;
  
  fprintf(stdout,"nevents = %d\n",nevents);
  fprintf(stdout,"Filling geometry and LDF fit tree ...\n");
  
  
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      

      // Add MC information if it's there
      if (p1.haveMC)
	{
	  mctheta =   p1.rusdmc->theta *  TMath::RadToDeg();
	  mcphi   =   p1.rusdmc->phi   *  TMath::RadToDeg();
	}

      // if (p1.rufptn->nborder > 0)
      // continue;

      if (p1.rufldf->bdist<1. || p1.rufldf->tdist<1.)
	continue;

      if(!gldf->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom,p1.rufldf)) 
	continue;

      if(p1.rufptn->nstclust < ngsds)
	continue;
      
      gldf->clean(dchi2,false);
      
      if (gldf->ntfitsds < ngsds)
	continue;
      if (gldf->nldffitsds < (ngsds-1))
	continue;
      
      // Make sure that the event is "fittable"
      if(!gldf->Ifit(false))
 	continue;
     

      //////////// FILLING LDF-GEOM. HISTOGRAMS ////////////////

      nsds    =  gldf->nfitsds;
      ntfsds  =  gldf->ntfitsds;
      nldfsds =  gldf->nldffitsds;
      
      theta   =  gldf->theta;
      phi     =  gldf->phi;
      dtheta  =  gldf->dtheta;
      dphi    =  gldf->dphi;
      
      S       =  gldf->S;
      dS      =  gldf->dS;
      
      chi2    =  gldf->chi2;
      ndof    =  gldf->ndof;
      
      
      if (ndof > 0) { chi2pdof = chi2 / (Double_t)ndof; }
      else { chi2pdof = 1.e5; }


      /****** CHI2 / DOF cut applied here ************/
      // Make sure that have a reasonable chi2 / dof
      // if(chi2pdof > mchi2pdof) continue;
      
      
      
      // Fill theta and phi histograms
      hTheta->Fill(theta);
      hPhi->Fill(phi);
      // Fill phi resolution histograms
      hPhiRes->Fill(TMath::Sin(TMath::DegToRad()*theta)*dphi);
      hPhiResVsTheta->Fill(theta,TMath::Sin(TMath::DegToRad()*theta)*dphi);
      pPhiResVsTheta->Fill(theta,TMath::Sin(TMath::DegToRad()*theta)*dphi);
      hPhiResVsN->Fill((double)nsds,TMath::Sin(TMath::DegToRad()*theta)*dphi);
      pPhiResVsN->Fill((double)nsds,TMath::Sin(TMath::DegToRad()*theta)*dphi);
      // Filling theta resolution histograms
      hThetaRes->Fill(dtheta);
      hThetaResVsTheta->Fill(theta,dtheta);
      pThetaResVsTheta->Fill(theta,dtheta);
      hThetaResVsN->Fill((double)nsds,dtheta);
      pThetaResVsN->Fill((double)nsds,dtheta);
      // Fill chi2 and ndof histograms
      hChi2->Fill(chi2);
      hChi2pDof->Fill(chi2pdof);
      hNdof->Fill((Double_t)ndof);
      

      ///////// FILLING GEOM. ROOT TREE ///////////////////
      
      // Load fit 
      memcpy(&xxyy[0],  &gldf->fxxyy[0],  NWFMAX*sizeof(Int_t));
      memcpy(&pflag[0], &gldf->fpflag[0], NWFMAX*sizeof(Int_t));

      Int_t isd=0;
      for(j=0;j<gldf->nfitsds;j++)
	{
	  memcpy(&X[isd][0],&gldf->fX[isd][0],3*sizeof(Double_t));
	  isd ++;
	}
      
      if (isd != nsds )
	{
	  fprintf(stderr,"Internal inconsistency!\n");
	  return;
	}
      
      
      memcpy(&t[0],    &gldf->ft[0],    NWFMAX*sizeof(Double_t));
      memcpy(&dt[0],   &gldf->fdt[0],   NWFMAX*sizeof(Double_t));
      memcpy(&rho[0],  &gldf->frho[0],  NWFMAX*sizeof(Double_t));
      memcpy(&drho[0], &gldf->fdrho[0], NWFMAX*sizeof(Double_t));
      
      // Starting values for geometry that are written into geom. root tree
      stheta  =  gldf->theta;
      sphi    =  gldf->phi;
      memcpy(sR,gldf->R,(Int_t)(2*sizeof(Double_t)));
      st0     =  gldf->T0;
      sS      =  gldf->S;
      
      // Fill the tree for the event
      gldfTree->Fill();
      eventsWritten++;
      fprintf(stdout,"Completed: %.0f%c\r", (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout); 
    }
  fprintf(stdout,"\n");
  fprintf(stdout,"%d events written\n",eventsWritten);
  gldfTree->Write();
  fl->Close();


  // Plot chi2 / dof
  c1->cd();
  hChi2pDof->Draw();
  
  // Plot theta and phi resolution
  c3->cd(1);
  hThetaRes->Draw();  
  c3->cd(2);
  hPhiRes->Draw();
  
  // Plot theta and phi resolution vs. number of counters
  c5->cd(1); gPad->SetLogx(false); gPad->SetLogy(false);
  pass1plot_plotScat(hThetaResVsN,pThetaResVsN);
  c5->cd(2); gPad->SetLogx(false); gPad->SetLogy(false);
  pass1plot_plotScat(hPhiResVsN,pPhiResVsN);
  
  // Plot chi2 and ndof histograms
  c6->cd(1);
  hChi2->Draw();
  c6->cd(2);
  hNdof->Draw();
  fprintf(stdout,"Mean chi2 = %f\n",hChi2->GetMean());
  fprintf(stdout,"Mean ndof = %f\n",hNdof->GetMean());
    
}
