using namespace TMath;


// Chi2/dof cut for expldf
const Double_t LDF_MCHI2PDOF = 4.0;

void ldf_graph_limits()
{
  Double_t xmin,xmax,ymin,ymax;
  xmin=ldf->gqvsr->GetXaxis()->GetXmin();
  xmax=ldf->gqvsr->GetXaxis()->GetXmax();
  ymin=ldf->ldfFun->Eval(xmax);
  ymax=ldf->ldfFun->Eval(xmin);
  ldf->gqvsr->GetYaxis()->SetRangeUser(ymin,ymax);
  ldf->ldfFun->SetRange(xmin,xmax);
}

bool ldfFit(bool fixCore = false)
{
  smallTitle();
  if(!ldf->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom))
    return false;
  ldf->Ifit(fixCore,(verbosity>=2));
  ldf->prepPlotVars();  
  c5->cd(1);
  gPad->Clear();
  ldf->gqvsr->SetMarkerColor(1); 
  ldf->gqvsr->SetMarkerStyle(20); 
  ldf->gqvsr->SetMarkerSize(0.7);
  ldf->gqvsr->Draw("a,e1p");
  gPad->Modified();
  ldf_graph_limits();
  ldf->ldfFun->SetLineColor(2);
  ldf->ldfFun->SetLineWidth(3);
  ldf->ldfFun->Draw("same");
  gPad->SetLogy();
  gPad->SetLogx();
  
  c5->cd(2);
  gPad->Clear();
  gPad->SetLogy(false);
  gPad->SetLogx(false);
  ldf->grsdvsr->SetMarkerColor(1); 
  ldf->grsdvsr->SetMarkerStyle(20); 
  ldf->grsdvsr->SetMarkerSize(0.7);
  ldf->grsdvsr->Draw("a1p");
  if(verbosity>=2)
    {
      fprintf(stdout,"%18s%20s\n",
	      "Geom. Fit",
	      "LDF Fit");
      
      fprintf(stdout,"coreX:%8.2f%22.2f\n",
	      p1.rusdgeom->xcore[1],ldf->R[0]);
      
      fprintf(stdout,"coreY:%8.2f%22.2f\n",
	      p1.rusdgeom->ycore[1],ldf->R[1]);
    }
  return true; 
}


void cleanLdf(Double_t deltaChi2 = 10.0, bool fixCore = true)
{
  
  smallTitle();
  if(!ldf->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom))
    return false;
  
  Int_t nDeletedPts = ldf->clean(deltaChi2,fixCore,(verbosity>=2));
  ldf -> Ifit(fixCore,(verbosity>=2));
  
  ldf->prepPlotVars();
  
  c5->cd(1);
  gPad->Clear();
  ldf->gqvsr->SetMarkerColor(1); 
  ldf->gqvsr->SetMarkerStyle(20); 
  ldf->gqvsr->SetMarkerSize(0.7);
  ldf->gqvsr->Draw("a1p");
  ldf->ldfFun->Draw("same");
  gPad->SetLogy();
  
  c5->cd(2);
  gPad->Clear();
  gPad->SetLogy(false);
  gPad->SetLogx(false);
  
  ldf->grsdvsr->SetMarkerColor(1); 
  ldf->grsdvsr->SetMarkerStyle(20); 
  ldf->grsdvsr->SetMarkerSize(0.7);
  ldf->grsdvsr->Draw("a1p");
  
  if(verbosity>=2)
    {
      fprintf(stdout,"%18s%20s\n",
	      "Geom. Fit",
	      "LDF Fit");
      
      fprintf(stdout,"coreX:%8.2f%22.2f\n",
	      p1.rusdgeom->xcore[2],ldf->R[0]);
      
      fprintf(stdout,"coreY:%8.2f%22.2f\n",
	      p1.rusdgeom->ycore[2],ldf->R[1]);
    }
}


 

// To make a simple root-tree with data necessary to do ldf fits.
// This file will be used for determining the modified time delay function
void writeLdfRt(const char* rootFile = "ldfdata.root", Int_t ngsds = 7)
{
  Int_t i,j,l;
  
  // Event ID's
  Int_t tower_id;
  Int_t trig_id;

  Int_t npts;         // Number of points to fit
  Int_t napts;        // Number of actual SDs which had a non-zero charge in them
  Double_t X[NWFMAX][3];  // Position of each counter, [1200m]
  Double_t rho[NWFMAX];   // Density (VEM/m^2)
  Double_t drho[NWFMAX];  // Sigma on density (VEM/m^2)
  Double_t sR[2];         // Starting value for core XY position
  Double_t theta,phi;     // Event direction, angles in degrees
  Double_t energy;        // Event energy, EeV
  Double_t mctheta,mcphi; // Thrown MC theta and phi, degree
  Double_t mcenergy;      // Thrown MC energy, EeV
  
  Int_t nevents;
  Int_t eventsWritten;
  
  Double_t gfchi2pdof;
  Double_t ldfchi2pdof;
  
  TFile *fl;
  fl = new TFile(rootFile,"recreate","Geom. var. root-tree file");
  TTree *ldfTree;
  ldfTree = new TTree("ldfTree","Geom. variables");
  ldfTree->Branch("tower_id",&tower_id,"tower_id/I");
  ldfTree->Branch("trig_id",&trig_id,"trig_id/I");
  ldfTree->Branch("npts",&npts,"npts/I");
  ldfTree->Branch("napts",&napts,"napts/I");
  ldfTree->Branch("X",X,"X[npts][3]/D");
  ldfTree->Branch("rho",rho,"rho[npts]/D");
  ldfTree->Branch("drho",drho,"drho[npts]/D");
  ldfTree->Branch("sR",sR,"sR[2]/D");
  ldfTree->Branch("theta",&theta,"theta/D");
  ldfTree->Branch("phi",&phi,"phi/D");
  ldfTree->Branch("energy",&energy,"energy/D");
  
  // Thrown MC variables
  ldfTree->Branch("mctheta",&mctheta,"mctheta/D");
  ldfTree->Branch("mcphi",&mcphi,"mcphi/D");
  ldfTree->Branch("mcenergy",&mcenergy,"mcenergy/D");

  
  nevents = (Int_t)p1.GetEntries();
  eventsWritten = 0;

  if(verbosity>=1)
    {
      fprintf(stdout,"nevents = %d\n",nevents);
      fprintf(stdout,"Filling LDF tree ...\n");
    }


  // dummy variables needed in calling the LDF fitter methods
  Bool_t verbose = false;
  Bool_t fixCore = false;


  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      
      if(verbosity>=1)
	{
	  fprintf(stdout,"Completed: %.0f%c\r", 
		  (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
	  fflush(stdout); 
	}
      
      // minimum number of counters in the space-time cluster
      if (p1.rufptn->nstclust < ngsds)
	continue;
      
      // Use events with no boundary counters
      if (p1.rufptn->nborder > 0)
	continue;
      
      // Geom. fit chi2/dof cut
      gfchi2pdof = 
	((p1.rusdgeom->ndof[2]>0) ? (p1.rusdgeom->chi2[2]/(Double_t)p1.rusdgeom->ndof[2]) : (p1.rusdgeom->chi2[2]));
      
      if (gfchi2pdof > LDF_MCHI2PDOF)
	continue;
      
      // Cut out events that are on the border      
      if (p1.rufldf->bdist<1.0 || p1.rufldf->tdist<1.0)
	continue;
      
      // If were not able to load the variables (not enough data points)
      if(!ldf->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom))
	continue;
      
      // If were not able to fit (not enough data points)
      if(!ldf->Ifit(verbose,fixCore))
	continue;
      
      // If there were no non-zero SDs in the fit
      if (ldf->napts < 1)
	continue;
      
      // LDF Chi2 cut
      ldfchi2pdof = ( (ldf->ndof>0) ? (ldf->chi2 / (Double_t)ldf->ndof) : (ldf->chi2) );
      
      if(ldfchi2pdof > LDF_MCHI2PDOF)
	continue;
      
      // fprintf(stdout,"%d\n",p1.rusdraw->site);
      // fflush(stdout);
      tower_id = p1.rusdraw->site;
      // trig_id = p1.rusdraw->trig_id[p1.rusdraw->site];

      trig_id = 0;

      // Load good fit points
      npts = 0;
      for(j=0;j<ldf->nfpts;j++)
	{
	  memcpy(X[npts],ldf->X[npts],3*sizeof(Double_t));
	  npts ++;
	}
      memcpy(rho,ldf->rho,npts*sizeof(Double_t));
      memcpy(drho,ldf->drho,npts*sizeof(Double_t));
      napts = ldf->napts;
      
      // Starting values for core (from geometry fit)
      sR[0] = p1.rusdgeom->xcore[1];
      sR[1] = p1.rusdgeom->ycore[1];
      
      // Event direction
      theta = p1.rusdgeom->theta[2];
      phi   = p1.rusdgeom->phi[2];

      // Event Energy
      energy = p1.rufldf->energy[0];
      
      // Thrown MC variables
      mctheta=RadToDeg()*p1.rusdmc->theta;
      mcphi=RadToDeg()*p1.rusdmc->phi;
      mcenergy=p1.rusdmc->energy;
      
      
      // Fill the tree for the event
      ldfTree->Fill();
      eventsWritten++;
    }
  if(verbosity>=1)
    {
      fprintf(stdout,"\n");   
      fprintf(stdout,"%d events written\n",eventsWritten);
    }
  ldfTree->Write();
  fl->Close();
  
}
