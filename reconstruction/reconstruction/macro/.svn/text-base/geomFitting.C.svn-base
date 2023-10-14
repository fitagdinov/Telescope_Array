using namespace TMath;

// delta chi2 for cutting out bad SDs
const Double_t GEOM_DCHI2 = 4.0;

// Chi2/dof cut for expgeom, which determines the
// form of Linsley Td, Ts
const Double_t GEOM_MCHI2PDOF = 4.0;


void plotVars(Int_t whatFCN)
{  
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
  
  //   if(ftd) delete ftd;  
  //   ftd = new TF1("ftd",tdpr,0.0,10.0,1);
  
  sdgeom->compVars(whatFCN);
  


  // Long and short axes
  u[0] = TMath::Cos(TMath::DegToRad()*sdgeom->phi);
  u[1] = TMath::Sin(TMath::DegToRad()*sdgeom->phi);
  v[0] = -u[1];
  v[1] = u[0];



  // Plot time delay vs. distance from core in ground plane
  c6->cd(1);
  sdgeom->gTrsdVsR->SetTitle("#DeltaT vs R");
  sdgeom->gTrsdVsR->GetXaxis()->SetTitle("Distance from core in ground plane, [1200m]");
  sdgeom->gTrsdVsR->GetYaxis()->SetTitle("#DeltaT, [1200m]");
  sdgeom->gTrsdVsR->SetMarkerStyle(20);
  sdgeom->gTrsdVsR->SetMarkerSize(0.7);
  sdgeom->gTrsdVsR->Draw("a1p");
  // Plot time delay vs. distance from core in shower front plane
  c6->cd(2);
  sdgeom->gTrsdVsS->SetTitle("#DeltaT vs S");
  sdgeom->gTrsdVsS->GetYaxis()->SetTitle("#DeltaT, [1200m]");
  sdgeom->gTrsdVsS->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  sdgeom->gTrsdVsS->SetMarkerStyle(20);
  sdgeom->gTrsdVsS->SetMarkerSize(0.7);
  sdgeom->gTrsdVsS->Draw("a1p");



//   ftd->SetParameter(0,sdgeom->a);
//   ftd->Draw("same");

  
  
  c5->cd(1);
  gPad->SetLogy();
  sdgeom->gQvsS->SetTitle("Charge vs S");
  sdgeom->gQvsS->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
  sdgeom->gQvsS->GetYaxis()->SetTitle("Charge, [VEM]");
  sdgeom->gQvsS->SetMarkerStyle(20);
  sdgeom->gQvsS->SetMarkerSize(0.7);
  sdgeom->gQvsS->Draw("a1p");

  c2->cd();
  // pass1plot_drawmark(sdgeom->R[0],sdgeom->R[1],2.0);
  switch (whatFCN)
    {
      
    case 0:
      lcolor=2;
      break;
    case 1:
      lcolor = 4;
      break;
    case 2:
      lcolor = kBlack;
      break;
    case 3:
      lcolor = 5;
      break;
    case 4:
      lcolor = 3;
      break;

    case 6:
      lcolor = 7;
      break;

    default:
      lcolor=2;
    }
  c1->cd();
  pass1plot_drawArrow (sdgeom->R, u, 0.05, 6.0, 2, lcolor);
  pass1plot_drawLine  (sdgeom->R, v, 3.0, 2, lcolor);
  c2->cd();
  pass1plot_drawArrow (sdgeom->R, u, 0.05, 6.0, 2, lcolor);
  pass1plot_drawLine  (sdgeom->R, v, 3.0, 2, lcolor);
  c3->cd(2);  
  gPad->Clear();
  gPad->SetLogx(0);
  gPad->SetLogy(0);
  sdgeom->gTvsU->SetMarkerStyle(20); 
  sdgeom->gTvsU->SetMarkerSize(0.5);
  sdgeom->gTvsU->SetTitle("T vs U'");
  sdgeom->gTvsU->GetYaxis()->SetTitle("T, [1200m]");
  sdgeom->gTvsU->GetXaxis()->SetTitle("U',[1200m]");
  sdgeom->gTvsU -> Draw("AP");
}

bool geomFit(Int_t whatFCN = 2, bool loadVariables=true, bool verbose=false)
{
  if(loadVariables)
    {
      if(!sdgeom->loadVariables_stclust(p1.rufptn,p1.rusdgeom))
	return false;
    }
  if(!sdgeom->Ifit(whatFCN,verbose))
    return false;
  
  fprintf (stdout, "\n RESULTS:\n");
  fprintf (stdout, "%35s%25s\n", "Tyro", "New");
  fprintf (stdout, "Theta :%30f%25f\n",
	   p1.rufptn->tyro_theta[2],sdgeom->theta);
  fprintf (stdout, "Phi   :%30f%25f\n",p1.rufptn->tyro_phi[2],sdgeom->phi);
  fprintf (stdout, "Core X:%30f%25f\n",
	   p1.rufptn->tyro_xymoments[2][0],sdgeom->R[0]);
  fprintf (stdout, "Core Y:%30f%25f\n",
	   p1.rufptn->tyro_xymoments[2][1],sdgeom->R[1]);
  fprintf (stdout, "Core T0:%29f%25f\n",
	   p1.rufptn->tyro_tfitpars[2][0],sdgeom->T0);
  fprintf (stdout, "\n");


  plotVars(whatFCN);
 
  return true;
}


void cleanClust(Double_t deltaChi2=GEOM_DCHI2)
{
  Int_t npts;
  npts = sdgeom->cleanClust(deltaChi2);
  geomFit(2,false);
  fprintf(stdout,"Removed %d points\n",npts);
}




bool plotCont(Int_t ipar1=0, Int_t ipar2=1, 
	      Int_t npts=10,Double_t nsigma=1.0)
{

  sdgeom->gMinuit->SetErrorDef((Double_t)nsigma*nsigma);
  if(gCont) delete gCont;
  gCont = (TGraph *)sdgeom->gMinuit->Contour(npts,ipar1,ipar2);
  if(!gCont)
    return false;
  c1->cd();
  c1->Clear();
  gCont->Draw("alp");
  return true;
}


void rn180(Double_t *x)
{
  while((*x)<-180.0) (*x)+=360.0;
  while((*x)>180.0)  (*x)-=360.0;
}

void rn360(Double_t *x)
{
  while((*x)<0.0)    (*x)+=360.0;
  while((*x)>360.0)  (*x)-=360.0;
}


// Loops over events fits the geometry, and fills the histograms
Bool_t geomHist_Fit(Int_t whatFCN = 2)
{
  Int_t i,j; 
  Int_t ix,nbins;
  Double_t phi;
  Double_t dphi,dtheta;
 
  static Double_t *rsdval = 0;
  static Double_t *rsdsw2 = 0;
  Double_t npts;
  
  Double_t x,y,dy;
  Double_t xm;
  
  if(rsdval) 
    delete rsdval;
  if(rsdsw2)
    delete rsdsw2;  

  hPhi->Reset();
  hTheta->Reset();
  hPhiDiff->Reset();
  hPhiDiff->SetTitle("#phi_{Fit} - #phi_{Tyro}");
  hThetaDiff->Reset();
  hThetaDiff->SetTitle("#theta_{Fit} - #theta_{Tyro}");


  hPhiRes->Reset();
  hThetaRes->Reset();
  hGeomRes->Reset();
  
  hLinsleyA->Reset();
  hChi2->Reset();
  hChi2pDof->Reset();
  hNdof->Reset();
  hRsd->Reset();
  nbins = hRsd->GetNbinsX();
  xm = hRsd->GetXaxis()->GetXmax();
  
  rsdval = new Double_t[nbins];
  rsdsw2 = new Double_t[nbins];

  for(i=0;i<nbins;i++)
    {
      rsdval[i] = 0.0;
      rsdsw2[i] = 0.0;
    }


  fprintf(stdout,"Filling geometry histograms ...\n");

  for (i = 0; i < p1.eventsRead; i++)
    {
      p1.GetEntry(i);
      if(!sdgeom->loadVariables_stclust(p1.rufptn,p1.rusdgeom))
	continue;
      
      // Clean the space-time cluster from bad data points.
      // If number of degrees of freedom is less than 1, then
      // countinue
      sdgeom->cleanClust(GEOM_DCHI2,false);
      if(sdgeom->ngpts<7) 
	{
	  fprintf(stdout,"Removed event i=%d\n",i);
	  continue;
	}
      if(!sdgeom->Ifit(whatFCN,false))
 	continue;

      // Fill the resolution histograms for given FCN
      hPhiRes->Fill(TMath::Sin(TMath::DegToRad()*sdgeom->theta)*(sdgeom->dphi));
      hThetaRes->Fill(sdgeom->dtheta);

      if((sdgeom->chi2 / (Double_t)sdgeom->ndof) > GEOM_MCHI2PDOF)
	continue;
      
      sdgeom->compVars(whatFCN);
      npts=sdgeom->gTrsdVsS->GetN();
      for(j=0;j<npts;j++)
	{
	  sdgeom->gTrsdVsS->GetPoint(j,x,y);
	  dy = sdgeom->gTrsdVsS->GetErrorY(j);
	  if(x<xm && dy > 1e-5)
	    {
	      ix = (Int_t)((x/xm)*(Double_t)nbins);
 	      // rsdval[ix] += y/dy/dy;
	      rsdval[ix] += y;
 	      // rsdsw2[ix] += 1.0/dy/dy;
	      rsdsw2[ix] += 1.0;
	      if((y > 0.8) || (y < -0.8))
		{
// 		  fprintf(
// 			  stdout,
// 			  "Event : trig_id = %d, tower=%d\n",
// 			  p1.rusdraw->trig_id[p1.rusdraw->site],p1.rusdraw->site
// 			  );
		}
	      
	    }
	}
      
      for(j=0; j<nbins; j++)
	{
	  if(rsdsw2[j]>1e-3)
	    {
	      hRsd->SetBinContent(j+1,rsdval[j]/rsdsw2[j]);
	      hRsd->SetBinError(j+1,sqrt(1.0/rsdsw2[j]));
	      
	    }
	}
      
      hTheta->Fill(sdgeom->theta);
      
      phi=sdgeom->phi;
      rn360(&phi);
      hPhi->Fill(phi);
      
      
      dtheta=sdgeom->theta-p1.rufptn->tyro_theta[2];
      hThetaDiff->Fill(dtheta);
      
      dphi=(sdgeom->phi-p1.rufptn->tyro_phi[2]);
      rn180(&dphi);
      hPhiDiff->Fill(dphi);
      
      hLinsleyA->Fill(sdgeom->linsleyA);
      //fprintf(stdout, "CHI / NDOF = %.2f / %d\n",sdgeom->chi2, sdgeom->ndof);
      hChi2->Fill(sdgeom->chi2);
      hChi2pDof->Fill(sdgeom->chi2 / (Double_t)sdgeom->ndof);
      hNdof->Fill((Double_t)sdgeom->ndof);
      
      
      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)i/(Double_t)(p1.eventsRead-1)*100.0,'%');
      fflush(stdout);
    }
  fprintf(stdout,"\n");
  
  
  
  c3->cd(1);
  hTheta->Draw();
  
  c3->cd(2);
  hPhi->Draw();
  
  c5->cd(1); gPad->SetLogx(false); gPad->SetLogy(false);
  hThetaRes->Draw();
  //   hThetaDiff->Fit("gaus","Q");
  
  
  c5->cd(2); gPad->SetLogx(false); gPad->SetLogy(false);
  hPhiRes->Draw();
  //   hPhiDiff->Fit("gaus","Q");








  c2->cd();
  hLinsleyA->Draw();

  c6->cd(1);
  hChi2->Draw();

  c6->cd(2);
  hNdof->Draw();

//   c1->cd();
//   hRsd->Draw();

  fprintf(stdout,"Mean chi2 = %f\n",hChi2->GetMean());
  fprintf(stdout,"Mean ndof = %f\n",hNdof->GetMean());
  
  return true;
  
}






// Loops over events fits the geometry, and fills the histograms
void geomHist()
{
  Int_t i,j;
  Double_t theta,phi;
  Double_t stheta,dtheta,dphi;
  Double_t chi2,chi2pdof;
  Int_t ndof;
  Int_t nsds;

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
  hGeomRes->Reset();
  hChi2->Reset();
  hChi2pDof->Reset();
  hNdof->Reset();


  fprintf(stdout,"Filling geometry histograms ...\n");

  for (i = 0; i < p1.eventsRead; i++)
    {
      
      p1.GetEntry(i);
      theta  = p1.rusdgeom->theta[2];
      stheta = TMath::Sin(TMath::DegToRad() * theta);
      phi    = p1.rusdgeom->phi[2];
      dtheta = p1.rusdgeom->dtheta[2];
      dphi   = p1.rusdgeom->dphi[2];
      chi2   = p1.rusdgeom->chi2[2];
      ndof   = p1.rusdgeom->ndof[2];
      nsds   = p1.rufptn->nstclust;

      if (p1.rufptn->nstclust < 7)
	continue;


      // if (p1.rufptn->nborder > 0)
      //	continue;
      
      
      if (p1.rufldf->bdist < 1. || p1.rufldf->tdist < 1.)
	continue;
      
      
      
      if (ndof > 0)
	{ 
	  chi2pdof = chi2 / (Double_t)ndof;
	}
      else
	{
	  chi2pdof = -1.0;
	}
      
      if(chi2pdof > 4.0)
 	continue;
      
      

      // Fill theta and phi histograms
      hTheta->Fill(theta);
      hPhi->Fill(phi);

      // Fill the resolution histograms
      hPhiRes->Fill(stheta*dphi);
      hPhiResVsTheta->Fill(theta,stheta*dphi);
      pPhiResVsTheta->Fill(theta,stheta*dphi);
      hPhiResVsN->Fill((double)nsds,stheta*dphi);
      pPhiResVsN->Fill((double)nsds,stheta*dphi);
      
      if (dtheta > 3.0)
	fprintf (stdout, "Event %d dtheta = %f\n",i,dtheta);
      
      
      hThetaRes->Fill(dtheta);
      hThetaResVsTheta->Fill(theta,dtheta);
      pThetaResVsTheta->Fill(theta,dtheta);
      hThetaResVsN->Fill((double)nsds,dtheta);
      pThetaResVsN->Fill((double)nsds,dtheta);


      hGeomRes->Fill (sqrt(dtheta*dtheta+stheta*stheta*dphi*dphi));
      

      // Fill chi2 and ndof histograms
      hChi2->Fill(chi2);
      hChi2pDof->Fill(chi2pdof);
      hNdof->Fill((Double_t)ndof);
      
            
      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)i/(Double_t)(p1.eventsRead-1)*100.0,'%');
      fflush(stdout);
    }
  fprintf(stdout,"\n");

  c1->cd();
  hChi2pDof->Draw();
  
  c3->cd(1);
  hTheta->Draw();
  
  c3->cd(2);
  hPhi->Draw();
  
  c5->cd(1); gPad->SetLogx(false); gPad->SetLogy(false);
  // hThetaRes->Draw();
  // pass1plot_plotScat(hThetaResVsTheta,pThetaResVsTheta);

  pass1plot_plotScat(hThetaResVsN,pThetaResVsN);

  
  c5->cd(2); gPad->SetLogx(false); gPad->SetLogy(false);
  // hPhiRes->Draw();
  pass1plot_plotScat(hPhiResVsN,pPhiResVsN);
  c6->cd(1);
  hChi2->Draw();

  c6->cd(2);
  hNdof->Draw();

  fprintf(stdout,"Mean chi2 = %f\n",hChi2->GetMean());
  fprintf(stdout,"Mean ndof = %f\n",hNdof->GetMean());
  
}



void plotRsd()
{
  
  Int_t i;
  Int_t nevents;
  nevents = p1.GetEntries();
  sdgeom->cleanRsd();
  for (i = 0; i < p1.eventsRead; i++)
    {
      p1.GetEntry(i);
      sdgeom->loadVariables_stclust(p1.rufptn,p1.rusdgeom);
      
      // Clean the space-time cluster from bad data points.
      // If number of degrees of freedom is less than 1, then
      // countinue
      sdgeom->cleanClust(GEOM_DCHI2,false);
      if(sdgeom->ngpts < 7)
	continue;
      

      if((sdgeom->chi2 / (Double_t)sdgeom->ndof) > GEOM_MCHI2PDOF)
	continue;
      
      
      sdgeom->fillRsd();

      
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout); 
    }
  fprintf(stdout,"\n");
  
  c3->cd(1);
  pass1plot_plotScat(sdgeom->hRsdS,sdgeom->pRsdS);

  c3->cd(2);
  pass1plot_plotScat(sdgeom->hRsdRho,sdgeom->pRsdRho);
}







// To make plots of reconstruction differences b/w the plane and modified Linsley's fit
void plotRecDiff()
{
  Int_t i,j,l;
  Int_t nevents;
  

  Double_t phi_l,theta_l,phi_p,theta_p;
  Double_t dphi,dtheta;

  Double_t theta_t,phi_t;

  nevents = (Int_t)p1.GetEntries();
  
  

  gStyle->SetOptStat(1);

  
  hPhiDiff->SetTitle("sin(#theta) #times (#phi_{Linsley} - #phi_{Plane})");
  hPhiDiff->Reset();
  hThetaDiff->SetTitle("#theta_{Linsley} - #theta_{Plane}");
  hThetaDiff->Reset();
  
  
  fprintf(stdout,"nevents = %d\n",nevents);
  fprintf(stdout,"Filling reconstruction difference histograms ...\n");
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);


      theta_t =  p1.rufptn->tyro_theta[2];
      phi_t   =  p1.rufptn->tyro_phi[2];
      
      if(!sdgeom->loadVariables_stclust(p1.rufptn,p1.rusdgeom)) 
	continue;
      
      sdgeom->cleanClust(GEOM_DCHI2,false);
      
      // make sure there are still at least 7 good data points after cleaning the cluster
      if(sdgeom->ngpts < 7)
	continue;
      
      // Make sure that the event is "fittable"
      if(!sdgeom->Ifit(2,false))
 	continue;
      
      if((sdgeom->chi2 / (Double_t)sdgeom->ndof) > GEOM_MCHI2PDOF)
	continue;


      
      // Get the modified Linsley's fit values
      phi_l = sdgeom->phi;
      theta_l = sdgeom->theta;
      
      
      
      // Get the plane fit values
      sdgeom->Ifit(0,false);
      phi_p = sdgeom->phi;
      theta_p = sdgeom->theta;
      
      

      
      dtheta = theta_l-theta_p;
      dphi = phi_l - phi_p;
      
//       dtheta = theta_l-theta_t;
//       dphi = phi_l - phi_t;
      
      

      while(dphi>180.0) 
	dphi-=360.0;
      while(dphi<=-180.0) 
	dphi+= 360.0;


      dphi *= TMath::Sin(TMath::DegToRad()*theta_l);
      
      hPhiDiff->Fill(dphi);
      hThetaDiff->Fill(dtheta);
      

      fprintf(stdout,"Completed: %.0f%c\r", (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout); 
    }
  fprintf(stdout,"\n");


  c3->cd(1);
  hThetaDiff->Draw();
  c3->cd(2);
  hPhiDiff->Draw();
  
}




// To make a simple root-tree with data necessary to do geometry fits.
// This file will be used for determining the modified time delay function
void writeGeomRt(Double_t dchi2 = GEOM_DCHI2,
		 Double_t mchi2pdof = GEOM_MCHI2PDOF,
		 Int_t ngsds = 7,
		 const char *rootFile = "geomdata.root")
{
  Int_t i,j,l;

  Int_t tower_id;
  Int_t trig_id;

  Int_t npts;            // Number of points to fit
  Double_t X[NWFMAX][3]; // Position of each counter, [1200m]
  Double_t t[NWFMAX];    // Time of each counter, [1200m]
  Double_t dt[NWFMAX];   // Time resolution for each counter
  Double_t rho[NWFMAX];  // Density (VEM/m^2)
  Double_t sR[2];        // Starting value for core XY position
  Double_t stheta;       // Starting value for zenith angle
  Double_t sphi;         // Starting value for azimuthal angle
  Double_t st0;          // Starting value for time when the core hits the ground
  Double_t energy;       // Event energy, EeV
  
  Int_t nevents;
  Int_t eventsWritten;

 


  ///////  For filling geometry histograms as we fill the root tree ////////
  Double_t theta,phi;
  Double_t mctheta,mcphi;
  Double_t dtheta,dphi;
  Double_t chi2,chi2pdof;
  Int_t ndof;
  Int_t nsds;
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
  fl = new TFile(rootFile,"recreate","Geom. var. root-tree file");
  TTree *geomTree;
  geomTree = new TTree("geomTree","Geom. variables");
  geomTree->Branch("tower_id",&tower_id,"tower_id/I");
  geomTree->Branch("trig_id",&trig_id,"trig_id/I");
  geomTree->Branch("npts",&npts,"npts/I");
  geomTree->Branch("X",X,"X[npts][3]/D");
  geomTree->Branch("t",t,"t[npts]/D");
  geomTree->Branch("dt",dt,"dt[npts]/D");
  geomTree->Branch("rho",rho,"rho[npts]/D");
  geomTree->Branch("sR",sR,"sR[2]/D");
  geomTree->Branch("stheta",&stheta,"stheta/D");
  geomTree->Branch("sphi",&sphi,"sphi/D");
  geomTree->Branch("st0",&st0,"st0/D");
  geomTree->Branch("energy",&energy,"energy/D");
  geomTree->Branch("mctheta",&mctheta,"mctheta/D");
  geomTree->Branch("mcphi",&mcphi,"mcphi/D");
  
  
  mctheta = 0.0;
  mcphi   = 0.0;
  
  nevents = (Int_t)p1.GetEntries();
  eventsWritten = 0;
  fprintf(stdout,"nevents = %d\n",nevents);
  fprintf(stdout,"Filling geometry tree ...\n");
  

  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);

      tower_id = p1.rusdraw->site;
      //trig_id = p1.rusdraw->trig_id[p1.rusdraw->site];

      // Add MC information if it's there
      if (p1.haveMC)
	{
	  mctheta =   p1.rusdmc->theta *  TMath::RadToDeg();
	  mcphi   =   p1.rusdmc->phi   *  TMath::RadToDeg();
	}

      if (p1.rufptn->nborder > 0)
	continue;

      if (p1.rufptn->nstclust < ngsds)
	continue;

      if(!sdgeom->loadVariables_stclust(p1.rufptn,p1.rusdgeom)) 
	continue;
      
      sdgeom->cleanClust(dchi2,false);
      
      if (sdgeom->ngpts < ngsds)
	continue;
      
      // Make sure that the event is "fittable"
      if(!sdgeom->Ifit(2,false))
 	continue;
     

      //////////// FILLING GEOM. HISTOGRAMS ////////////////
      
      theta  = sdgeom->theta;
      phi    = sdgeom->phi;
      dtheta = sdgeom->dtheta;
      dphi   = sdgeom->dphi;
      chi2   = sdgeom->chi2;
      ndof   = sdgeom->ndof;
      nsds   = sdgeom->ngpts;
      if (ndof > 0) { chi2pdof = chi2 / (Double_t)ndof; }
      else { chi2pdof = chi2; }


      /****** CHI2 / DOF cut applied here ************/
      // Make sure that have a reasonable chi2 / dof
      if(chi2pdof > mchi2pdof) continue;
      
      
      
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
      
      // Load good fit points
      npts = 0;
      for(j=0;j<sdgeom->ngpts;j++)
	{
	  
	  l = sdgeom->goodpts[j];
	  X[npts][0] = p1.rufptn->xyzclf[l][0] - sd_origin_x_clf;
	  X[npts][1] = p1.rufptn->xyzclf[l][1] - sd_origin_y_clf;
	  X[npts][2] = p1.rufptn->xyzclf[l][2];     
	  t[npts] = 0.5 * (p1.rufptn->reltime[l][0]+p1.rufptn->reltime[l][1]);
	  dt[npts] = 0.5 * sqrt(p1.rufptn->timeerr[l][0]*p1.rufptn->timeerr[l][0]+
				p1.rufptn->timeerr[l][1]*p1.rufptn->timeerr[l][1]);
	  rho[npts] = 0.5 * (p1.rufptn->pulsa[l][0]+p1.rufptn->pulsa[l][1]) / 3.0;
	  npts ++;
	}



      
      
      // Starting values for geometry that are written into geom. root tree
      stheta = sdgeom->theta;
      sphi   = sdgeom->phi;
      memcpy(sR,sdgeom->R,(Int_t)(2*sizeof(Double_t)));
      st0    = sdgeom->T0;
      energy = p1.rufldf->energy[0];
      
      // Fill the tree for the event
      geomTree->Fill();
      eventsWritten++;
      fprintf(stdout,"Completed: %.0f%c\r", (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout); 
    }
  fprintf(stdout,"\n");
  fprintf(stdout,"%d events written\n",eventsWritten);
  geomTree->Write();
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





void plotQsides()
{
  
 
  Int_t ievent,ihit;
  Int_t nevents;

  Double_t xy[2],d[2],theta,phi;
  Double_t u,q,qU,qD;

  
  Double_t deg2rad = TMath::DegToRad();
 

  gStyle->SetOptStat(1);
  
  nevents = p1.GetEntries();

  fprintf(stdout,"Filling histograms\n");
  for(ievent=0;ievent<nevents;ievent++)
    {


      fprintf(stdout,"Completed: %.0f%c\r",
	      (Double_t)ievent/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);
      
      p1.GetEntry(ievent);
      
      

      theta = p1.rusdgeom->theta[2];
      phi = p1.rusdgeom->phi[2];

      qU = 0.0;
      qD = 0.0;
      for(ihit=0;ihit<p1.rufptn->nhits;ihit++)
	{
	  if(p1.rufptn->isgood[ihit] < 4)
	    continue;
	  
	  xy[0] = p1.rufptn->xyzclf[ihit][0] - sd_origin_x_clf;
	  xy[1] = p1.rufptn->xyzclf[ihit][1] - sd_origin_y_clf;
	  d[0] = xy[0]-p1.rusdgeom->xcore[2];
	  d[1] = xy[1]-p1.rusdgeom->ycore[2];
	  u = d[0]*cos(deg2rad*phi)+d[1]*sin(deg2rad*phi);
	  q = 0.5 * (p1.rufptn->pulsa[ihit][0]+p1.rufptn->pulsa[ihit][1]);
	  
	  if(u<0.0)
	    {
	      qU += q;
	    }
	  if(u>0.0)
	    {
	      qD += q;
	    }
	  
	}
      
      hQu->Fill(qU);
      hQd->Fill(qD);
      hQud->Fill(TMath::Log10(qU/qD));
      hQudVsTheta->Fill(theta,TMath::Log10(qU/qD));
      pQudVsTheta->Fill(theta,TMath::Log10(qU/qD));
      
    }


  

  c3->cd(1);
  hQu->Draw();

  c3->cd(2);
  hQd->Draw();

  c1->cd();
  hQud->Draw();

  c2->cd();
  pass1plot_plotScat(hQudVsTheta,pQudVsTheta);
  
  fprintf(stdout,"\n");
  
  
  
}
