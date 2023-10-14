{
  
  // Maximum size of the event display
  const Int_t MAXEDSIZE = 50;
  
  // Pixel resolution
  const Int_t NEDPRES = 40;
  
  // maximum number of SDs (for plotting FADC traces)
  const Int_t NSDMAX = 512;
  
  // maximum number of waveforms per event
  const Int_t NWFMAX = 0x400;
  
  // minimuim and maximum for XY coordinates
  const Int_t SD_X_MAX = 24;
  const Int_t SD_Y_MAX = 28;
  
  // which mode the ED is in, 0 = SD, 1 = FD
  const Int_t ED_SD_MODE  = 0;
  const Int_t ED_FD_MODE  = 1;
  const Int_t ED_NO_MODE  = -1;
  
  Bool_t use_grid  = true;  // if off then don't put on the grid
  Bool_t use_ticks = true;  // if off then don't put on the ticks

  Bool_t Draw_Tyro_Arrow = false; // if true, Tyro analysis arrows are drawn
  
  // ASCII file with paths to ROOT tree files
  TString start_file_name = "currentChain.txt"; // default
  if(gSystem->Getenv("PASS1PLOT_FLIST")) // if passed using the environmental variable PASS1PLOT_FLIST
    start_file_name = gSystem->Getenv("PASS1PLOT_FLIST");
  
  Int_t verbosity = 1;

  // which mode the ED is in
  Int_t curEdMode = ED_NO_MODE;
  
  // Pixes for drawing events
  TH2F *hEVENT =
    new TH2F("hEVENT","Event",NEDPRES*MAXEDSIZE,0.0,(Double_t)MAXEDSIZE,
	     NEDPRES*MAXEDSIZE,0.0,(Double_t)MAXEDSIZE);
  hEVENT->GetXaxis()->SetLabelSize(0.04);
  hEVENT->GetXaxis()->SetTitle("Distance East,  [1200m]");
  hEVENT->GetYaxis()->SetLabelSize(0.04);
  hEVENT->GetYaxis()->SetTitle("Distance North, [1200m]");
  hEVENT->Reset();
  hEVENT->SetStats(0);

  // Pixes for drawing events in a ST cluster
  TH2F *hCLUSTER =
    new TH2F("hCLUSTER","Cluster",NEDPRES*MAXEDSIZE,0.0,(Double_t)MAXEDSIZE,
	     NEDPRES*MAXEDSIZE,0.0,(Double_t)MAXEDSIZE);
  hCLUSTER->GetXaxis()->SetLabelSize(0.04);
  hCLUSTER->GetXaxis()->SetTitle("Distance East,  [1200m]");
  hCLUSTER->GetYaxis()->SetLabelSize(0.04);
  hCLUSTER->GetYaxis()->SetTitle("Distance North, [1200m]");
  hCLUSTER->Reset();
  hCLUSTER->SetStats(0);

  TH1F *hFADCD = new TH1F("hFADCD","FADC derivative",128,-0.5,127.5);
  TH1F *hFADC[2] = {0,0}; // Full FADC trace for a given counter
  
  // for plotting many FADC traces on the top of each other
  TH1F *hFADC_all_counters[NSDMAX][2];
  TH1F *hFADC_chosen_counters[NSDMAX][2];

  // frame for plotting FADC traces on the top of each other
  TH1F *hFADCfrm = new TH1F("hFADCfrm","",100000,-10.0,1000.0);
  
  TGraph *gCont = 0; // To plot TMinuit contours
  TF1 *ftd = 0; // time delay function for plotting


  // Histograms for geom. fit    

  //    TH1F *hTheta = new TH1F ("hTheta","Theta",13,-2.5,62.5);
  TH1F *hTheta = new TH1F ("hTheta","Theta",90,0.0,0.0);
  TH1F *hPhi = new TH1F ("hPhi","Phi",180,0.0,0.0);
  TH1F *hThetaDiff = new TH1F ("hThetaDiff","#theta_{Fit} - #theta_{Tyro}",90,-90.0,90.0);
  hThetaDiff->GetXaxis()->SetTitle("#Delta#theta, [Degree]");
  TH1F *hPhiDiff = new TH1F ("hPhiDiff","#phi_{Fit} - #phi_{Tyro}",180,-180.0,180.0);
  hPhiDiff->GetXaxis()->SetTitle("#Delta#phi, [Degree]");
  TH1F *hLinsleyA = new TH1F ("hLinsleyA","Linsley a",40,0.0,4.0);
  hLinsleyA->GetXaxis()->SetTitle("Linsley a");
  TH1F *hChi2 = new TH1F ("hChi2","FCN (#chi^{2}) ",50,0,0);
  TH1F *hChi2pDof = new TH1F ("hChi2pDof","#chi^{2}/dof ",50,0,0);
  TH1F *hNdof = new TH1F ("hNdof","NDOF",50,0,0);
  TH1F *hRsd = new TH1F ("hRsd","Cumulative #DeltaT",20,0.0,6.0);


  TH1F *hPhiRes = new TH1F("hPhiRes","sin(#theta) #times #sigma_{#phi}",30,0,0);
  TH1F *hThetaRes = new TH1F("hThetaRes","#sigma_{#theta}",30,0,0);
  TH1F *hGeomRes = new TH1F("hGeomRes","Geometry resolution",30,0,0);
  hGeomRes->GetXaxis()->SetTitle ("#sqrt{ #sigma_{theta}^{2} + (sin(#theta) #times #sigma_{phi})^{2} }, [degree]");
    

  TH2F *hThetaResVsTheta;
  TProfile *pThetaResVsTheta;


  hThetaResVsTheta = new TH2F("hThetaResVsTheta","#theta resolution",15,-0.25,92.5,50,0.0,5.0);
  hThetaResVsTheta->GetXaxis()->SetTitle("#theta, [degree]");
  hThetaResVsTheta ->GetYaxis()->SetTitle("#sigma_{#theta}, [degree]");
  pThetaResVsTheta = new TProfile("pThetaResVsTheta","#theta resolution",15,-0.25,92.5,0.0,5.0,"S");
  TH2F *hPhiResVsTheta;
  TProfile *pPhiResVsTheta;
    
  hPhiResVsTheta = new TH2F("hPhiResVsTheta","#phi resolution",15,-0.25,92.5,50,0.0,5.0);
  hPhiResVsTheta->GetXaxis()->SetTitle("#theta, [degree]");
  hPhiResVsTheta ->GetYaxis()->SetTitle("sin(#theta) #times #sigma_{phi}, [degree]");
  pPhiResVsTheta = new TProfile("pPhiResVsTheta","#phi.dehs resolution",15,-0.25,92.5,0.0,5.0,"S");

  TH2F *hThetaResVsN;
  TProfile *pThetaResVsN;

    
  hThetaResVsN = new TH2F("hThetaResVsN","#theta resolution",33,-0.5,32.5,50,0.0,5.0);
  hThetaResVsN->GetXaxis()->SetTitle("Number of counters, [degree]");
  hThetaResVsN ->GetYaxis()->SetTitle("#sigma_{#theta}, [degree]");
  pThetaResVsN = new TProfile("pThetaResVsN","#theta resolution",33,-0.5,32.5,0.0,5.0,"S");

  TH2F *hPhiResVsN;
  TProfile *pPhiResVsN;
    
  hPhiResVsN = new TH2F("hPhiResVsN","#phi resolution",33,-0.5,32.5,50,0.0,5.0);
  hPhiResVsN->GetXaxis()->SetTitle("Number of counters, [degree]");
  hPhiResVsN ->GetYaxis()->SetTitle("sin(#theta) #times #sigma_{#phi}, [degree]");
  pPhiResVsN = new TProfile("pPhiResVsN","#phi resolution",33,-0.5,32.5,0.0,5.0,"S");

  hRsd->GetXaxis()->SetTitle("Distance from core in ground plane, [1200m]");
  hRsd->GetYaxis()->SetTitle("#DeltaT, [1200m]");


    



  // Histograms for LDF fit
  TH1F *hLdfChi2 = new TH1F ("hLdfChi2","LDF FIT #chi^{2}/dof",50,0,0);
  TH1F *hLdfNdof = new TH1F ("hLdfNdof","LDF FIT ndof",50,0,0);
    
    
  TH1F *hS600 = new TH1F("hS600","S600_{#theta}",50,0,0);
  TH1F *hS600_0 = new TH1F("hS600_0","S600_{0}",50,0,0);
  TH1F *hEnergy = new TH1F ("hEnergy","Energies",40,17.0,21.0);
  // TH1F *hEnergy = new TH1F ("hEnergy","Energies",100,0.,0.);
  //40,17.0,21.0);
  hEnergy->GetXaxis()->SetTitle("log_{10}(E in eV)");

  TH1F *hEnergyRes = new TH1F("hEnergyRes","E_{Reconstructed}/E_{Thrown} - 1)",100,-5.0,5.0);
    

  TH2F *hNcVsEn = new TH2F("hNcVsEn","Number of counters vs Energy",21,18.05,20.15,30,0.5,30.5);
  hNcVsEn->GetXaxis()->SetTitle("log_{10}(E in eV)");
  hNcVsEn->GetYaxis()->SetTitle("Number of counters");

  TProfile *pNcVsEn = new TProfile("pNcVsEn","Number of counters vs Energy",21,18.05,20.15,0.0,30.0,"S");    
  pNcVsEn->GetXaxis()->SetTitle("log_{10}(E in eV)");
  pNcVsEn->GetYaxis()->SetTitle("Number of counters");

    
  TH1F *hLdfDchi2 = new TH1F("hLdfDchi2","#chi^{2}_{Fixed Core} - #chi^{2}_{Unfixed Core}",
			     100,0,0);
  

  // Histograms for just the lateral profile
    
  TProfile *pRhoVsS[3];
  TH1F     *hSigmaRhoVsS[3];
    
  TH1F *hSlope = new TH1F ("hSlope","Slope parameter",20,0.0,0.0);

  for (Int_t i = 0; i < 3; i++) 
    {
      TString hName;
      TString hTitle;
      Double_t sectheta1, sectheta2;
      sectheta1 = 1.0 + 1.0 * (Double_t) i;
      sectheta2 = 1.0 + 1.0 * ((Double_t)i + 1.0);	
      hName.Form("pRhoVsS%d", i);
      hTitle.Form("#rho vs S, sec(#theta) = [%.1f,%.1f)",sectheta1,sectheta2);
      pRhoVsS[i] = new TProfile (hName,hTitle,10,0.0,5.0,0.0,1e3,"S");
      pRhoVsS[i]->GetXaxis()->SetTitle("Lateral Distance, [1200m]");
      pRhoVsS[i]->GetYaxis()->SetTitle("Charge density, [VEM/m^{2}]");
      hName.Form("hSigmaRhoVsS%d", i);
      hTitle.Form("#sigma_{#rho} vs S, sec(#theta) = [%.1f,%.1f)",sectheta1,sectheta2);
      hSigmaRhoVsS[i] = new TH1F (hName,hTitle,10,0.0,5.0);
      hSigmaRhoVsS[i]->GetXaxis()->SetTitle("Lateral Distance, [1200m]");
      hSigmaRhoVsS[i]->GetYaxis()->SetTitle("#sigma_{#rho}, [VEM/m^{2}]");	
    }
    



  // Some misc charge histograms
    
  TH1F *hQu = new TH1F("hQu","Q_{Upstream}",50,0,0);
  TH1F *hQd = new TH1F("hQd","Q_{Downstream}",50,0,0);
  TH1F *hQud = new TH1F("hQud","log_{10}(Q_{Upstream}/Q_{Downstream})",50,0,0);
   
  TH2F *hQudVsTheta = new TH2F("hQudVsTheta","log_{10}(Q_{Upstream}/Q_{Downstream}) vs #theta",
			       45,0.0,90.0,100,-3.0,3.0);

  TProfile *pQudVsTheta = new TProfile("pQudVsTheta","log_{10}(Q_{Upstream}/Q_{Downstream}) vs #theta",
				       9,0.0,90.0,-3.0,3.0,"S");
  
  TGraph *gLtVsU = 0;
  TGraph *gLtVsS = 0;
  
  // For lateral profile studies
  TF1 *fCos=new TF1("fCos","[0]+[1]*cos(x/57.296+[2]/57.296)",-180.0,180.0);
  fCos->SetParName(0,"Add. Const");
  fCos->SetParName(1,"Amplitude");
  fCos->SetParName(2,"Phase");

  // FD - SD histograms
  Double_t FD_Tref[3]      = {0.0,0.0,0.0}; // FD reference second fractions; useful for synchronizing FD plots
  TGraphErrors *gTvsAfd[3] = {0,0,0};       // Time vs Angle (FD Mono)
  TGraphErrors *gTvsAhb[3] = {0,0,0};       // Time vs Angle (FD Hybrid)
  TGraphErrors *gTvsAhb_sd[3] = {0,0,0};    // For SD points
  TF1 *fTvsAfd = new TF1("fTvsAfd","[0]+[1]/299.792458/tan(([2]+x)/2.0/57.296)",-180.0,180.0);
  fTvsAfd->SetParNames("T_{0}","R_{P}","#Psi");
  
  TF1 *fTvsAfd_CLF = new TF1("fTvsAfd_CLF","[0]-[1]/299.792458/tan(([2]+x)/2.0/57.296)",180.0,360.0);
  fTvsAfd_CLF->SetParNames("T_{0}","R_{P}","#Psi");
  
  Double_t sd_sdp_azm;     // Azimuth angle for SD core in FD SDP, [Degree]
  Double_t sd_sdp_alt;     // Altitude angle for SD core above FD SDP, [Degree] 
  Double_t sdcore_fdtime;  // Time when light reaches FD from SD core, [uS]

  TF1 *fTvsAhb = new TF1("fTvsAhb","[0]+[1]/299.792458 * tan((180.0-[2]-x)/2.0/57.296)",-180.0,180.0);
  fTvsAfd->SetParNames("T_{0}","R_{P}","#Psi");
  
  // Position of BR FD site relative to CLF (meters): 17028.099 -12044.217   -12.095
  Double_t br_origin_clf[3] = {17028.099,-12044.217,-12.095};
    
  /* BR FD site to CLF rotation matrix
     0.999994 -0.002173  0.002666
     0.002178  0.999996 -0.001893
     -0.002661  0.001899  0.999995 */
  Double_t br2clf_mat[3][3] = 
    {
      {0.999994,-0.002173,0.002666},
      {0.002178,0.999996,-0.001893},
      {-0.002661,0.001899,0.999995}
    };

    
  // Position of LR FD site relative to CLF (meters) : -18377.572 -9862.693   137.922
  Double_t lr_origin_clf[3] = {-18377.572,-9862.693,137.922};
    
  /* LR FD site to CLF rotation matrix
     0.999993  0.002347 -0.002877
     -0.002351  0.999996 -0.001550
     0.002873  0.001557  0.999995 */
  Double_t lr2clf_mat[3][3] = 
    {
      {0.999993,0.002347,-0.002877},
      {-0.002351,0.999996,-0.001550},
      {0.002873,0.001557,0.999995}
    };
  
  // Position of MD FD site relative to CLF (meters) : -7308.07,19536.12,183.828
  Double_t md_origin_clf[3] = {-7308.07,19536.12,183.828};
  
  // MD Site to CLF rotation matrix
  Double_t md2clf_mat[3][3] = 
    {
      {0.999999,0.000942,-0.001144},
      {-0.000939,0.999995,0.003070},
      {0.001147,-0.003069,0.999995}
    };

  // Names of the FDs
  const char fd_name[3][3] = {"BR","LR","MD"};
  
  // SD origin with respect to CLF, in [1200m] units
  Double_t sd_origin_x_clf = -12.2435;
  Double_t sd_origin_y_clf = -16.4406;

  // For plotting Minuit contour plots
  TGraph *gHbContFDonly = 0; // when SD points are shut off
  TGraph *gHbCont = 0; // full hybrid fit contours
  
  // For plotting the FD
  TGraph *gEDfd[3]      = {0,0,0}; // ED graph (FD)
  TH2F   *hEDfd[3]      = {0,0,0}; // ED hist (FD with time and pulse heights)
  TF1 *fEleVsAzi_sdp[3] = {0,0,0}; // For drawing SDP normal
  
}
