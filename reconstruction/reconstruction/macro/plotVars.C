#define nsecIn1200m 4.0027691424e3 // nS in 1200 meters

void whatBranches()
{
  p1.pass1tree->Print("toponly"); 
}


// LDF function
// r     :  Perpendicular distance from shower axis, m
// theta :  Zenith angle, degree
static Double_t ldffun(Double_t r, Double_t theta)
  {
    Double_t r0;    // Moliere radius
    Double_t alpha; // Constant slope parameter
    Double_t beta;  // Another constant slope parameter
    Double_t eta;   // Zenith angle dependent slope parameter
    Double_t rsc;   // Scaling factor for r in quadratic term in power
    
    r0    =  91.6;
    alpha =  1.2;
    beta  =  0.6;
    eta   =  3.97-1.79*(1.0/Cos(DegToRad()*theta)-1.0);
    rsc   = 1000.0;
    
    return  
      Power(r/r0, -alpha) * 
      Power((1.0+r/r0), -(eta-alpha)) * 
      Power((1.0+ r*r/rsc/rsc), -beta);
  }


// Modified Linsley Td,Ts in [nS]
// Rho is in [VEM/m^2], R is perpendicular dist. from shower axis, in [m]
static void ltdts(Double_t Rho, Double_t R, Double_t theta, Double_t *td, Double_t *ts)
  {
    Double_t a;

    if (theta < 25.0)
      {
        a = 3.3836 - 0.01848 * theta;
      }
    else if ((theta >= 25.0) && (theta <35.0))
      {
        a=(0.6511268210e-4*(theta-.2614963683))*(theta*theta-134.7902422*theta+4558.524091);
      }
    else
      {
        a = exp( -3.2e-2 * theta + 2.0);
      }         
    //     Contained, >=4 counters, DCHI2 = 4.0
    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    (*ts) = 0.70 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
  }


void mkLcurve()
{    
  // Variables plotted
  Double_t u[10000],t[10000];
  
  // Shower fit variables
  Double_t theta;
  Double_t phi;
  Double_t t0;
  Double_t sc;
  
  // aux. variables
  Int_t iu,nbu;
  Double_t ulo,uval,uup,bsu;
  Double_t s,rho,tp,td,ts;
  
  // Get the shower fit parameters
  theta=DegToRad() * p1.rusdgeom->theta[1];
  phi=DegToRad() * p1.rusdgeom->phi[1];
  t0=p1.rusdgeom->t0[1];
  sc=p1.rufldf->sc[0];
  
  // Prepare a 'Time vs U' graph
  ulo=-5.0; uup=5.0; nbu=100;
  bsu = (uup-ulo)/(Double_t)nbu;
  for(iu=0;iu<nbu;iu++)
    {
      uval=ulo+bsu*((Double_t)iu + 0.5);
      tp=uval*sin(theta);
      s=fabs(uval*cos(theta));
      rho=sc*ldffun(s*1.2e3, theta*RadToDeg());
      ltdts(rho,s*1.2e3,theta*RadToDeg(),&td,&ts);
      td/=nsecIn1200m; ts/=nsecIn1200m;
      u[iu] = uval;
      t[iu] = t0+tp+td;
    }
  if(gLtVsU)
    delete gLtVsU;
  gLtVsU = new TGraph(nbu,u,t);
  gLtVsU->SetMarkerStyle(20);
  gLtVsU->SetMarkerSize(0.5);
}


void plotFitResults()
{
  
  TString xtitle;
  
  c3->SetBorderSize(0);
  c3->cd(1);
  gPad->Clear();
  gPad->SetTickx();
  gPad->SetTicky();
  gPad->SetGridx(0);
  gPad->SetGridy(0);
  gPad->Modified();
  gPad->Update();
  mkLcurve();
  sdgeom->gTvsU->SetTitle("SD Time Fit");
  xtitle="Distance along shower axis on the ground, [1200m]";
  sdgeom->gTvsU->GetXaxis()->SetTitle(xtitle);
  sdgeom->gTvsU->GetXaxis()->SetLabelSize(0.05);
  sdgeom->gTvsU->GetXaxis()->SetTitleSize(0.05);
  sdgeom->gTvsU->GetYaxis()->SetLabelSize(0.05);
  sdgeom->gTvsU->GetYaxis()->SetTitleSize(0.05);
  sdgeom->gTvsU->GetXaxis()->SetTitleOffset(0.9);
  sdgeom->gTvsU->GetYaxis()->SetTitleOffset(0.8);
  sdgeom->gTvsU->SetMarkerSize(1.0);
  sdgeom->gTvsU->Draw("a,e1p");
  gLtVsU->SetLineColor(2);
  gLtVsU->SetLineWidth(3);
  gLtVsU->Draw("l,same");

  
  c3->cd(2);
  gPad->Clear();
  gPad->SetTickx();
  gPad->SetTicky();
  gPad->SetGridx(0);
  gPad->SetGridy(0);
  gPad->SetLogx();
  gPad->SetLogy();
  gPad->Modified();
  gPad->Update();
  ldf->gqvsr->SetTitle("SD LDF Fit");
  xtitle="Perpendicular distance from shower axis, [1200m]";
  ldf->gqvsr->GetXaxis()->SetTitle(xtitle);
  ldf->gqvsr->GetXaxis()->SetLabelSize(0.05);
  ldf->gqvsr->GetXaxis()->SetTitleSize(0.05);
  ldf->gqvsr->GetYaxis()->SetLabelSize(0.05);
  ldf->gqvsr->GetYaxis()->SetTitleSize(0.05);
  ldf->gqvsr->GetYaxis()->SetTitleOffset(0.8);
  ldf->gqvsr->SetMarkerSize(1.0);
  ldf->gqvsr->Draw("A,e1p");
  ldf->ldfFun->SetLineColor(2);
  ldf->ldfFun->SetLineWidth(3);
  ldf->ldfFun->Draw("same");
  c1->cd();
}
