#include "rusdhist_class.h"
#include "tacoortrans.h"
/*
 * rusdhist_class.cpp
 *
 *  Created on: Nov 30, 2009
 *      Author: ivanov
 */
using namespace TMath;
rusdhist_class::rusdhist_class(listOfOpt& passed_opt) : opt(passed_opt)
{
  book_hist(opt.outfile);
}
rusdhist_class::~rusdhist_class()
{
  end_hist();
}

void rusdhist_class::book_hist(const char* rootfilename)
{  
  TString hName;
  TString hTitle;
  Int_t icut;
  Int_t ienergy;
  
#define _book_1d_hist_icut_(_hist_,_hist_title_,_hist_x_title_,_nbinsX_,_xlo_,_xup_) \
  getNameAndTitle(#_hist_,_hist_title_,icut,&hName,&hTitle);		\
  _hist_[icut] = new TH1D(hName,hTitle,_nbinsX_,_xlo_,_xup_);		\
  setXtitle(_hist_[icut],_hist_x_title_);				\
  _hist_[icut]->GetXaxis()->CenterTitle();
  
  rootfile = new TFile(rootfilename, "recreate");
  if (rootfile->IsZombie())
    {
      printErr("Can't start %s", rootfilename);
      exit(2);
    }
  if (opt.tbflag == 1)
    cutName[0] = "T";
  else if (opt.tbflag == 2)
    cutName[0] = "NT";
  else
    cutName[0] = "";
  cutName[1] = "#frac{#chi^{2}}{dof}<4";
  cutName[2] = "bdist>1200m";
  cutName[3] = "ngsd>=5";
  cutName[4] = "#theta<45^{o}";
  cutName[5] = "pderr<5^{o}";
  cutName[6] = "#frac{#sigma_{S800}}{S800}<0.25";
  cutName[7] = "log_{10}(#frac{E}{eV})>18.0";

  // Allocate the histograms

  for (icut = 0; icut < NCUTLEVELS; icut++)
    {

      // Zenith angle
      getNameAndTitle("hTheta", "#theta", icut, &hName, &hTitle);
      hTheta[icut] = new TH1D(hName, hTitle, 18, 0.0, 90.0);
      setXtitle(hTheta[icut], "Zenith Angle, [Degree]");

      // Azimuth angle
      getNameAndTitle("hPhi", "#phi", icut, &hName, &hTitle);
      hPhi[icut] = new TH1D(hName, hTitle, 18, 0.0, 360.0);
      setXtitle(hPhi[icut], "Azimuth Angle, [Degree]");

      // Geometry fit chi2 / ndof
      getNameAndTitle("hGfChi2Pdof", "Geom. #chi2/ndof", icut, &hName, &hTitle);
      hGfChi2Pdof[icut] = new TH1D(hName, hTitle, 100, 0.0, 10.0);
      setXtitle(hGfChi2Pdof[icut], "#chi2/ndof");

      // LDF fit chi2 / ndof
      getNameAndTitle("hLdfChi2Pdof", "LDF #chi2/ndof", icut, &hName, &hTitle);
      hLdfChi2Pdof[icut] = new TH1D(hName, hTitle, 100, 0.0, 10.0);
      setXtitle(hLdfChi2Pdof[icut], "#chi2/ndof");

      // Core X
      getNameAndTitle("hXcore", "Core_{X}", icut, &hName, &hTitle);
      hXcore[icut] = new TH1D(hName, hTitle, 30, 0.0, 30.0);
      setXtitle(hXcore[icut], "xcore, [1200m]");

      // Core Y
      getNameAndTitle("hYcore", "Core_{Y}", icut, &hName, &hTitle);
      hYcore[icut] = new TH1D(hName, hTitle, 30, 0.0, 30.0);
      setXtitle(hYcore[icut], "ycore, [1200m]");

      // S800
      _book_1d_hist_icut_(hS800,"S800","log_{10}[S800/(VEM/m^{2})]",50,-1.0,4.0);
      
      // Energy
      _book_1d_hist_icut_(hEnergy,"Energy","log_{10}(E/EeV)",40,17.0,21.0);

      // Number of good SDs
      _book_1d_hist_icut_(hNgSd,"number of good SDs","number of good SDs", 33, -0.5, 32.5);
      
      // Total charge / event
      _book_1d_hist_icut_(hQtot,"charge/event","log_{10}(charge/[VEM])",60,-1.0,5.0);
      
      // Total charge / event, without the saturated SDs
      _book_1d_hist_icut_(hQtotNoSat,"charge/event w/o saturated SDs","log_{10}(charge/[VEM])",60,-1.0,5.0);
      
      // Charge / SD
      _book_1d_hist_icut_(hQpSd,"charge/SD","log_{10}(charge/[VEM])",60,-1.0,5.0);
      
      // Total charge / SD, without the saturated SDs
      _book_1d_hist_icut_(hQpSdNoSat,"charge/SD, w/o saturated SDs","log_{10}(charge/[VEM])",60,-1.0, 5.0);
      
      // Number of SDs not in cluster
      _book_1d_hist_icut_(hNsdNotClust,"number of SDs not in cluster",
			  "number of SDs not in cluster",51,-0.5,50.5);
      
      // Charge / SD not in cluster
      _book_1d_hist_icut_(hQpSdNotClust,"charge/SD not in cluster","charge [VEM]",100, 0.0, 10.0);
      
      // Pointing direction error
      _book_1d_hist_icut_(hPdErr,"pointing direction error",
			  "#sqrt{[sin(#theta)#times#sigma_{#phi}]^{2}+[#sigma_{#theta}]^{2}} [Degree]",
			  100,0.0,10.0);
      
      // S800 fluctuation
      _book_1d_hist_icut_(hSigmaS800oS800,"#sigma_{S800}/S800","#sigma_{S800}/S800",100,0.0,1.0);
      
      // Anisotropy variables
      _book_1d_hist_icut_(hHa,"Hour Angle","ha [Degree]",36,-180.0,180.0);
      _book_1d_hist_icut_(hSid,"Sidereal Time","t [Degree]",36,0.0,360.0);
      _book_1d_hist_icut_(hRa,"Right Ascension","#alpha [Degree]",36,0.0,360.0);
      _book_1d_hist_icut_(hDec,"Declination","#delta [Degree]",36,-90.0,90.0);
      _book_1d_hist_icut_(hL,"Galactic Longitude","l [Degree]",36,0.0,360.0);
      _book_1d_hist_icut_(hB,"Galactic Latitude","b [Degree]",36,-90.0,90.0);
      _book_1d_hist_icut_(hSgl,"Super Galactic Longitude","sgl [Degree]",36,0.0,360.0);
      _book_1d_hist_icut_(hSgb,"Super Galactic Latitude","sgb [Degree]",36,-90.0,90.0);
      
      // Number of good SDs vs Energy
      getNameAndTitle("pNgSdVsEn", "number of good SDs", icut, &hName, &hTitle);
      pNgSdVsEn[icut] = new TProfile(hName, hTitle, 20, 17.0, 21.0, "S");
      pNgSdVsEn[icut]->GetXaxis()->SetTitle("log_{10}(E/eV)");
      pNgSdVsEn[icut]->GetYaxis()->SetTitle("Number of good SDs");

      // Number of SDs not in cluster vs Energy
      getNameAndTitle("pNsdNotClustVsEn", "number of SDs not in cluster", icut, &hName, &hTitle);
      pNsdNotClustVsEn[icut] = new TProfile(hName, hTitle, 20, 17.0, 21.0, "S");
      pNsdNotClustVsEn[icut]->GetXaxis()->SetTitle("log_{10}(E/eV)");
      pNsdNotClustVsEn[icut]->GetYaxis()->SetTitle("Number of SDs not in cluster");

      // Total charge / event vs energy
      getNameAndTitle("pQtotVsEn", "charge/event", icut, &hName, &hTitle);
      pQtotVsEn[icut] = new TProfile(hName, hTitle, 20, 17.0, 21.0, "S");
      pQtotVsEn[icut]->GetXaxis()->SetTitle("log_{10}(E/eV)");
      pQtotVsEn[icut]->GetYaxis()->SetTitle("log_{10}(charge/[VEM])");

      // Total charge / event, vs energy without the saturated SDs
      getNameAndTitle("pQtotNoSatVsEn", "charge/event w/o saturated SDs", icut, &hName, &hTitle);
      pQtotNoSatVsEn[icut] = new TProfile(hName, hTitle, 20, 17.0, 21.0, "S");
      pQtotNoSatVsEn[icut]->GetXaxis()->SetTitle("log_{10}(E/eV)");
      pQtotNoSatVsEn[icut]->GetYaxis()->SetTitle("log_{10}(charge/[VEM])");

      ////////// Resolution Histograms (MC only) ////////////

      // Zenith angle resolution
      getNameAndTitle("hThetaRes", "Zenith angle resolution", icut, &hName, &hTitle);
      hThetaRes[icut] = new TH1D(hName, hTitle, 100, -10.0, 10.0);
      setXtitle(hThetaRes[icut], "#theta_{Rec} + 0.5 - #theta_{Thr}, [Degree]");

      // Azimuthal angle resolution
      getNameAndTitle("hPhiRes", "Azimuthal resolution", icut, &hName, &hTitle);
      hPhiRes[icut] = new TH1D(hName, hTitle, 100, -10.0, 10.0);
      setXtitle(hPhiRes[icut], "sin(#theta_{Rec}) #times (#phi_{Rec}-#phi_{Thr}), [Degree]");

      // Core X resolution
      getNameAndTitle("hXcoreRes", "Core_{X} resolution", icut, &hName, &hTitle);
      hXcoreRes[icut] = new TH1D(hName, hTitle, 100, -3.0, 3.0);
      setXtitle(hXcoreRes[icut], "xcore_{Rec}-xcore_{Thr}, [1200m]");

      // Core Y resolution
      getNameAndTitle("hYcoreRes", "Core_{Y} resolution", icut, &hName, &hTitle);
      hYcoreRes[icut] = new TH1D(hName, hTitle, 100, -3.0, 3.0);
      setXtitle(hYcoreRes[icut], "ycore_{Rec}-ycore_{Thr}, [1200m]");

      // Energy resolution (ratio - 1)
      getNameAndTitle("hEnergyResRat", "Energy resolution (ratio)", icut, &hName, &hTitle);
      hEnergyResRat[icut] = new TH1D(hName, hTitle, 100, -5.0, 5.0);
      setXtitle(hEnergyResRat[icut], "E_{Rec}/E_{Thr} - 1");

      // Energy resolution (log of ratio)
      getNameAndTitle("hEnergyResLog", "Energy resolution, log(ratio)", icut, &hName, &hTitle);
      hEnergyResLog[icut] = new TH1D(hName, hTitle, 100, -1.0, 1.0);
      setXtitle(hEnergyResLog[icut], "log_{10}(E_{Rec}/E_{Thr})");

      // log reconstructed vs log thrown scatter plot
      getNameAndTitle("hEnergyRes2D", "Energy resolution, (scatter)", icut, &hName, &hTitle);
      hEnergyRes2D[icut] = new TH2D(hName, hTitle, 40, 17.0, 21.0, 40, 17.0, 21.0);
      hEnergyRes2D[icut]->GetXaxis()->SetTitle("log_{10}(E_{Thr}/EeV)");
      hEnergyRes2D[icut]->GetYaxis()->SetTitle("log_{10}(E_{Rec}/EeV)");

      // (ratio - 1) versus the log of the thrown energy
      getNameAndTitle("pEnergyRes", "Energy resolution, (profile)", icut, &hName, &hTitle);
      pEnergyRes[icut] = new TProfile(hName, hTitle, 15, 18.0, 21.0, "S");
      pEnergyRes[icut]->GetXaxis()->SetTitle("log_{10}(E_{Thr}/EeV)");
      pEnergyRes[icut]->GetYaxis()->SetTitle("E_{Rec}/E_{Thr} - 1");

      // S800 vs sec(theta) profile plots

      for (ienergy = 0; ienergy < NLOG10EBINS; ienergy++)
        {
          getNameAndTitle("pS800vsSecTheta", "S800 vs sec(#theta)", icut, &hName, &hTitle);
          hName += "_";
          hName += ienergy;
          pS800vsSecTheta[icut][ienergy] = new TProfile(hName, hTitle, 4, 1.0, 1.4);
          pS800vsSecTheta[icut][ienergy]->GetXaxis()->SetTitle("sec(#theta)");
          pS800vsSecTheta[icut][ienergy]->GetYaxis()->SetTitle("<S800/[VEM]>");
        }
    }

  // Histograms that do not depend on cuts

  hFadcPmip[0] = new TH1D("hFadcPmip0", "FADC/MIP (lower)", 100, 0.0, 100.0);
  setXtitle(hFadcPmip[0], "1MIP FADC counts above the pedestal");

  hFadcPmip[1] = new TH1D("hFadcPmip1", "FADC/MIP (upper)", 100, 0.0, 100.0);
  setXtitle(hFadcPmip[1], "1MIP FADC counts above the pedestal");

  hFwhmMip[0] = new TH1D("hFwhmMip0", "1MIP FWHM (lower)", 30, 0.0, 3.0);
  setXtitle(hFwhmMip[0], "1MIP FWHM, [MIP]");

  hFwhmMip[1] = new TH1D("hFwhmMip1", "1MIP FWHM (upper)", 30, 0.0, 3.0);
  setXtitle(hFwhmMip[1], "1MIP FWHM, [MIP]");

  hPchPed[0] = new TH1D("hPchPed0", "Peak channel of pedestal hist, (lower)", 100, 0.0, 100.0);
  setXtitle(hPchPed[0], "FADC counts");

  hPchPed[1] = new TH1D("hPchPed1", "Peak channel of pedestal hist, (upper)", 100, 0.0, 100.0);
  setXtitle(hPchPed[1], "FADC counts");

  hFwhmPed[0] = new TH1D("hFwhmPed0", "PED FWHM (lower)", 15, 0.0, 15.0);
  setXtitle(hFwhmMip[0], "PED FWHM, [FADC counts]");

  hFwhmPed[1] = new TH1D("hFwhmPed1", "PED FWHM (upper)", 15, 0.0, 15.0);
  setXtitle(hFwhmMip[1], "PED FWHM, [FADC counts]");

#undef _book_1d_hist_icut_
  
}

void rusdhist_class::end_hist()
{
  rootfile->Write();
  rootfile->Close();
}

void rusdhist_class::printErr(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "rusdhist_class: %s\n", mess);
}

void rusdhist_class::getNameAndTitle(const char *hNameBare, const char *hTitleBare, Int_t icut, TString *hName,
    TString *hTitle)
{
  Int_t i;
  if (icut < 0 || icut >= NCUTLEVELS)
    {
      printErr("Internal error: icut must be in 0 - %d range", (NCUTLEVELS - 1));
      exit(2);
    }
  (*hName) = hNameBare;
  (*hName) += icut;
  (*hTitle) = hTitleBare;
  for (i = 0; i <= icut; i++)
    {
      if ((i == 0) && cutName[i].Length())
        (*hTitle) += ": ";
      if (cutName[i].Length() && (i != 0))
        (*hTitle) += ", ";
      (*hTitle) += cutName[i];
    }
}
void rusdhist_class::setXtitle(TH1 *h, const char *xtitle)
{
  h->GetXaxis()->SetTitle(xtitle);
}
void rusdhist_class::Fill(bool have_mc_banks)
{
  Int_t isd;
  Int_t irufptn, irusdraw;
  Int_t icut, ienergy;
  Double_t log10ethr, log10en;
  Int_t cut_level; // cut level for given event
  bool pass_cut; // accumulating cut levels
  Double_t gfchi2pdof, ldfchi2pdof, tdist, bdist, pderr, sectheta, ds800os800, energy;
  Int_t year,month,day,hour,minute,second,usec;
  Double_t second_since_midnight;
  Double_t theta,phi,jday,ha,lmst,ra,dec,l,b,sgl,sgb;
  Double_t qtot, qtot_notsat;
  Double_t dphi;
  Int_t n_notinclust;
  Double_t w; // weight
  
  if(opt.e3wopt)
    {
      if (have_mc_banks)
	{
	  // re-scaled MC energy is used in aperture calculations,
	  // so it must follow the correct spectral power laws
	  w = ankle_weight_for_e3(rusdmc_.energy / 1.27);
	}
      else
	{
	  fprintf(stderr,"^^^^^^ WARNING: ankle weight to E^-3 MC option used but no MC banks present. weight=1.\n");
	  w = 1.0;
	}
    }
  else
    w = 1.0;

  //////////////// DATA SET CUTS (OPTIONAL) ////////////////////////
  if (rusdraw_.yymmdd < opt.yymmdd_start || rusdraw_.yymmdd > opt.yymmdd_stop)
    return;
  
  ////// TRIGGER BACKUP CUT (OPTIONAL) ////////////
  
  /// Applying the trigger backup cut in such a way that
  // 1) event must have triggered in trigger backup program
  // 2) did not have to lower the pedestals in order to trigger the event
  if ((opt.tbflag == 1) && (sdtrgbk_.igevent < 2))
    return;
  
  // to histogram the events that don't pass the trigger backup
  // cut ( including those for which had to lower the pedestals)
  if ((opt.tbflag == 2) && (sdtrgbk_.igevent >= 2))
    return;
  
  // calculate chi2/ndof

  // geometry fit
  gfchi2pdof = (rusdgeom_.ndof[2] <= 0 ? rusdgeom_.chi2[2] : rusdgeom_.chi2[2] / (double) rusdgeom_.ndof[2]);

  // LDF fit
  ldfchi2pdof = (rufldf_.ndof[0] <= 0 ? rufldf_.chi2[0] : rufldf_.chi2[0] / (double) rufldf_.ndof[0]);

  // calculate the border distances
  tdist = rufldf_.tdist;
  bdist = rufldf_.bdist;

  // calculate the pointing direction error
  pderr = sqrt(sin(rusdgeom_.theta[2] * DegToRad()) * sin(rusdgeom_.theta[2] * DegToRad()) * rusdgeom_.dphi[2]
      * rusdgeom_.dphi[2] + rusdgeom_.dtheta[2] * rusdgeom_.dtheta[2]);

  // calculate sigma_s800 / s800
  ds800os800 = rufldf_.dsc[0] / rufldf_.sc[0];

  // calculate the anisotropy variables
  year   = rusdraw_.yymmdd/10000+2000;
  month  = (rusdraw_.yymmdd % 10000)/100;
  day    = (rusdraw_.yymmdd % 100);  
  hour   = (rusdraw_.hhmmss / 10000);
  minute = (rusdraw_.hhmmss % 10000)/100;
  second = (rusdraw_.hhmmss % 100);
  usec   = rusdraw_.usec;
  second_since_midnight = (double)(hour*3600+minute*60+second)+
    ((double)usec)/1e6;
  jday  = tacoortrans::utc_to_jday(year,month,day,second_since_midnight);
  lmst  = RadToDeg() * (tacoortrans::jday_to_LMST(jday,tacoortrans_CLF_Longitude*DegToRad()));
  
  theta = rusdgeom_.theta[2]+0.5;
  phi   = rusdgeom_.phi[2]+180.0;
  if(phi > 360.0) phi -= 360.0;
  
  ha = RadToDeg() * tacoortrans::get_ha(DegToRad()*theta,DegToRad()*phi,
					DegToRad()*tacoortrans_CLF_Latitude);
  ra = lmst - ha;
  while(ra >= 360.0) ra -= 360.0;
  while(ra < 0.0) ra += 360.0;
  
  dec = RadToDeg() * tacoortrans::get_dec(DegToRad()*theta,DegToRad()*phi,
					  DegToRad()*tacoortrans_CLF_Latitude);  
  // galactic coordinates         
  l = RadToDeg() * tacoortrans::gall(DegToRad()*ra,DegToRad()*dec);
  b = RadToDeg() * tacoortrans::galb(DegToRad()*ra,DegToRad()*dec);
  
  // supergalactic coordinates
  sgl = RadToDeg() * tacoortrans::sgall(DegToRad()*ra,DegToRad()*dec);
  sgb = RadToDeg() * tacoortrans::sgalb(DegToRad()*ra,DegToRad()*dec);
  
  
  sectheta = 1.0 / Cos(rusdgeom_.theta[2] * DegToRad());

  // Latest MC energy (Jun 2010).  Nomralization to FD energy scale not done
  energy = sden_jun2010(rufldf_.s800[0],rusdgeom_.theta[2]);
  
  log10en = 18.0 + Log10(energy);

  if (have_mc_banks)
    {
      log10ethr = 18.0 + Log10(rusdmc_.energy);
      ienergy = (Int_t) Floor((log10ethr - LOG10EMIN) / (LOG10EMAX - LOG10EMIN) * (Double_t) NLOG10EBINS);
    }
  else
    {
      log10ethr = 0.0;
      ienergy = 0;
    }
  /////////////// MINIMUM EVENT CUTS (BELOW) //////////////////////////

  // Making sure that we're using only Data Set 1
  
  if (rufptn_.nstclust < 3)
    return;

  if (gfchi2pdof > 10.0)
    return;

  if (ldfchi2pdof > 10.0)
    return;

  if (rusdgeom_.theta[2] > 45.0)
    return;

  if (pderr > 10.0)
    return;

  /////////////// MINIMUM EVENT CUTS (ABOVE) //////////////////////////


  //////////////// DETERMINE THE CUT LEVEL FOR THE EVENT (BELOW) ///////////////////

  cut_level = 0;
  pass_cut = true;

  // Chi2 / ndof cut
  pass_cut = pass_cut && ((gfchi2pdof < 4.0) && (ldfchi2pdof < 4.0));
  if (pass_cut)
    cut_level++;

  // Border cut ( T-shape border cut is necessary for DS1 only)  
  if (rusdraw_.yymmdd <= 81110)
    {
      pass_cut = pass_cut && ((bdist > 1.0) && (tdist > 1.0));
      if (pass_cut)
	cut_level++;
    }
  else
    {
      pass_cut = pass_cut && (bdist > 1.0);
      if (pass_cut)
	cut_level++;
    }
  // Number of good SDs cut
  pass_cut = pass_cut && (rufptn_.nstclust >= 5);
  if (pass_cut)
    cut_level++;

  // Zenith angle cut
  pass_cut = pass_cut && (rusdgeom_.theta[2] < 45.0);
  if (pass_cut)
    cut_level++;

  // Pointing direction error cut
  pass_cut = pass_cut && (pderr < 5.0);
  if (pass_cut)
    cut_level++;

  // sigma_s800 / s800 cut
  pass_cut = pass_cut && (ds800os800 < 0.25);
  if (pass_cut)
    cut_level++;

  // Energy cut
  pass_cut = pass_cut && ((18.0 + Log10(energy)) > 18.0);
  if (pass_cut)
    cut_level++;

  //////////////// DETERMINE THE CUT LEVEL FOR THE EVENT (ABOVE) ///////////////////

  // Histograms over SDs
  qtot = 0.0;
  qtot_notsat = 0.0;
  n_notinclust = 0;
  for (isd = 0; isd < rusdgeom_.nsds; isd++)
    {
      if (rusdgeom_.igsd[isd] < 1)
        continue;

      // Calibration information
      irufptn = rusdgeom_.irufptn[isd][0];
      irusdraw = rufptn_.wfindex[irufptn];
      hFadcPmip[0]->Fill(rusdraw_.mip[irusdraw][0],w);
      hFadcPmip[1]->Fill(rusdraw_.mip[irusdraw][1],w);
      hFwhmMip[0]->Fill(((double) (rusdraw_.rhpchmip[irusdraw][0] - rusdraw_.lhpchmip[irusdraw][0]))
			/ rusdraw_.mip[irusdraw][0],w);
      hFwhmMip[1]->Fill(((double) (rusdraw_.rhpchmip[irusdraw][1] - rusdraw_.lhpchmip[irusdraw][1]))
			/ rusdraw_.mip[irusdraw][1],w);
      hPchPed[0]->Fill(rusdraw_.pchped[irusdraw][0],w);
      hPchPed[1]->Fill(rusdraw_.pchped[irusdraw][1],w);
      hFwhmPed[0]->Fill(rusdraw_.rhpchped[irusdraw][0] - rusdraw_.lhpchped[irusdraw][0],w);
      hFwhmPed[1]->Fill(rusdraw_.rhpchped[irusdraw][1] - rusdraw_.lhpchped[irusdraw][1],w);

      if (rusdgeom_.igsd[isd] == 1)
        {
          n_notinclust++;
          for (icut = 0; icut <= cut_level; icut++)
            hQpSdNotClust[icut]->Fill(rusdgeom_.pulsa[isd],w);
        }
      if (rusdgeom_.igsd[isd] > 1)
        {
          for (icut = 0; icut <= cut_level; icut++)
            hQpSd[icut]->Fill(Log10(rusdgeom_.pulsa[isd]),w);
          qtot += rusdgeom_.pulsa[isd];
        }
      if ((rusdgeom_.igsd[isd] == 2))
        {
          for (icut = 0; icut <= cut_level; icut++)
            hQpSdNoSat[icut]->Fill(Log10(rusdgeom_.pulsa[isd]),w);
          qtot_notsat += rusdgeom_.pulsa[isd];
        }

    }

  for (icut = 0; icut <= cut_level; icut++)
    {

      hTheta[icut]->Fill(rusdgeom_.theta[2],w);
      hPhi[icut]->Fill(rusdgeom_.phi[2],w);
      hGfChi2Pdof[icut]->Fill(gfchi2pdof,w);
      hLdfChi2Pdof[icut]->Fill(ldfchi2pdof,w);
      hXcore[icut]->Fill(rufldf_.xcore[0],w);
      hYcore[icut]->Fill(rufldf_.ycore[0],w);
      hS800[icut]->Fill(Log10(rufldf_.s800[0]),w);
      hEnergy[icut]->Fill(18.0 + Log10(energy),w);
      hNgSd[icut]->Fill(rufptn_.nstclust,w);
      hQtot[icut]->Fill(Log10(qtot),w);
      hQtotNoSat[icut]->Fill(Log10(qtot_notsat),w);
      hNsdNotClust[icut]->Fill(n_notinclust,w);
      hPdErr[icut]->Fill(pderr,w);
      hSigmaS800oS800[icut]->Fill(ds800os800,w);
      hHa[icut]->Fill(ha);
      hSid[icut]->Fill(lmst);
      hRa[icut]->Fill(ra);
      hDec[icut]->Fill(dec);
      hL[icut]->Fill(l);
      hB[icut]->Fill(b);
      hSgl[icut]->Fill(sgl);
      hSgb[icut]->Fill(sgb);

      pNgSdVsEn[icut]->Fill(log10en, rufptn_.nstclust,w);
      pNsdNotClustVsEn[icut]->Fill(log10en, n_notinclust,w);
      pQtotVsEn[icut]->Fill(log10en, Log10(qtot),w);
      pQtotNoSatVsEn[icut]->Fill(log10en, Log10(qtot_notsat),w);

      // If have MC banks, then fill the resolution histograms and S800 vs sec(theta)
      // profiles
      if (have_mc_banks)
        {
          hThetaRes[icut]->Fill(rusdgeom_.theta[2] + 0.5 - rusdmc_.theta * RadToDeg(),w);
          dphi = rusdgeom_.phi[2] - rusdmc_.phi * RadToDeg();
          if (dphi >= 180.0)
            dphi -= 360.0;
          if (dphi < -180.0)
            dphi += 360.0;
          hPhiRes[icut]->Fill(sin(rusdmc_.theta) * dphi,w);
          hXcoreRes[icut]->Fill(rufldf_.xcore[0] - rusdmc1_.xcore,w);
          hYcoreRes[icut]->Fill(rufldf_.ycore[0] - rusdmc1_.ycore,w);
          hEnergyResRat[icut]->Fill(energy / rusdmc_.energy - 1.0,w);
          hEnergyResLog[icut]->Fill(Log10(energy / rusdmc_.energy),w);
          hEnergyRes2D[icut]->Fill(18.0 + Log10(rusdmc_.energy), 18.0 + Log10(energy),w);
          pEnergyRes[icut]->Fill(18.0 + Log10(rusdmc_.energy), energy / rusdmc_.energy - 1.0,w);

          // Fill S800 vs sec(theta) profiles
          if ((ienergy >= 0) && (ienergy < NLOG10EBINS) && (sectheta <= 2.0))
            pS800vsSecTheta[icut][ienergy]->Fill(sectheta, rufldf_.s800[0],w);
        }

    }

}

double rusdhist_class::ankle_weight_for_e3(double energyEeV)
{
  
  // ankle position given by HiRes
  const double E_ankleHR = 4.46683592151; // energy at the angle in EeV
  const double c0      = 1.34896288259; // E_ankleHR ^ 0.2
  const double c1      = 0.7413102413 ; // E_angleHR ^ -0.2
  // ankle position wanted by TASD
  const double E_ankleSD = 5.6234132519; // energy at the angle in EeV
  const double c2      = 1.4125375446; // E_ankleSD ^ 0.2
  const double c3      = 0.7079457844; // E_angleSD ^ -0.2

  if (opt.e3wopt == 1)
    return ((energyEeV<=E_ankleHR) ? (c0*Power(energyEeV,-0.2)) : c1*Power(energyEeV,0.2));
  else if (opt.e3wopt==2)
    return ((energyEeV<=E_ankleSD) ? (c2*Power(energyEeV,-0.2)) : c3*Power(energyEeV,0.2));
  else
    return 1.0;
}
