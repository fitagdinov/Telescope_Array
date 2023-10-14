{

  gROOT->Macro("preset.C");
  TString hst_file="hst.root";
  
  /////////////////////// OPEN THE HISTOGRAM FILES //////////////////////
  
  TFile *hstfl;
  hstfl = new TFile("hst.root");
  if (hstfl->IsZombie())
    return;
  
  ///////////////////////////////////////////////////////////////////////

  ////////////////////// PREPARE THE CANVAS ///////////////////

  TCanvas *c1 = new TCanvas("c1","scratch",0,10,730,450);
  c1->SetGrid(); c1->cd();
  c1->SetTickx(); c1->SetTicky();

  ///////////////////////////////////////////////////////////////
  

  ///////////////// LOAD THE MACROS ////////////////////////////
  gROOT->LoadMacro("hstplot.C");
}
