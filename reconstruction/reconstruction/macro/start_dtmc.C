{

  gROOT->Macro("preset.C");
  
  if(!successFlag)
    return;
  
  TString dt_hst_file="dt.hst.root";
  TString mc_hst_file="mc.hst.root";
  
  /////////////////////// OPEN THE HISTOGRAM FILES //////////////////////
  Bool_t fopen_flag;
  TFile *dthistfl, *mchistfl;
  fopen_flag = true;
  dtfl = new TFile(dt_hst_file);
  if (dtfl->IsZombie())
    fopen_flag = false;
  mcfl = new TFile(mc_hst_file);
  if(mcfl->IsZombie())
    fopen_flag = false;
  if (!fopen_flag)
    return;
  ///////////////////////////////////////////////////////////////////////

  ////////////////////// PREPARE THE CANVAS ///////////////////

  TCanvas *c1 = new TCanvas("c1","scratch",0,10,730,450);
  c1->SetGrid(); c1->cd();
  c1->SetTickx(); c1->SetTicky();
  
  TCanvas *cdtmc = new TCanvas("cdtmc","DATA/MC",0,10,730,800);
  cdtmc->Divide(1,2);
  for (int i=1; i <=2; i++)
    {
      cdtmc->cd(i);
      gPad->SetGridx();
      gPad->SetGridy();
      gPad->SetTickx(); 
      gPad->SetTicky();
    }
  ///////////////////////////////////////////////////////////////
  

  ///////////////// LOAD THE MACROS ////////////////////////////
  gROOT->LoadMacro("dtmc_comp.C");
}
