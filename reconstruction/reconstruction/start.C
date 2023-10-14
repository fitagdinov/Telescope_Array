{
  gROOT->SetStyle("Plain");
  gStyle->SetLineWidth(3);
  gStyle->SetOptFit(1);
  gROOT->Macro("load_lib.C");
  gROOT->Macro("init_atmparfitter.C");
  gROOT->LoadMacro("atmfun.C+");
  gROOT->LoadMacro("atmpar_rsd.C");
  gROOT->LoadMacro("atmfit.C");

}
