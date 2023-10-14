void fitAtm(Int_t entry=0)
{
  taTree->GetEntry(entry); 
  atm.loadVariables(gdas); 
  atm.Fit(); 
  atm.GetMoVsHderiv()->Draw("a,e1p");
}
