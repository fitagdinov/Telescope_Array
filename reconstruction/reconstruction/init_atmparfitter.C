{
  TFile *gdasFile = new TFile("gdas.root");
  if(gdasFile->IsZombie())
    {
      fprintf(stderr,"error: gdas.root file not found in current directory\n");
      exit(2);
    }  
  TTree *taTree = (TTree *)gdasFile->Get("taTree");
  if(!taTree)
    {
      fprintf(stderr,"Error: taTree not found in %s\n",gdasFile->GetName());
      exit(2);
    }
  taTree->SetMarkerStyle(21);
  taTree->SetLineWidth(3);
  gdas_class   *gdas   = new gdas_class;
  atmpar_class *atmpar = new atmpar_class;
  if(taTree->GetBranch("atmpar"))
    taTree->SetBranchAddress("gdas",&gdas);
  else
    {
      fprintf(stderr,"error: no gdas branch in %s!\n",taTree->GetName());
      exit(2);
    }
  Bool_t have_atmpar = false;
  if(taTree->GetBranch("atmpar"))
    {
      taTree->SetBranchAddress("atmpar",&atmpar);
      have_atmpar = true;
    }
  taTree->GetEntry(0);
  atmparfitter atm;
}
