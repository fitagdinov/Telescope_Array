#include "TFile.h"
#include "TCut.h"
#include "TChain.h"
#include "TMath.h"


// 0 = winter season
// 1 = summer season
Int_t get_season(Int_t yymmdd)
{
  Int_t m = ((yymmdd%10000)/100);
  if(m >= 4 && m <= 9)
    return 1;
  return 0;
}

Int_t get_summer_density_bin(Int_t hhmmss)
{
  if(73000 <= hhmmss && hhmmss <= 163000)
    return 0;
  return 1;
}

TCut low_density_hours="get_summer_density_bin(hhmmss)==0";
TCut high_density_hours="get_summer_density_bin(hhmmss)==1";

void pick_summer_hours_event_root_tree(TCut cut_hour, 
		       const char *infile, 
		       const char* outfile)
{
  TCut cut_season="get_season(yymmdd)==1"; // must be summer
  TCut cut = cut_season + cut_hour;        // low density or high density hours?
  
  TFile *fIn = new TFile(infile);
  if(fIn->IsZombie())
    {
      fprintf(stderr,"error: faild to open %s\n",infile);
      return;
    }
  TFile *fOut = new TFile(outfile,"recreate");
  if(fOut->IsZombie())
    fprintf(stderr,"error: failed to create %s\n",outfile);
  TTree *tIn = (TTree *)fIn->Get("resTree");
  if(!tIn)
    {
      fprintf(stderr,"error: failed to obtain the input tree resTree from %s\n",infile);
      return;
    }
  fOut->cd();
  TTree *tOut = tIn->CopyTree(cut);
  fprintf(stderr,"nevents_read = %ld nevents_written = %ld\n",tIn->GetEntries(),tOut->GetEntries());
  fOut->cd();
  tOut->Write();
  fOut->Close();
  
}

void pick_summer_hours_event_root_tree()
{
  fprintf(stderr,"pick_summer_hours_event_root_tree([low_density_hours or high_density_hours],infile,outfile)\n");
}
