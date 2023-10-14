#include "prepmc.h"
#include "sddstio.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TMath.h"
#include "TProfile.h"
#include "TString.h"
#include "TMath.h"
#include "sdparamborder.h"


using namespace TMath;

int main(int argc, char **argv)
{
  listOfOpt opt; // to handle the program arguments.
  char *dInFile; // pass1 dst input files
  TString outname;

  // Dummy variables
  double v[2];
  double tdistbr, vbr[2];
  double tdistlr, vlr[2];
  double tdistsk, vsk[2];
  
  integer4 size,wantBanks;
  
  sddstio_class *dstio; // to handle the DST I/O

  // parses cmd line and checks that it's possible to start the RZ file
  if (!opt.getFromCmdLine(argc, argv))
    return -1;

  opt.printOpts();

  // Initialize SD run class
  dstio = new sddstio_class();
  
  size = 100;
  wantBanks = newBankList(size);
  addBankList(wantBanks,RUSDRAW_BANKID);
  addBankList(wantBanks,RUSDMC_BANKID);
  dstio->setWantBanks(wantBanks);
  
  ////////////// SIMPLE ROOT TREE WITH THROWN MC INFORMATION ///////
  TFile *thmcOut;
  TTree *mcthTree;



  // MC variables
  Int_t    yymmdd;      // date
  Int_t    hhmmss;      // time
  Int_t    usec;        // event micro second
  Int_t    nofwf;       // total number of waveforms in the readout
  Int_t    parttype;    // Corsika particle type, proton=14, iron=5626
  Int_t    corecounter; // Counter closest to core, useful in debugging
  Double_t corexyz[3];  // 3D core position in CLF reference frame as given in Ben's MC, [cm]
  Double_t height;      // Height of the first interaction, cm
  Double_t mctheta;     // Zenith    angle, [Degree]
  Double_t mcphi;       // Azimuthal angle, [Degree]
  Double_t mcxcore;     // Core X position in SD coordinates, [1200m]
  Double_t mcycore;     // Core Y position in SD coordinates, [1200m]
  Double_t mcenergy;    // Energy in EeV
  Int_t    mctc;        // Core clock count, as in Ben's MC
  Double_t mcbdist;     // Distances from the SD edge for the thrown cores, [1200m]
  Double_t mctdist;     // Distances from the T-shape boundary for the thrown cores, [1200m]

  // Thrown MC
  outname=(TString)opt.dout;
  if (opt.outpr[0])
    {
      outname += (TString)opt.outpr;
      outname += (TString)".thrown.root";
    }
  else
    {
      outname += (TString)"thrown.root";
    }
  fprintf(stdout, "OUTFILE: %s\n",(char *)outname.Data());
  fflush(stdout);

  thmcOut =
    new TFile(outname,"recreate");
  mcthTree = new TTree("mcthTree","SD variables");


  
  mcthTree->Branch ( "yymmdd",      &yymmdd,      "yymmdd/I"     );
  mcthTree->Branch ( "hhmmss",      &hhmmss,      "hhmmss/I"     );
  mcthTree->Branch ( "usec",        &usec,        "usec/I"       );
  mcthTree->Branch ( "nofwf",       &nofwf,       "nofwf/I"      );
  mcthTree->Branch ( "parttype",    &parttype,    "parttype/I"   );
  mcthTree->Branch ( "corecounter", &corecounter, "corecounter/I");
  mcthTree->Branch ( "corexyz",     corexyz,      "corexyz[3]/D" );
  mcthTree->Branch ( "height",      &height,      "height/D"     );
  mcthTree->Branch ( "mctheta",     &mctheta,     "mctheta/D"    );
  mcthTree->Branch ( "mcphi",       &mcphi,       "mcphi/D"      );
  mcthTree->Branch ( "mcxcore",     &mcxcore,     "mcxcore/D"    );
  mcthTree->Branch ( "mcycore",     &mcycore,     "mcycore/D"    );
  mcthTree->Branch ( "mcenergy",    &mcenergy,    "mcenergy/D"   );
  mcthTree->Branch ( "mctc",        &mctc,        "mctc/I"       );
  mcthTree->Branch ( "mcbdist",     &mcbdist,     "mcbdist/D"    );
  mcthTree->Branch ( "mctdist",     &mctdist,     "mctdist/D"    );


  while ((dInFile=pullFile()))
    {
      fprintf(stdout,"DATA FILE: %s\n",dInFile);
      fflush (stdout);
      
      // Open event DST file
      if (!dstio->openDSTinFile(dInFile))
	return -1;
      
      // Go over all the events in the run.
      while (dstio->readEvent())
	{
	  yymmdd   = rusdraw_.yymmdd;
	  hhmmss   = rusdraw_.hhmmss;
	  usec     = rusdraw_.usec;
	  nofwf    = rusdraw_.nofwf;
	  
	  // MC variables
	  parttype    = rusdmc_.parttype;
	  corecounter = rusdmc_.corecounter;
	  corexyz[0]  = rusdmc_.corexyz[0];
	  corexyz[1]  = rusdmc_.corexyz[1];
	  corexyz[2]  = rusdmc_.corexyz[2];
	  height      = rusdmc_.height;
	  mctheta     = RadToDeg() * (rusdmc_.theta);
	  mcphi       = RadToDeg() * (rusdmc_.phi);
	  mcenergy    = rusdmc_.energy;
	  
	  mctc        = rusdmc_.tc; 
	  
	  mcxcore = ((double)rusdmc_.corexyz[0]) / 1.2e5 - RUSDGEOM_ORIGIN_X_CLF ;
	  mcycore = ((double)rusdmc_.corexyz[1]) / 1.2e5 - RUSDGEOM_ORIGIN_Y_CLF;
	  
	  sdbdist(mcxcore, mcycore, &v[0], &mcbdist, &vbr[0], &tdistbr, &vlr[0], &tdistlr,
		  &vsk[0], &tdistsk);
	  

	  // Pick out the actual T-shape boundary distance for whatever subarray
	  mctdist = tdistbr;
	  if (tdistlr > mctdist) mctdist = tdistlr;
	  if (tdistsk > mctdist) mctdist = tdistsk;

	  mcthTree->Fill();


	} // while(dstio->readEvent ...

      dstio->closeDSTinFile();

    }

  //////////////// Finish the outputs //////////////////

  thmcOut = mcthTree->GetCurrentFile();
  thmcOut->Write();
  thmcOut->Close();

  fprintf(stdout,"\n\nDone\n");
  return 0;
}
