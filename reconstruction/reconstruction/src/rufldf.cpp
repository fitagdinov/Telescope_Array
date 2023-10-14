#include "rufldf.h"
#include "sddstio.h"
#include "rufldfAnalysis.h"

int main(int argc, char **argv)
  {
    listOfOpt opt; // to handle the program arguments.
    char *dInFile; // input dst files
    char dOutFile[0x400]; // for making output dst files

    integer4 size;
    integer4 addBanks;
    integer4 reqBanks;

    sddstio_class *dstio; // DST I/O
    rufldfAnalysis *ana; // analysis
    
    int nevents_processed = 0;
    int nevents_analyzed  = 0;
    
    size = 10;
    addBanks = newBankList(size);
    // This bank needs to be added to the event
    // after successful analysis
    addBankList(addBanks, RUFLDF_BANKID);

   
    // for checking whether events have these
    // required banks
    reqBanks = newBankList(size);
    addBankList(reqBanks, RUSDRAW_BANKID);
    addBankList(reqBanks, RUFPTN_BANKID);
    addBankList(reqBanks, RUSDGEOM_BANKID);
    
    if (!opt.getFromCmdLine(argc, argv))
      return -1;

    opt.printOpts();

    // Initialize SD run class
    dstio = new sddstio_class(opt.verbose);

    // initialize the analysis here
    ana = new rufldfAnalysis(opt);

    //////////////// Prepare the outputs /////////////////////////
    
    // Initialize the DST output file
    dOutFile[0] = 0;

    if (opt.outfile[0])
      {
        if (!dstio->openDSToutFile(opt.outfile, opt.fOverwriteMode))
          return -1;
      }

    nevents_processed = 0;
    nevents_analyzed  = 0;
    
    while ((dInFile = pullFile()))
      {
        if (opt.outfile[0])
          {
            if (!dstio->openDSTinFile(dInFile))
              return -1;
          }
        else
          {
            if (!SDIO::makeOutFileName(dInFile, opt.dout, (char *) RUFLDF_DST_GZ,
                dOutFile))
              return -1;
            if (!dstio->openDSTfiles(dInFile, dOutFile, opt.fOverwriteMode))
              return -1;
          }

        while (dstio->readEvent())
          {
            // analyze the event if it has all required banks
            if(dstio->haveBanks(reqBanks,opt.bank_warning_opt))
	      {
		ana->analyzeEvent();
		if (!dstio->writeEvent(addBanks, true))
		  return -1;
		nevents_analyzed ++;
	      }
	    else
	      {
		if (!dstio->writeEvent())
		  return -1;
	      }
	    nevents_processed ++;
	    if (opt.verbose > 0)
	      {
		if (nevents_processed % 1000 == 0)
		  {
		    fprintf(stdout,"events processed: %d\n", nevents_processed);
		    fflush(stdout);
		  }
	      }
          } // while(sdrun->readEvent ...
	
        if (opt.outfile[0])
	  dstio->closeDSTinFile();
        else
	  dstio->closeDSTfiles();
      }
    
    fprintf(stdout,"rufldf_events_processed: %d rufldf_events_with_proper_banks %d\n", 
	    nevents_processed, nevents_analyzed);
    
    if (opt.outfile[0])
      dstio->closeDSToutFile();
    
    fprintf(stdout, "\n\nDone\n");
    return 0;
  }

