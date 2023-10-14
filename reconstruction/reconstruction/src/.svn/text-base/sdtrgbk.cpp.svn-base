#include "sdtrgbk.h"
#include "sduti.h"
#include "sddstio.h"
#include "sdtrgbkAnalysis.h"

int main(int argc, char **argv)
{

  integer4 size, addBanks;
  int nevents_processed, nevents_written;
  char *infile, outfile[0x100];
  sdtrgbkAnalysis *trig;
  listOfOpt opt;

  if (!opt.getFromCmdLine(argc, argv))
    return -1;

  opt.printOpts();

  trig = new sdtrgbkAnalysis(opt);

  size = 10;
  addBanks = newBankList(size);
  addBankList(addBanks, SDTRGBK_BANKID);

  sddstio_class *dstio = new sddstio_class();

  // start the single output file
  // if such option is used
  if (opt.outfile[0])
    {
      if (!dstio->openDSToutFile(opt.outfile, opt.fOverwriteMode))
        return -1;
    }

  // going over all input dst files
  nevents_processed = 0;
  nevents_written = 0;
  while ((infile = pullFile()))
    {
      // for single output file option,
      // open only the input files
      if (opt.outfile[0])
        {
          if (!dstio->openDSTinFile(infile))
            return -1;
        }
      else
        {
          // output file for each input file
          if (SDIO::makeOutFileName(infile, opt.dout, (char *) ".sdtrgbk.dst.gz", outfile) != 1)
            return -1;
          if (!dstio->openDSTfiles(infile, outfile, opt.fOverwriteMode))
            return -1;
        }

      // reading each DST file
      while (dstio->readEvent())
        {
	  
	  // if bank that describes information on bad SDs is absent for the event, then
	  // make sure the corresponding structure is zeroed out.
	  if(!dstio->haveBank(BSDINFO_BANKID))
	    memset(&bsdinfo_,0,sizeof(bsdinfo_dst_common));
	  
          // use ICRR bank
          if (opt.icrrbankoption == 0)
            {
              if (!dstio->haveBank(TASDEVENT_BANKID))
                {
                  fprintf(stderr, "Error: needed 'tasdevent' bank is absent but '-icrr 0' option is used\n");
                  return -1;
                }
              trig->analyzeEvent(&tasdevent_,&bsdinfo_);
            }
          else if (opt.icrrbankoption == 2)
            {
              if (!dstio->haveBank(TASDCALIBEV_BANKID))
                {
                  fprintf(stderr, "Error: needed 'tasdcalibev' bank is absent but '-icrr 2' option is used\n");
                  return -1;
                }
              trig->analyzeEvent(&tasdcalibev_,&bsdinfo_);
            }
	  // if icrr option was not used then use any bank that's available
	  else if (opt.icrrbankoption == -1)
	    {
	      bool analyzed_event = false;
	      // try using Rutgers bank first
              if (dstio->haveBank(RUSDRAW_BANKID))
		{
		  trig->analyzeEvent(&rusdraw_,&bsdinfo_);
		  analyzed_event = true;
		}
	      // if no rusdraw, then try using tasdcalibev
	      if(!analyzed_event && dstio->haveBank(TASDCALIBEV_BANKID))
		{
		  trig->analyzeEvent(&tasdcalibev_,&bsdinfo_);
		  analyzed_event = true;
		}
	      // if no rusdraw and no tasdcalibev, then try using tasdevent
	      if(!analyzed_event && dstio->haveBank(TASDEVENT_BANKID))
		{
		  trig->analyzeEvent(&tasdevent_,&bsdinfo_);
		  analyzed_event = true;
		}
	      // if none of the SD event banks are present, then zero out rusdraw
	      // bank and run the trigger back up (which will return a non-trigger)
	      if(!analyzed_event)
		{
		  memset(&rusdraw_,0,sizeof(rusdraw_dst_common));
		  trig->analyzeEvent(&rusdraw_,&bsdinfo_);
		}
	    }
      

          if ((!opt.write_trig_only && !opt.write_notrig_only) || (opt.write_trig_only && trig->hasTriggered())
              || (opt.write_notrig_only && !trig->hasTriggered()))
            {
              dstio->writeEvent(addBanks, true);
              nevents_written++;
            }
          nevents_processed++;
          if (nevents_processed % 1000 == 0)
            {
              fprintf(stdout, "events processed: %d\n", nevents_processed);
              fflush(stdout);
            }
        } // while(dstio->readEvent ...

      // if single output file option is used,
      // close only the input dst file at this point
      if (opt.outfile[0])
        dstio->closeDSTinFile();
      else
        dstio->closeDSTfiles();

    } // while (infile = pullFile ...


  fprintf(stdout, "sdtrgbk_events_processed: %d\n", nevents_processed);
  fprintf(stdout, "sdtrgbk_events_written:   %d\n", nevents_written);

  // close the output dst file if single output file
  // option is used
  if (opt.outfile[0])
    dstio->closeDSToutFile();

  fprintf(stdout, "\n\nDone\n");

  return 0;
}
