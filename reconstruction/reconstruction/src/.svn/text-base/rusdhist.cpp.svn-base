///// Program to make histograms for data and MC //////////////
///// Dmitri Ivanov, ivanov@physics.rutgers.edu //////////////
//////////////////////////////////////////////////////////////

#include "rusdhist.h"
#include "rusdhist_class.h"

int main(int argc, char **argv)
{
  listOfOpt opt; // to handle the program arguments.
  sddstio_class *dstio; // to handle the DST files
  char *dInFile; // pass1 dst input files
  rusdhist_class *rusdhist; // class that fills the histograms
  bool have_mc_banks;
  integer4 size, needBanks;
  integer4 bankid, n;
  char bankname[0x20];
  integer4 len = 0x20;

  if (!opt.getFromCmdLine(argc, argv))
    return -1;

  size = 100;
  needBanks = newBankList(size);

  // these banks are mandatory
  addBankList(needBanks, RUSDRAW_BANKID);
  addBankList(needBanks, RUFPTN_BANKID);
  addBankList(needBanks, RUSDGEOM_BANKID);
  addBankList(needBanks, RUFLDF_BANKID);
  if (opt.tbopt)
    addBankList(needBanks, SDTRGBK_BANKID);
  
  
  opt.printOpts();

  dstio = new sddstio_class();
  rusdhist = new rusdhist_class(opt);

  while ((dInFile = pullFile()))
    {
      if (!dstio->openDSTinFile(dInFile))
        return -1;
      while (dstio->readEvent())
        {
          if (!dstio->haveBanks(needBanks))
            {
              int n_nf = 0;
              n = 0;
	      if(opt.bank_warning_opt)
		{
		  fprintf(stderr, "warning: needed bank(s) not found: ");
		  while ((bankid = itrBankList(needBanks, &n)))
		    {
		      if (!dstio->haveBank(bankid))
			{
			  eventNameFromId(bankid, bankname, len);
			  if (n_nf)
			    fprintf(stderr, ", ");
			  fprintf(stderr, "%s", bankname);
                      n_nf++;
			}
		    }
		  fprintf(stderr, " skipping event (s)\n");
		}
              continue;
            }
          have_mc_banks = (dstio->haveBank(RUSDMC_BANKID) && dstio->haveBank(RUSDMC1_BANKID));
          rusdhist->Fill(have_mc_banks); // fills the histograms
        }
      dstio->closeDSTinFile();
    }

  delete rusdhist; // finalizes the output root file

  fprintf(stdout, "\n\nDone\n");
  return 0;
}
