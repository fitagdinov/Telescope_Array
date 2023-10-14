#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "sddstio.h"
#include "mc2tasdcalibev_class.h"
#include <map>

using namespace std;

class sdmc_conv2tasdcalibev_cmdline_opt
{

  public:

    char progName[0x400];
    char constfile[0x400]; // dst file with tasd constants
    char mcfile[0x400];
    char calfile[0x400];
    char outfile[0x400];
    bool fOverWriteMode;
    bool addRu;
    sdmc_conv2tasdcalibev_cmdline_opt()
    {
      progName[0] = '\n';
      constfile[0] = 0;
      mcfile[0] = 0;
      calfile[0] = 0;
      outfile[0] = 0;
      fOverWriteMode = false;
      addRu = false;
    }
    virtual ~sdmc_conv2tasdcalibev_cmdline_opt()
    {
    }

    bool getFromCmdLine(int argc, char **argv)
    {

      int i;
      if (argc == 1)
        {
          memcpy(progName, argv[0], (strlen(argv[0]) + 1 <= 0x400 ? strlen(argv[0]) + 1 : 0x400));
          printMan();
          return false;
        }
      for (i = 1; i < argc; i++)
        {
          // print the manual
          if ((strcmp("-h", argv[i]) == 0) || (strcmp("--h", argv[i]) == 0) || (strcmp("-help", argv[i]) == 0)
              || (strcmp("--help", argv[i]) == 0) || (strcmp("-?", argv[i]) == 0) || (strcmp("--?", argv[i]) == 0)
              || (strcmp("/?", argv[i]) == 0))
            {
              printMan();
              return false;
            }
          // input MC file
          else if (strcmp("-imc", argv[i]) == 0)
            {
              if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
                {
                  fprintf(stderr, "error: -imc: specify the input MC file!\n");
                  return false;
                }
              else
                sscanf(argv[i], "%1023s", mcfile);
            }
          // input constants file
          else if (strcmp("-ico", argv[i]) == 0)
            {
              if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
                {
                  fprintf(stderr, "error: -ico: specify the input constants file!\n");
                  return false;
                }
              else
                sscanf(argv[i], "%1023s", constfile);
            }
          // input calibration file
          else if (strcmp("-icl", argv[i]) == 0)
            {
              if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
                {
                  fprintf(stderr, "error: -icl: specify the input calibration file!\n");
                  return false;
                }
              else
                sscanf(argv[i], "%1023s", calfile);
            }
          // output file
          else if (strcmp("-o", argv[i]) == 0)
            {
              if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
                {
                  fprintf(stderr, "error: -o: specify the output file!\n");
                  return false;
                }
              else
                sscanf(argv[i], "%1023s", outfile);
            }
          // force overwrite mode
          else if (strcmp("-f", argv[i]) == 0)
            fOverWriteMode = true;
          else if (strcmp("-ru", argv[i]) == 0)
            addRu = true;
          else
            {
              fprintf(stderr, "error: '%s': unrecognized option\n", argv[i]);
              return false;
            }

        }
      return checkOpt();
    }

    void printMan()
    {
      fprintf(stderr, "\nCalibrate Rutgers MC with tasdcalib and write the results in ICRR (tasdcalibev) dst format\n");
      fprintf(
          stderr,
          "\nusage: %s -imc [dst file with rusdmc,rusdraw]  -icl [dst file with tasdcalib] -o [tasdcalibev dst file]\n",
          progName);
      fprintf(stderr, "-ico <string> : input dst file with ta sd constants (with tasdconst dst bank)\n");
      fprintf(stderr, "-imc <string> : input dst file with rusdmc events (with rusdmc,rusdraw banks)\n");
      fprintf(stderr, "-icl <string> : input dst file with calibration information (tasdcalib bank)\n");
      fprintf(stderr, "-o:  <string> : dst output file ( with tasdcalibev bank) (default is './tasdcalibev.dst')\n");
      fprintf(stderr, "-f            : don't check if the output file exists; overwrite it ( default: -f not set)\n");
      fprintf(stderr, "-ru           : output includes the original rusdraw, rusdmc banks (by default it is off)\n");
      fprintf(stderr, "\n");
    }

  private:

    bool checkOpt()
    {
      bool fflag = true;
      if (!mcfile[0])
        {
          fprintf(stderr, "input MC file not specified!\n");
          fflag = false;
        }
      if (!calfile[0])
        {
          fprintf(stderr, "input calibration file not specified!\n");
          fflag = false;
        }
      if (!outfile[0])
        sprintf(outfile, "./tasdcalibev.dst");
      return fflag;
    }
};

int main(int argc, char **argv)
{

  int events_read;
  int events_filled;
  int events_triggered;
  int yymmdd_set; // date of the 1st event.  Should be consistent with other events and calibration cycles.

  sdmc_conv2tasdcalibev_cmdline_opt *opt = new sdmc_conv2tasdcalibev_cmdline_opt();
  if (!opt->getFromCmdLine(argc, argv))
    return 2;

  sddstio_class *dstio = new sddstio_class(); // dst I/O handler class
  mc2tasdcalibev_class *conv = new mc2tasdcalibev_class(); // the converter class;

  // input constants bank
  integer4 constBank = newBankList(10);
  addBankList(constBank, TASDCONST_BANKID);

  // input calibration banks
  integer4 inCalBanks = newBankList(10);
  addBankList(inCalBanks, TASDCALIB_BANKID);

  // input MC banks
  integer4 inMcBanks = newBankList(10);
  addBankList(inMcBanks, RUSDMC_BANKID);
  addBankList(inMcBanks, RUSDRAW_BANKID);

  // output MC banks
  integer4 outMcBanks = newBankList(10);
  addBankList(outMcBanks, TASDCALIBEV_BANKID);

  /////////////// read in the tasd constants into the tasdconst dst bank, fill the latest available for each SD
  dstio->setWantBanks(constBank);
  if (!dstio->openDSTinFile(opt->constfile))
    return 2;
  while (dstio->readEvent())
    {
      if (!dstio->haveBank(TASDCONST_BANKID, true))
        continue;
      conv->add_cnst_info(&tasdconst_);
    }
  dstio->closeDSTinFile();

  ///////  read all calibration information for the day into a buffer //////
  dstio->setWantBanks(inCalBanks);
  if (!dstio->openDSTinFile(opt->calfile))
    return 2;
  while (dstio->readEvent())
    {
      if (!dstio->haveBank(TASDCALIB_BANKID))
        continue;
      // fatal error occurred if can't add the calibration information
      if (!conv->add_calib_info(&tasdcalib_))
        return 2;
    }
  dstio->closeDSTinFile();

  //////////////////////// calibrate and write out the MC events ////////////////////

  if (!dstio->openDSTinFile(opt->mcfile))
    return 2;
  dstio->setWantBanks(inMcBanks);

  events_read = 0;
  events_filled = 0;
  events_triggered = 0;
  yymmdd_set = -1;
  while (dstio->readEvent())
    {
      events_read++;
      if (events_read > 1)
        {
          if (rusdraw_.yymmdd != yymmdd_set)
            {
              fprintf(stderr, "warning: event date %06d not equal to 1st event date in the file %06d; skipping\n",
                  rusdraw_.yymmdd, yymmdd_set);
              continue;
            }
        }
      else
        {
          yymmdd_set = rusdraw_.yymmdd;
          if (yymmdd_set != conv->getCalibDate())
            {
              fprintf(stderr, "fatal error: 1st event date %06d doesn't match the date of the calibration %06d\n",
                  yymmdd_set, conv->getCalibDate());
              return 2;
            }
        }
      // do the conversion
      if (!conv->Convert())
        continue;
      // write out the event
      if (!events_filled)
        {
          if (!dstio->openDSToutFile(opt->outfile, opt->fOverWriteMode))
            return 2;
        }
      dstio->writeEvent(outMcBanks, opt->addRu);
      events_filled++;
      if (tasdcalibev_.numTrgwf > 0)
        events_triggered++;
    }

  dstio->closeDSTinFile();
  if (events_filled)
    dstio->closeDSToutFile();

  fprintf(stdout, "events_read: %d events_filled: %d events_triggered: %d\n", events_read, events_filled,
      events_triggered);

  fprintf(stdout, "\nDone\n");
  return 0;
}
