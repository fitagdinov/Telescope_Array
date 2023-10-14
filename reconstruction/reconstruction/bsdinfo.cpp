#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include "event.h"
#include "filestack.h"
#include <errno.h>
#include "tacoortrans.h"
#include "sdxyzclf_class.h"
#include "sdmc_tadate.h"
#include "sdmc_bsd_bitf.h"
#include <vector>
#include <map>

using namespace std;



static sdxyzclf_class sdxyzclf;

/* Taken from sdmc code (below) */

#define NCOUNTERS_MAX 507
#define NCAL 26 /* sdmc.h */
/* typical value for the counter saturation information that's currently used by TA SD MC */
#define FADC_NONLIN1 2000.


/* sdmc uses an array of 4 integers to describe each counter:
[0] - counter XXYY poistion ID
[1] - counter X position, CLF frame, [cm]
[2] - counter Y position, CLF frame, [cm]
[3] - counter Z position, CLF frame, [cm]
We fill this array with the most recent information available
in the SD analysis. */
static int NCOUNTERS = 0;
static int sdcoor0[NCOUNTERS_MAX][4];


/* From sdmc code (above) */

/* Maximum number of entries in the arrays that describe the saturation */
#define SATNUM 50000

static int sdnum[SATNUM], taperfrom[SATNUM], taperto[SATNUM], lu_sat[SATNUM][2];
static int satmax = 0;

static int load_saturation_info(const char *tasdconstfile);


/* to load the counter information into sdcoor0 array */
static bool load_sdcoor0(int yymmdd);


int main(int argc, char **argv) 

{
  integer4 dst_unit_in  = 1;
  integer4 dst_unit_out = 2;
  integer4 banklist_want, banklist_got;
  integer4 rcode, read_rcode=0, write_rcode=0, arg, ssf, i, j, M, Y, ta_period, k, ii, flg;
  integer4 dst_read_mode    = MODE_READ_DST;
  integer4 dst_write_mode   = MODE_WRITE_DST;
  integer1 *dstinfile     = 0;
  integer1 *tasdconstfile = 0;
  integer1 *dstoutfile    = 0;
  real4 D, hour, minute, second, buf[NCAL];
  integer4 verbosity = 1, ievent=0;
 
   
  if (argc == 1) 
    {
      fprintf(stderr,"\nLabel bad counters if they fail sdmc check criteria (ICRR calibration or Rutgers calibration is bad)\n");
      fprintf(stderr,"The program adds bsdinfo DST bank.  The bsdinfo DST bank is then used by the reconstruction programs to exclude bad counters.\n");
      fprintf(stderr,"The program requires presence of tasdcalibev DST bank for each event\n");
      fprintf(stderr,"In the case of SD Monte Carlo, these checks are made before throwing events (bad counters are not used); running this program on\n");
      fprintf(stderr,"SD Monte Carlo is OK to do but it is not required because SD MC output has this bank.\n");
      fprintf(stderr,"\nAuthors: B. T. Stokes, D. Ivanov <dmiivanov@gmail.com>\n");
      fprintf(stderr,"\nUsage: %s [-i event_dst_intfile] -c [path to recent tasdconst_pass2.dst file] -o [event_dst_dstoutfile]\n",argv[0]);
      fprintf(stderr,"     -i <string> : full path to an input event tasdcalibev_pass2_YYMMDD.dst DST file\n");
      fprintf(stderr,"     -o <string> : output DST file name for events that have (in addition) bsdinfo DST bank\n");
      fprintf(stderr,"     -c <string> : (optional) full path to the most recent tasdconst_pass2.dst file\n");
      fprintf(stderr,"                   needed only to check if the counter has saturation info available, which most counters nearly always do.\n");
      fprintf(stderr,"                   So far, this is not essential.  Also, in the future, SDMC may be changed to use only typical values\n");
      fprintf(stderr,"     -v <int>    : (optional) verbosity level > 0 prints more; default: %d\n\n", verbosity);
      exit(1);
    }
  
  for (arg = 1; arg < argc; ++arg) 
    {
      if (argv[arg][0] != '-') 
	{
	  pushFile(argv[arg]);
	  continue;
	}
      switch (argv[arg][1]) 
	{
	case 'i':
	  arg++; dstinfile = argv[arg];
	  break;
	case 'o': 
	  arg++; dstoutfile = argv[arg]; 
	  break;
	case 'c': 
	  arg++; tasdconstfile = argv[arg]; 
	  break;
	case 'v':
	  arg++; sscanf(argv[arg],"%d", &verbosity);
	  break;
	default: 
	  fprintf(stderr,"Warning: unknown option: %s\n",argv[arg]); 
	  break;
	}
    }
  
  /* check the command line arguments */
  rcode = 0;
  if(!dstinfile)
    {
      fprintf(stderr,"error: input DST file not specified, use -i option!\n");
      rcode ++;
    }
  if(!dstoutfile)
    {
      fprintf(stderr,"error: output DST file not specified, use -o option!\n");
      rcode ++;
    }
  if(rcode)
    exit(EXIT_FAILURE);

  if(verbosity >= 1)
    {
      fprintf(stdout,"DST input file: %s\n", dstinfile);
      fprintf(stdout,"DST output file: %s\n", dstoutfile);
      fprintf(stdout,"DST constants file: %s\n", (tasdconstfile ? tasdconstfile : "NOT USED"));
      fprintf(stdout,"Verbosity level: %d\n", verbosity);
    }

  if(verbosity >= 2)
    {
      fprintf(stderr,"Check %d counters\n", NCOUNTERS);
      for (int i=0; i<NCOUNTERS; i++)
	fprintf(stdout,"%04d %d %d %d\n",sdcoor0[i][0],sdcoor0[i][1],sdcoor0[i][2],sdcoor0[i][3]);
    }
  // load the saturation information if the tasdconst file has been given
  if(tasdconstfile)
    {
      if(load_saturation_info(tasdconstfile) == FALSE)
	exit(EXIT_FAILURE);
    }

  banklist_want = newBankList(512);
  banklist_got  = newBankList(512);
  eventAllBanks(banklist_want);


  if ((rcode = dst_open_unit_(&dst_unit_in, dstinfile, &dst_read_mode))) 
    { 
      fprintf(stderr,"\n  Unable to open/read file: %s\n\n", dstinfile);
      exit(1);
    }
  if ((rcode = dst_open_unit_(&dst_unit_out, dstoutfile, &dst_write_mode))) 
    { 
      fprintf(stderr,"\n  Unable to open/write file: %s\n\n", dstoutfile);
      exit(1);
    }
  ievent = 0;
  for (;;) 
    {
      read_rcode = eventRead(dst_unit_in, banklist_want, banklist_got, &ssf);
      if ( read_rcode < 0 ) break ;
      ievent++;
      // need tasdcalibev DST bank, but if this is a Monte-Carlo event
      // then it's not a problem because TA SD Monte Carlo does not use
      // bad counters in the simulation.
      if(!tstBankList(banklist_got,TASDCALIBEV_BANKID))
	{
	  if(!tstBankList(banklist_got,RUSDMC_BANKID))
	    {
	      if(verbosity >= 1)
		fprintf(stderr,"tasdcalibev absent for real data event %d in %s\n",ievent,dstinfile);
	    }
	  write_rcode = eventWrite(dst_unit_out, banklist_got, TRUE);
	  if( write_rcode < 0 ) break;
	  continue;
	}
      Y=(int)tasdcalibev_.date/1e4;
      M=((int)tasdcalibev_.date/1e2)-1e2*Y;
      D=tasdcalibev_.date-1.e4*(real4)Y-1.e2*(real4)M;
      Y += 2000;
      hour=floor(tasdcalibev_.time/1.e4);
      minute=floor(tasdcalibev_.time/1.e2)-1.e2*hour;
      second=tasdcalibev_.time-1.e4*hour-1.e2*minute;
      D += hour/24.+minute/1440.+second/86400.;
      ta_period = (int) rint(Date2TADay( D, M, Y )*144.);
      bsdinfo_.yymmdd = (int)tasdcalibev_.date;
      bsdinfo_.hhmmss = (int)tasdcalibev_.time;
      bsdinfo_.usec   = (int)tasdcalibev_.usec;
      bsdinfo_.nbsds  = 0;
      if(!load_sdcoor0((int)tasdcalibev_.date))
	 exit(EXIT_FAILURE);
      for (j = 0; j < tasdcalibev_.numTrgwf; j++ )
	{
	  for (k = 0; k < NCOUNTERS; k++)
	    if (tasdcalibev_.sub[j].lid == sdcoor0[k][0]) break;
	  if(k == NCOUNTERS)
	    {
	      if(verbosity >= 2)
		fprintf(stderr,"counter %04d present in event but absent in analysis\n",
			(int)tasdcalibev_.sub[j].lid);
	      continue;
	    }
	  buf[0] = (float)ta_period;
	  buf[1] = (float)k;
	  buf[2] = (float)tasdcalibev_.sub[j].dontUse;
	  buf[3] = tasdcalibev_.sub[j].umipMev2pe;
	  buf[4] = tasdcalibev_.sub[j].lmipMev2pe;
	  buf[5] = tasdcalibev_.sub[j].umipMev2cnt;
	  buf[6] = tasdcalibev_.sub[j].lmipMev2cnt;
	  buf[7] = tasdcalibev_.sub[j].mip[1];
	  buf[8] = tasdcalibev_.sub[j].mip[0];
	  buf[9] = tasdcalibev_.sub[j].upedAvr;
	  buf[10] = tasdcalibev_.sub[j].lpedAvr;
	  buf[11] = tasdcalibev_.sub[j].upedStdev;
	  buf[12] = tasdcalibev_.sub[j].lpedStdev;
	  buf[13] = (float)tasdcalibev_.sub[j].pchped[1];
	  buf[14] = (float)tasdcalibev_.sub[j].pchped[0];
	  buf[15] = (float)tasdcalibev_.sub[j].lhpchped[1];
	  buf[16] = (float)tasdcalibev_.sub[j].lhpchped[0];
	  buf[17] = (float)tasdcalibev_.sub[j].rhpchped[1];
	  buf[18] = (float)tasdcalibev_.sub[j].rhpchped[0];
	  buf[19] = (float)tasdcalibev_.sub[j].mftndof[1];
	  buf[20] = (float)tasdcalibev_.sub[j].mftndof[0];
	  buf[21] = tasdcalibev_.sub[j].mftchi2[1];
	  buf[22] = tasdcalibev_.sub[j].mftchi2[0];
	  buf[23] = tasdcalibev_.sub[j].site;
	  buf[24] = FADC_NONLIN1;
	  buf[25] = FADC_NONLIN1;
	  for (i=0; i<satmax; i++)
	    {
	      if (taperfrom[i] <= ta_period && ta_period <= taperto[i] && k == sdnum[i])
		{
		  buf[24]=(float)lu_sat[i][0];
		  buf[25]=(float)lu_sat[i][1];
		  break;
		}
	    }
	  // Now, check if the counter has bad Rutgers OR ICRR calibration,
	  // if that's the case, add the counter to the list of bad SDs
	  if((flg=failed_Ben_Stokes_D_Ivanov_Live_Counter_Criteria(buf)))
	    {
	      int have_in_list = 0;
	      for (ii=0; ii < bsdinfo_.nbsds; ii++)
		{
		  if(bsdinfo_.xxyy[ii] == (int)tasdcalibev_.sub[j].lid)
		    {
		      have_in_list = 1;
		      break;
		    }
		}
	      if(!have_in_list && bsdinfo_.nbsds < BSDINFO_NBSDS)
		{
		  bsdinfo_.xxyy[bsdinfo_.nbsds] = (int)tasdcalibev_.sub[j].lid;
		  bsdinfo_.bitf[bsdinfo_.nbsds] = flg;
		  bsdinfo_.nbsds ++;
		}
	      if(!have_in_list && bsdinfo_.nbsds >= BSDINFO_NBSDS)
		fprintf(stderr,"warning: number of bad SDs %d exeeds maximum allowed %d; not filling\n",
			bsdinfo_.nbsds,  BSDINFO_NBSDS);
	    }
	}
      bsdinfo_.nsdsout = 0; /* fill SDs that are completely out - they are listed as dead in tasdcalibev */
      map<int,int> live_counter;
      for (k = 0; k < NCOUNTERS; k++)
	live_counter[sdcoor0[k][0]] = 1;
      for (j = 0; j < tasdcalibev_.numDead; j++)
	live_counter[(int)tasdcalibev_.deadDetLid[j]] = 0;
      for (k = 0; k < NCOUNTERS; k++)
	{
	  if(live_counter[sdcoor0[k][0]])
	    continue;
	  int have_in_list = 0;
	  for (ii=0; ii < bsdinfo_.nbsds; ii++)
	    {
	      if(bsdinfo_.xxyy[ii] == sdcoor0[k][0])
		{
		  have_in_list = 1;
		  break;
		}
	    }
	  if(have_in_list)
	    continue;
	  for (ii=0; ii < bsdinfo_.nsdsout; ii++)
	    {
	      if(bsdinfo_.xxyyout[ii] == sdcoor0[k][0])
		{
		  have_in_list = 1;
		  break;
		}
	    }
	  if(have_in_list)
	    continue;
	  bsdinfo_.xxyyout[bsdinfo_.nsdsout] = sdcoor0[k][0];
	  bsdinfo_.bitfout[bsdinfo_.nsdsout] = 0xFFFF;
	  bsdinfo_.nsdsout++;
	}
      addBankList(banklist_got,BSDINFO_BANKID);
      write_rcode = eventWrite(dst_unit_out, banklist_got, TRUE);
      if( write_rcode < 0 ) break;
    }
  if( write_rcode < 0 )
    {
      fprintf(stderr,"Error writing %s\n",dstoutfile);
      return -3;
    }
  if (read_rcode != END_OF_FILE) 
    {
      fprintf(stderr,"Error reading %s\n",dstinfile);
      return -3;
    }
  dst_close_unit_(&dst_unit_in);
  dst_close_unit_(&dst_unit_out);
  fprintf(stdout,"\nDone\n");
  fflush(stdout);
  return 0;
}
 
bool load_sdcoor0(int yymmdd)
{
  vector<sdxyzclf_class::sdpos>& sdpos = sdxyzclf.get_counters(yymmdd);
  if((int)sdpos.size() > NCOUNTERS_MAX)
    {
      fprintf(stderr,"error: load_sdcoor0: number of counters (%d) exceeds maximum (%d)\n",
	       (int)sdpos.size(),NCOUNTERS_MAX);
      return false;
    }
  NCOUNTERS = 0;
  for (vector<sdxyzclf_class::sdpos>::iterator isd=sdpos.begin(); isd != sdpos.end(); isd++)
    {
      if(NCOUNTERS > NCOUNTERS_MAX)
	{
	  fprintf(stderr,"error: load_sdcoor0: number of counters (%d) exceeds maximum (%d)\n",
		  NCOUNTERS,NCOUNTERS_MAX);
	  return false;
	}
      sdcoor0[NCOUNTERS][0] = (*isd).xxyy;
      for (int i=0; i<2; i++)
	sdcoor0[NCOUNTERS][i+1] = (int)floor(0.5+100.0*1200.0*((*isd).xyz[i])); // in cm units
      // in cm units and including full height above sea level, not just height above CLF
      sdcoor0[NCOUNTERS][3] = (int)floor(0.5+100.0*(tacoortrans_CLF_Altitude+1200.0*((*isd).xyz[2])));
      NCOUNTERS++;
    }
  return true;
}

int load_saturation_info(const char *tasdconstfile)
{
  int xxyy, datefrom, timefrom, dateto, timeto, umip, lmip, uled, lled, year, month;
  int ta_period1, ta_period2, i, usat, lsat;
  float day, hour, minute, second;
  integer4 rc=0, unit=1, mode=MODE_READ_DST;
  integer1 name[1024];
  integer4 wantBanks = 0;
  integer4 gotBanks = 0;
  integer4 event=0;
  sprintf(name,"%s",tasdconstfile);
  if ((rc=dstOpenUnit(unit, name, mode))) 
    {
      fprintf(stderr, "Error %d: failed to open for reading dst file: %s\n", rc, name);
      return FALSE;
    }
  rc=150;
  wantBanks = newBankList(rc);
  gotBanks  = newBankList(rc);
  satmax    = 0;
  while ((rc=eventRead(unit,wantBanks,gotBanks,&event))>0)
    {
      if(!tstBankList(gotBanks,TASDCONST_BANKID))
	continue;
      xxyy    = tasdconst_.lid; 
      datefrom = tasdconst_.dateFrom; 
      timefrom = tasdconst_.timeFrom; 
      dateto   = tasdconst_.dateTo; 
      timeto   = tasdconst_.timeTo; 
      umip     = tasdconst_.udec5pmip; 
      lmip     = tasdconst_.ldec5pmip; 
      uled     = tasdconst_.udec5pled; 
      lled     = tasdconst_.ldec5pled;
      year = datefrom/1e4;
      month = (datefrom - year*1e4)/100;
      day = (float)(datefrom - year*1e4 - month*100);
      year +=2000;
      hour=floor((float)timefrom/1.e4);
      minute=floor((float)timefrom/1.e2)-1.e2*hour;
      second=(float)timefrom-1.e4*hour-1.e2*minute;
      day += hour/24.+minute/1440.+second/86400.;
      ta_period1 = (int) rint(Date2TADay( day, month, year )*144.);
      year = dateto/1e4;
      month = (dateto - year*1e4)/100;
      day = (float)(dateto - year*1e4 - month*100);
      year +=2000;
      hour=floor((float)timeto/1.e4);
      minute=floor((float)timeto/1.e2)-1.e2*hour;
      second=(float)timeto-1.e4*hour-1.e2*minute;
      day += hour/24.+minute/1440.+second/86400.;
      ta_period2 = (int) rint(Date2TADay( day, month, year )*144.);
      for ( i = 0; i < NCOUNTERS; i++ )
	{
	  if ( xxyy == sdcoor0[i][0] )
	    break;
	}
      if (umip > uled) usat = uled;
      else usat = umip;
      if (lmip > lled) lsat = lled;
      else lsat = lmip;
      if (i < NCOUNTERS)
	{
	  if(satmax >= SATNUM)
	    {
	      fprintf(stderr,"error: (%s):%d: increase SATNUM constant!\n",__FILE__,__LINE__);
	      return FALSE;
	    }
	  sdnum[satmax]        = i;
	  taperfrom[satmax]    = ta_period1;
	  taperto[satmax]      = ta_period2;
	  lu_sat[satmax][0]    = lsat;
	  lu_sat[satmax][1]    = usat;
	  satmax ++;
	}
    }
  dstCloseUnit(unit);
  return TRUE;
}
