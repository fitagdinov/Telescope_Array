#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "univ_dst.h"
#include "event.h"
#include "filestack.h"
#include "sdmc.h"
#include "sdmc_tadate.h"
#include <errno.h>


/* Maximum number of entries in the arrays that describe the saturation */
#define SATNUM 50000

static int sdnum[SATNUM], taperfrom[SATNUM], taperto[SATNUM], lu_sat[SATNUM][2];
static int satmax = 0;

int load_saturation_info(const char *tasdconstfile);

int main(int argc, char **argv) 

{
  integer4 dst_unit_in = 1;
  integer4 banklist_want, banklist_got;
  integer4 rcode, arg, ssf, i, j, M, Y, ta_period, k;
  integer4 dst_mode = MODE_READ_DST;
  integer1 *outfile       = 0;
  integer1 *tasdconstfile = 0;
  integer1 *filename      = 0;
  real4 D, hour, minute, second, buf[NCAL];
  FILE *fout;
  
  if (argc == 1) 
    {
      fprintf(stderr,"\nProgram to convert tasdconst_pass2.dst, tasdcalib_pass2_YYMMDD.dst ... files to\n");
      fprintf(stderr,"a format that can be readily used by the TA SD Monte-Carlo program.\n");
      fprintf(stderr,"It is recommended to use no more than 30 days worth of tasdcalib_pass2_YYMMDD.dst files.\n\n");
      fprintf(stderr,"Date and time of each SD monitoring cycle (every 10 min) is converted by TADay2Date\n");
      fprintf(stderr,"into ta period, and the calibration information for each ta period for each counter\n");
      fprintf(stderr,"is written into the binary output file.  The binary output file is used by the TA SD MC.\n");
      fprintf(stderr,"\nAuthors: B. T. Stokes, D. Ivanov <dmiivanov@gmail.com>\n");
      fprintf(stderr,"\nUsage: %s [-o outfile] -c [path to recent tasdconst_pass2.dst file] ",argv[0]);
      fprintf(stderr," tasdcalib_pass2_YYMMDD.dst ... files\n\n");
      fprintf(stderr,"Pass tasdcalib_pass2_YYMMDD.dst files on the command line without prefixes or switches\n");
      fprintf(stderr,"     -c <string> : full path to the most recent tasdconst_pass2.dst file\n");
      fprintf(stderr,"     -o <string> : output binary (.bin) SD calibration file name\n\n");
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
	case 'o': 
	  arg++; outfile = argv[arg]; 
	  break;
	case 'c': 
	  arg++; tasdconstfile = argv[arg]; 
	  break;
	default: 
	  fprintf(stderr,"Warning: unknown option: %s\n",argv[arg]); 
	  break;
	}
    }
  
  /* check the command line arguments */
  rcode = 0;
  if(!outfile)
    {
      fprintf(stderr,"error: output file not specified, use -o option!\n");
      rcode ++;
    }
  if(!tasdconstfile)
    {
      fprintf(stderr,"error: tasdconst_pass2.dst not given, use -c option!\n");
      rcode++;
    }
  if(rcode)
    exit(EXIT_FAILURE);

  if(load_saturation_info(tasdconstfile) == FALSE)
    exit(EXIT_FAILURE);
  
  if (!(fout=fopen(outfile, "wb")))
    {
      fprintf(stderr,"Cannot open %s file: %s\n", argv[1], strerror(errno));
      exit(EXIT_FAILURE);
    }
  banklist_want = newBankList(512);
  banklist_got  = newBankList(512);
  eventAllBanks(banklist_want);
  
  /* read tasdcalib_pass2_YYMMDD.dst files, fill and write out the calibration bufers for 
     all TA period values found in the DST files */
  while ((filename = pullFile()))
    {
      if ( (rcode = dst_open_unit_(&dst_unit_in, filename, &dst_mode)) ) 
	{ 
	  fprintf(stderr,"\n  Unable to open/read file: %s\n\n", filename);
	  exit(1);
	}
      for (;;) 
	{
	  rcode = eventRead(dst_unit_in, banklist_want, banklist_got, &ssf);
	  if ( rcode < 0 ) break ;
	  Y=(int)tasdcalib_.date/1e4;
	  M=((int)tasdcalib_.date/1e2)-1e2*Y;
	  D=tasdcalib_.date-1.e4*(real4)Y-1.e2*(real4)M;
	  Y += 2000;
	  hour=floor(tasdcalib_.time/1.e4);
	  minute=floor(tasdcalib_.time/1.e2)-1.e2*hour;
	  second=tasdcalib_.time-1.e4*hour-1.e2*minute;
	  D += hour/24.+minute/1440.+second/86400.;
	  ta_period = (int) rint(Date2TADay( D, M, Y )*144.);
	  for (j = 0; j < tasdcalib_.num_det; j++ )
	    {
	      for (k = 0; k < NCOUNTERS; k++)
		if (tasdcalib_.sub[j].lid == sdcoor0[k][0]) break;
	      if (k < NCOUNTERS)
		{
		  buf[0] = (float)ta_period;
		  buf[1] = (float)k;
		  buf[2] = (float)tasdcalib_.sub[j].dontUse;
		  buf[3] = tasdcalib_.sub[j].umipMev2pe;
		  buf[4] = tasdcalib_.sub[j].lmipMev2pe;
		  buf[5] = tasdcalib_.sub[j].umipMev2cnt;
		  buf[6] = tasdcalib_.sub[j].lmipMev2cnt;
		  buf[7] = tasdcalib_.sub[j].mip[1];
		  buf[8] = tasdcalib_.sub[j].mip[0];
		  buf[9] = tasdcalib_.sub[j].upedAvr;
		  buf[10] = tasdcalib_.sub[j].lpedAvr;
		  buf[11] = tasdcalib_.sub[j].upedStdev;
		  buf[12] = tasdcalib_.sub[j].lpedStdev;
		  buf[13] = (float)tasdcalib_.sub[j].pchped[1];
		  buf[14] = (float)tasdcalib_.sub[j].pchped[0];
		  buf[15] = (float)tasdcalib_.sub[j].lhpchped[1];
		  buf[16] = (float)tasdcalib_.sub[j].lhpchped[0];
		  buf[17] = (float)tasdcalib_.sub[j].rhpchped[1];
		  buf[18] = (float)tasdcalib_.sub[j].rhpchped[0];
		  buf[19] = (float)tasdcalib_.sub[j].mftndof[1];
		  buf[20] = (float)tasdcalib_.sub[j].mftndof[0];
		  buf[21] = tasdcalib_.sub[j].mftchi2[1];
		  buf[22] = tasdcalib_.sub[j].mftchi2[0];
		  buf[23] = tasdcalib_.sub[j].site;
		  buf[24]=0.0;
		  buf[25]=0.0;
		  for (i=0; i<satmax; i++)
		    {
		      if (taperfrom[i] <= ta_period && ta_period <= taperto[i] && k == sdnum[i])
			{
			  buf[24]=(float)lu_sat[i][0];
			  buf[25]=(float)lu_sat[i][1];
			  break;
			}
		    }
		  fwrite(buf, sizeof(float), NCAL, fout);
		}
	    }
	}
      dst_close_unit_(&dst_unit_in);
      if (rcode != END_OF_FILE) 
	{
	  fprintf(stderr,"Error reading %s\n",filename);
	  return -3;
	}
    }
  fclose(fout);
  return 0;
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
