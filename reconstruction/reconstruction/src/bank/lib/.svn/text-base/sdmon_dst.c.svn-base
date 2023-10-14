/*
 * C functions for sdmon (SD Monitoring Information)
 * Dmitri Ivanov, dmiivanov@gmail.com
 * Apr 30, 2019
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "sdmon_dst.h"



sdmon_dst_common sdmon_;	/* allocate memory to sdmon_common */

static integer4 sdmon_blen = 0;
static integer4 sdmon_maxlen =
  sizeof (integer4) * 2 + sizeof (sdmon_dst_common);
static integer1 *sdmon_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* sdmon_bank_buffer_ (integer4* sdmon_bank_buffer_size)
{
  (*sdmon_bank_buffer_size) = sdmon_blen;
  return sdmon_bank;
}



static void
sdmon_bank_init ()
{
  sdmon_bank = (integer1 *) calloc (sdmon_maxlen, sizeof (integer1));
  if (sdmon_bank == NULL)
    {
      fprintf (stderr,
	       "sdmon_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"sdmon_bank allocated memory %d\n",sdmon_maxlen); */
}

integer4 sdmon_common_to_bank_ ()
{
  static integer4 id = SDMON_BANKID, ver = SDMON_BANKVERSION;
  integer4 rcode, nobj, i, j;

  if (sdmon_bank == NULL)
    sdmon_bank_init ();

  rcode = dst_initbank_ (&id, &ver, &sdmon_blen, &sdmon_maxlen, sdmon_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = 1;

  rcode +=
    dst_packi4_ (&sdmon_.event_num, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  rcode +=
    dst_packi4_ (&sdmon_.site, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  nobj=3;
  
  rcode +=
    dst_packi4_ (&sdmon_.run_id[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  nobj=1;
  
  rcode +=
    dst_packi4_ (&sdmon_.errcode, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);

  rcode +=
    dst_packi4_ (&sdmon_.yymmddb, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  rcode +=
    dst_packi4_ (&sdmon_.hhmmssb, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  
  rcode +=
    dst_packi4_ (&sdmon_.yymmdde, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  rcode +=
    dst_packi4_ (&sdmon_.hhmmsse, &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);

  
  rcode +=
    dst_packi4_ (&sdmon_.lind, &nobj, sdmon_bank, &sdmon_blen, &sdmon_maxlen);

  nobj = sdmon_.lind + 1;	/* lind is the largest index in the monitoring cycle, 
				   hence # of detectors in a cycle = lind+1. */

  rcode +=
    dst_packi4_ (&sdmon_.xxyy[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);

  nobj = SDMON_NMONCHAN;
  for (i = 0; i <= sdmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_packi4_ (&sdmon_.hmip[i][j][0], &nobj, sdmon_bank,
			 &sdmon_blen, &sdmon_maxlen);
	}
    }


  nobj = SDMON_NMONCHAN / 2;		/* Ped histograms have half as many channels */

  for (i = 0; i <= sdmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_packi4_ (&sdmon_.hped[i][j][0], &nobj, sdmon_bank,
			 &sdmon_blen, &sdmon_maxlen);

	}
    }



  
  nobj = SDMON_NMONCHAN / 4;		/* Lin. histograms have 1/4 as many channels */

  for (i = 0; i <= sdmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_packi4_ (&sdmon_.hpht[i][j][0], &nobj, sdmon_bank,
			 &sdmon_blen, &sdmon_maxlen);
	  rcode +=
	    dst_packi4_ (&sdmon_.hpcg[i][j][0], &nobj, sdmon_bank,
			 &sdmon_blen, &sdmon_maxlen);
	}
    }


  nobj = 2;			/* For upper and lower counters */
  for (i = 0; i <= sdmon_.lind; i++)
    {
      rcode +=
	dst_packi4_ (&sdmon_.pchmip[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);

      rcode +=
	dst_packi4_ (&sdmon_.pchped[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);

      rcode +=
	dst_packi4_ (&sdmon_.lhpchmip[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);

      rcode +=
	dst_packi4_ (&sdmon_.lhpchped[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);


      rcode +=
	dst_packi4_ (&sdmon_.rhpchmip[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);

      rcode +=
	dst_packi4_ (&sdmon_.rhpchped[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
    }


  /* Status variables */

  for (i = 0; i <= sdmon_.lind; i++)
    {

      nobj = 600;
      
      rcode +=
	dst_packi4_ (&sdmon_.tgtblnum[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mclkcnt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen); 
      nobj = 10;
      
      /* CC */

      rcode +=
	dst_packi4_ (&sdmon_.ccadcbvt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.blankvl1[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.ccadcbct[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.blankvl2[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.ccadcrvt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.ccadcbtm[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.ccadcsvt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.ccadctmp[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);


      /* Mainboard */

      rcode +=
	dst_packi4_ (&sdmon_.mbadcgnd[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadcsdt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadc5vt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadcsdh[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadc33v[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadcbdt[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadc18v[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.mbadc12v[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);


      /* Rate monitoring */

      rcode +=
	dst_packi4_ (&sdmon_.crminlv2[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packi4_ (&sdmon_.crminlv1[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);

    }

  /* GPS Monitoring */
  nobj = sdmon_.lind + 1;
  rcode +=
    dst_packi4_ (&sdmon_.gpsyymmdd[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  rcode +=
    dst_packi4_ (&sdmon_.gpshhmmss[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  rcode +=
    dst_packi4_ (&sdmon_.gpsflag[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  rcode +=
    dst_packi4_ (&sdmon_.curtgrate[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);

  rcode +=
    dst_packi4_ (&sdmon_.num_sat[0], &nobj, sdmon_bank, &sdmon_blen,
		 &sdmon_maxlen);
  
  /* Results for 1MIP fitting */
  
  nobj=2;
  for(i=0;i<=sdmon_.lind;i++)
    {
      rcode +=
	dst_packi4_ (&sdmon_.mftndof[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packr8_ (&sdmon_.mip[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      rcode +=
	dst_packr8_ (&sdmon_.mftchi2[i][0], &nobj, sdmon_bank, &sdmon_blen,
		     &sdmon_maxlen);
      nobj=4;
      for(j=0;j<2;j++)
	{
	  rcode +=
	    dst_packr8_ (&sdmon_.mftp[i][j][0], &nobj, sdmon_bank, &sdmon_blen,
			 &sdmon_maxlen);
	  rcode +=
	    dst_packr8_ (&sdmon_.mftpe[i][j][0], &nobj, sdmon_bank, &sdmon_blen,
			 &sdmon_maxlen);
	  
	}
      
    }

  return rcode;
}





integer4 sdmon_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &sdmon_blen, sdmon_bank);
  free (sdmon_bank);
  sdmon_bank = NULL;
  return rcode;
}

integer4 sdmon_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = sdmon_common_to_bank_ ()))
    {
      fprintf (stderr, "sdmon_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = sdmon_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "sdmon_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4 sdmon_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj, i, j;
  integer4 bankid, bankversion;
  
  sdmon_blen = 0;
  
  nobj = 1;
  
  rcode += dst_unpacki4_ (&bankid, &nobj, bank, &sdmon_blen, &sdmon_maxlen);
  rcode += dst_unpacki4_ (&bankversion, &nobj, bank, &sdmon_blen, &sdmon_maxlen);

  rcode +=
    dst_unpacki4_ (&sdmon_.event_num, &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  rcode +=
    dst_unpacki4_ (&sdmon_.site, &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  nobj=3;
  rcode +=
    dst_unpacki4_ (&sdmon_.run_id[0], &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  nobj=1;
  rcode +=
    dst_unpacki4_ (&sdmon_.errcode, &nobj, bank, &sdmon_blen, &sdmon_maxlen);

  rcode +=
    dst_unpacki4_ (&sdmon_.yymmddb, &nobj, bank, &sdmon_blen, &sdmon_maxlen);
  rcode +=
    dst_unpacki4_ (&sdmon_.hhmmssb, &nobj, bank, &sdmon_blen, &sdmon_maxlen);

  rcode +=
    dst_unpacki4_ (&sdmon_.yymmdde, &nobj, bank, &sdmon_blen, &sdmon_maxlen);
  rcode +=
    dst_unpacki4_ (&sdmon_.hhmmsse, &nobj, bank, &sdmon_blen, &sdmon_maxlen);
  
  rcode +=
    dst_unpacki4_ (&sdmon_.lind, &nobj, bank, &sdmon_blen, &sdmon_maxlen);

  nobj = sdmon_.lind + 1;	/* lind is the largest index in the monitoring cycle, 
				   hence # of detectors in a cycle = lind+1. */
  
  rcode +=
    dst_unpacki4_ (&sdmon_.xxyy[0], &nobj, bank, &sdmon_blen, &sdmon_maxlen);

  nobj = SDMON_NMONCHAN;
  for (i = 0; i <= sdmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_unpacki4_ (&sdmon_.hmip[i][j][0], &nobj, bank, &sdmon_blen,
			   &sdmon_maxlen);
	}
    }



  nobj = SDMON_NMONCHAN / 2;		/* Ped histograms have half as many channels */
  
  for (i = 0; i <= sdmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_unpacki4_ (&sdmon_.hped[i][j][0], &nobj, bank,
			   &sdmon_blen, &sdmon_maxlen);

	}
    }



  
  nobj = SDMON_NMONCHAN / 4;		/* Lin. histograms have 1/4 as many channels */

  for (i = 0; i <= sdmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_unpacki4_ (&sdmon_.hpht[i][j][0], &nobj, bank,
			   &sdmon_blen, &sdmon_maxlen);
	  rcode +=
	    dst_unpacki4_ (&sdmon_.hpcg[i][j][0], &nobj, bank,
			   &sdmon_blen, &sdmon_maxlen);
	}
    }



  nobj = 2;			/* For upper and lower counters */
  for (i = 0; i <= sdmon_.lind; i++)
    {
      rcode +=
	dst_unpacki4_ (&sdmon_.pchmip[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);

      rcode +=
	dst_unpacki4_ (&sdmon_.pchped[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);

      rcode +=
	dst_unpacki4_ (&sdmon_.lhpchmip[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);

      rcode +=
	dst_unpacki4_ (&sdmon_.lhpchped[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);


      rcode +=
	dst_unpacki4_ (&sdmon_.rhpchmip[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);

      rcode +=
	dst_unpacki4_ (&sdmon_.rhpchped[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
    }


  /* Status variables */

  for (i = 0; i <= sdmon_.lind; i++)
    {
      
      nobj = 600;
      
      rcode +=
	dst_unpacki4_ (&sdmon_.tgtblnum[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mclkcnt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen); 
      nobj = 10;

      /* CC */

      rcode +=
	dst_unpacki4_ (&sdmon_.ccadcbvt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.blankvl1[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.ccadcbct[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.blankvl2[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.ccadcrvt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.ccadcbtm[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.ccadcsvt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.ccadctmp[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);


      /* Mainboard */

      rcode +=
	dst_unpacki4_ (&sdmon_.mbadcgnd[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadcsdt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadc5vt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadcsdh[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadc33v[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadcbdt[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadc18v[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.mbadc12v[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);


      /* Rate monitoring */

      rcode +=
	dst_unpacki4_ (&sdmon_.crminlv2[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpacki4_ (&sdmon_.crminlv1[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
    }

  /* GPS Monitoring */
  nobj = sdmon_.lind + 1;
  rcode +=
    dst_unpacki4_ (&sdmon_.gpsyymmdd[0], &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  rcode +=
    dst_unpacki4_ (&sdmon_.gpshhmmss[0], &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  rcode +=
    dst_unpacki4_ (&sdmon_.gpsflag[0], &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  rcode +=
    dst_unpacki4_ (&sdmon_.curtgrate[0], &nobj, bank, &sdmon_blen,
		   &sdmon_maxlen);
  if(bankversion >= 1)
    {
      rcode +=
	dst_unpacki4_ (&sdmon_.num_sat[0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
    }

  /* Results for 1MIP fitting */
  
  nobj=2;
  for(i=0;i<=sdmon_.lind;i++)
    {
      rcode +=
	dst_unpacki4_ (&sdmon_.mftndof[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpackr8_ (&sdmon_.mip[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      rcode +=
	dst_unpackr8_ (&sdmon_.mftchi2[i][0], &nobj, bank, &sdmon_blen,
		       &sdmon_maxlen);
      nobj=4;
      for(j=0;j<2;j++)
	{
	  rcode +=
	    dst_unpackr8_ (&sdmon_.mftp[i][j][0], &nobj, bank, &sdmon_blen,
			   &sdmon_maxlen);
	  rcode +=
	    dst_unpackr8_ (&sdmon_.mftpe[i][j][0], &nobj, bank, &sdmon_blen,
			   &sdmon_maxlen);
	  
	}
      
    }
  
  return rcode;
}

integer4
sdmon_common_to_dump_ (integer4 * long_output)
{
  return sdmon_common_to_dumpf_ (stdout, long_output);
}

integer4
sdmon_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i, j;
  fprintf (fp,"event_num %d site ",sdmon_.event_num);
  switch(sdmon_.site)
    {
    case SDMON_BR:
      fprintf(fp,"BR ");
      break;
    case SDMON_LR:
      fprintf(fp,"LR ");
      break;
    case SDMON_SK:
      fprintf(fp,"SK ");
      break;
    case SDMON_BRLR:
      fprintf(fp,"BRLR ");
      break;
    case SDMON_BRSK:
      fprintf(fp,"BRSK ");
      break;
    case SDMON_LRSK:
      fprintf(fp,"LRSK ");
      break;
    case SDMON_BRLRSK:
      fprintf(fp,"BRLRSK ");
      break;
    default:
      fprintf(fp,"?? ");
      break;
    }
  fprintf(fp,"run_id: %d %d %d ",sdmon_.run_id[0],sdmon_.run_id[1],sdmon_.run_id[2]);
  fprintf (fp, "err_code %d yymmddb %06d hhmmssb %06d yymmdde %06d hhmmsse %06d lind %d\n",
	   sdmon_.errcode, sdmon_.yymmddb, sdmon_.hhmmssb,
	   sdmon_.yymmdde, sdmon_.hhmmsse,sdmon_.lind);



  if (*long_output == 0)
    {

      fprintf (fp, "%s",
	       "ind xxyy pchmip    pchped  lhpchmip lhpchped  rhpchmip rhpchped gpsyymmdd gpshhmmss gpsflag curtgrate\n");
      for (i = 0; i <= sdmon_.lind; i++)
	{
	  fprintf (fp, "%03d %04d", i, sdmon_.xxyy[i]);
	  fprintf (fp, "%4d,%3d%5d,%3d%5d,%3d%5d,%3d %5d,%3d%5d,%3d %8.06d %9.06d %8d %6d\n",
		   sdmon_.pchmip[i][0], sdmon_.pchmip[i][1],
		   sdmon_.pchped[i][0], sdmon_.pchped[i][1],
		   sdmon_.lhpchmip[i][0], sdmon_.lhpchmip[i][1],
		   sdmon_.lhpchped[i][0], sdmon_.lhpchped[i][1],
		   sdmon_.rhpchmip[i][0], sdmon_.rhpchmip[i][1],
		   sdmon_.rhpchped[i][0], sdmon_.rhpchped[i][1],
		   sdmon_.gpsyymmdd[i],sdmon_.gpshhmmss[i],
		   sdmon_.gpsflag[i],sdmon_.curtgrate[i]);
	}
    }

  else if (*long_output == 1)
    {


      for (i = 0; i <= sdmon_.lind; i++)
	{
	  fprintf (fp, "%s",
		   "ind xxyy pchmip    pchped  lhpchmip lhpchped  rhpchmip rhpchped gpsyymmdd gpshhmmss gpsflag curtgrate\n");
	  fprintf (fp, "%03d %04d", i, sdmon_.xxyy[i]);
	  fprintf (fp, "%4d,%3d%5d,%3d%5d,%3d%5d,%3d %5d,%3d%5d,%3d %8.06d %9.06d %8d %6d\n",
		   sdmon_.pchmip[i][0], sdmon_.pchmip[i][1],
		   sdmon_.pchped[i][0], sdmon_.pchped[i][1],
		   sdmon_.lhpchmip[i][0], sdmon_.lhpchmip[i][1],
		   sdmon_.lhpchped[i][0], sdmon_.lhpchped[i][1],
		   sdmon_.rhpchmip[i][0], sdmon_.rhpchmip[i][1],
		   sdmon_.rhpchped[i][0], sdmon_.rhpchped[i][1],
		   sdmon_.gpsyymmdd[i],sdmon_.gpshhmmss[i],
		   sdmon_.gpsflag[i],sdmon_.curtgrate[i]);
	  fprintf(fp,"HIST:\n");
	  fprintf(fp,"%16s %25s %27s %25s\n","MIP","PED","PHLIN","CHLIN");

	  fprintf(fp,"%s %12s %12s %12s %12s %12s %12s %12s %12s\n","BIN","CH0","CH1",
		  "CH0","CH1","CH0","CH1","CH0","CH1");


	  for(j=0;j<SDMON_NMONCHAN;j++)
	    {
	      fprintf(fp,"%03d %12d %12d",j,sdmon_.hmip[i][0][j],sdmon_.hmip[i][1][j]);
	      
	      if (j < (SDMON_NMONCHAN/2))
		{
		  fprintf(fp,"%12d %12d",
			  sdmon_.hped[i][0][j],sdmon_.hped[i][1][j]);
		}


	      if (j < (SDMON_NMONCHAN/4))
		{
		  fprintf(fp," %12d %12d %12d %12d",
			  sdmon_.hpht[i][0][j],sdmon_.hpht[i][1][j],
			  sdmon_.hpcg[i][0][j],sdmon_.hpcg[i][1][j]);
		}

	      
	      

	      
	      fprintf(fp,"\n");
	    }
	  

	}
      





      fprintf (fp,
	       "---------- OTHER INFORMATION (DET. STATUS VARIABLES) ------------- \n");



      for (i = 0; i <= sdmon_.lind; i++)
	{
	  fprintf (fp, "/////////// ind %03d xxyy %04d ////////////////\n", i, sdmon_.xxyy[i]);
	  
	  /* CC */
	  fprintf (fp, "ccadcbvt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.ccadcbvt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "blankvl1\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.blankvl1[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcbct\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.ccadcbct[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "blankvl2\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.blankvl2[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcrvt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.ccadcrvt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcbtm\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.ccadcbtm[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcsvt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.ccadcsvt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadctmp\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.ccadctmp[i][j]);
	  fprintf (fp, "\n");




	  /* MB */
	  fprintf (fp, "mbadcgnd\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadcgnd[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadcsdt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadcsdt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc5vt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadc5vt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadcsdh\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadcsdh[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc33v\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadc33v[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadcbdt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadcbdt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc18v\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadc18v[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc12v\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.mbadc12v[i][j]);
	  fprintf (fp, "\n");



	  /* RM */
	  fprintf (fp, "crminlv2\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.crminlv2[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "crminlv1\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", sdmon_.crminlv1[i][j]);
	  fprintf (fp, "\n");

	  /* GPS */
	  fprintf(fp,"gpsyymmdd %06d gpshhmmss %06d gpsflag %d curtgrate %d\n",
		  sdmon_.gpsyymmdd[i],
		  sdmon_.gpshhmmss[i],
		  sdmon_.gpsflag[i],
		  sdmon_.curtgrate[i]
		  );
	}
      
      
    }

  return 0;
}
