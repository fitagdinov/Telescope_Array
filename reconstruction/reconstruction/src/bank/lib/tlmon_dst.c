/*
 * C functions for tlmon (TALE SD Monitoring Information)
 * Dmitri Ivanov, <dmiivanov@gmail.com>
 * Nov 22, 2019
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "tlmon_dst.h"



tlmon_dst_common tlmon_;	/* allocate memory to tlmon_common */

static integer4 tlmon_blen;
static integer4 tlmon_maxlen =
  sizeof (integer4) * 2 + sizeof (tlmon_dst_common);
static integer1 *tlmon_bank = NULL;

integer1* tlmon_bank_buffer_ (integer4* tlmon_bank_buffer_size)
{
  (*tlmon_bank_buffer_size) = tlmon_blen;
  return tlmon_bank;
}

static void tlmon_bank_init ()
{
  tlmon_bank = (integer1 *) calloc (tlmon_maxlen, sizeof (integer1));
  if (tlmon_bank == NULL)
    {
      fprintf (stderr,
	       "tlmon_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"tlmon_bank allocated memory %d\n",tlmon_maxlen); */
}

integer4 tlmon_common_to_bank_ ()
{
  static integer4 id = TLMON_BANKID, ver = TLMON_BANKVERSION;
  integer4 rcode, nobj, i, j;

  if (tlmon_bank == NULL)
    tlmon_bank_init ();

  rcode = dst_initbank_ (&id, &ver, &tlmon_blen, &tlmon_maxlen, tlmon_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = 1;

  rcode +=
    dst_packi4_ (&tlmon_.event_num, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.site, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  nobj=1;
  
  rcode +=
    dst_packi4_ (&tlmon_.run_id, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  
  nobj = TLMON_NCT;
  rcode +=
    dst_packi4_ (&tlmon_.run_num[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  
  nobj = 1;
  rcode +=
    dst_packi4_ (&tlmon_.errcode, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);

  rcode +=
    dst_packi4_ (&tlmon_.yymmddb, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.hhmmssb, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  
  rcode +=
    dst_packi4_ (&tlmon_.yymmdde, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.hhmmsse, &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);

  
  rcode +=
    dst_packi4_ (&tlmon_.lind, &nobj, tlmon_bank, &tlmon_blen, &tlmon_maxlen);

  nobj = tlmon_.lind + 1;	/* lind is the largest index in the monitoring cycle, 
				   hence # of detectors in a cycle = lind+1. */

  rcode +=
    dst_packi4_ (&tlmon_.xxyy[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);

  nobj = TLMON_NMONCHAN;
  for (i = 0; i <= tlmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_packi4_ (&tlmon_.hmip[i][j][0], &nobj, tlmon_bank,
			 &tlmon_blen, &tlmon_maxlen);
	}
    }


  nobj = TLMON_NMONCHAN / 2;		/* Ped histograms have half as many channels */

  for (i = 0; i <= tlmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_packi4_ (&tlmon_.hped[i][j][0], &nobj, tlmon_bank,
			 &tlmon_blen, &tlmon_maxlen);

	}
    }



  
  nobj = TLMON_NMONCHAN / 4;		/* Lin. histograms have 1/4 as many channels */

  for (i = 0; i <= tlmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_packi4_ (&tlmon_.hpht[i][j][0], &nobj, tlmon_bank,
			 &tlmon_blen, &tlmon_maxlen);
	  rcode +=
	    dst_packi4_ (&tlmon_.hpcg[i][j][0], &nobj, tlmon_bank,
			 &tlmon_blen, &tlmon_maxlen);
	}
    }


  nobj = 2;			/* For upper and lower counters */
  for (i = 0; i <= tlmon_.lind; i++)
    {
      rcode +=
	dst_packi4_ (&tlmon_.pchmip[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);

      rcode +=
	dst_packi4_ (&tlmon_.pchped[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);

      rcode +=
	dst_packi4_ (&tlmon_.lhpchmip[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);

      rcode +=
	dst_packi4_ (&tlmon_.lhpchped[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);


      rcode +=
	dst_packi4_ (&tlmon_.rhpchmip[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);

      rcode +=
	dst_packi4_ (&tlmon_.rhpchped[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
    }


  /* Status variables */

  for (i = 0; i <= tlmon_.lind; i++)
    {

      nobj = 600;
      
      rcode +=
	dst_packi4_ (&tlmon_.tgtblnum[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mclkcnt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen); 
      nobj = 10;
      
      /* CC */

      rcode +=
	dst_packi4_ (&tlmon_.ccadcbvt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.blankvl1[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.ccadcbct[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.blankvl2[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.ccadcrvt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.ccadcbtm[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.ccadcsvt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.ccadctmp[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);


      /* Mainboard */

      rcode +=
	dst_packi4_ (&tlmon_.mbadcgnd[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadcsdt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadc5vt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadcsdh[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadc33v[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadcbdt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadc18v[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.mbadc12v[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);


      /* Rate monitoring */

      rcode +=
	dst_packi4_ (&tlmon_.crminlv2[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packi4_ (&tlmon_.crminlv1[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);

    }

  /* GPS Monitoring */
  nobj = tlmon_.lind + 1;
  rcode +=
    dst_packi4_ (&tlmon_.gpsyymmdd[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.gpshhmmss[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.gpsflag[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.curtgrate[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  rcode +=
    dst_packi4_ (&tlmon_.num_sat[0], &nobj, tlmon_bank, &tlmon_blen,
		 &tlmon_maxlen);
  
  /* Results for 1MIP fitting */
  
  nobj=2;
  for(i=0;i<=tlmon_.lind;i++)
    {
      rcode +=
	dst_packi4_ (&tlmon_.mftndof[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packr8_ (&tlmon_.mip[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packr8_ (&tlmon_.mftchi2[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      nobj=4;
      for(j=0;j<2;j++)
	{
	  rcode +=
	    dst_packr8_ (&tlmon_.mftp[i][j][0], &nobj, tlmon_bank, &tlmon_blen,
			 &tlmon_maxlen);
	  rcode +=
	    dst_packr8_ (&tlmon_.mftpe[i][j][0], &nobj, tlmon_bank, &tlmon_blen,
			 &tlmon_maxlen);
	  
	}
      
      nobj = 3;
      rcode +=
	dst_packr8_ (&tlmon_.lat_lon_alt[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      rcode +=
	dst_packr8_ (&tlmon_.xyz_cor_clf[i][0], &nobj, tlmon_bank, &tlmon_blen,
		     &tlmon_maxlen);
      
    }

  return rcode;
}





integer4
tlmon_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &tlmon_blen, tlmon_bank);
  free (tlmon_bank);
  tlmon_bank = NULL;
  return rcode;
}

integer4
tlmon_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = tlmon_common_to_bank_ ()))
    {
      fprintf (stderr, "tlmon_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = tlmon_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "tlmon_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4
tlmon_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj, i, j;
  integer4 bankid, bankversion;

  tlmon_blen = 0;
  
  nobj = 1;
  
  rcode += dst_unpacki4_ (&bankid, &nobj, bank, &tlmon_blen, &tlmon_maxlen);
  rcode += dst_unpacki4_ (&bankversion, &nobj, bank, &tlmon_blen, &tlmon_maxlen);

  rcode +=
    dst_unpacki4_ (&tlmon_.event_num, &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmon_.site, &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  nobj=1;
  rcode +=
    dst_unpacki4_ (&tlmon_.run_id, &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  if(bankversion >= 1)
    {
      nobj = TLMON_NCT;
      rcode +=
	dst_unpacki4_ (&tlmon_.run_num[0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
    }
  else
    memset(&tlmon_.run_num[0],0,TLMON_NCT*sizeof(integer4));
  
  nobj = 1;
  rcode +=
    dst_unpacki4_ (&tlmon_.errcode, &nobj, bank, &tlmon_blen, &tlmon_maxlen);

  rcode +=
    dst_unpacki4_ (&tlmon_.yymmddb, &nobj, bank, &tlmon_blen, &tlmon_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmon_.hhmmssb, &nobj, bank, &tlmon_blen, &tlmon_maxlen);

  rcode +=
    dst_unpacki4_ (&tlmon_.yymmdde, &nobj, bank, &tlmon_blen, &tlmon_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmon_.hhmmsse, &nobj, bank, &tlmon_blen, &tlmon_maxlen);
  
  rcode +=
    dst_unpacki4_ (&tlmon_.lind, &nobj, bank, &tlmon_blen, &tlmon_maxlen);

  nobj = tlmon_.lind + 1;	/* lind is the largest index in the monitoring cycle, 
				   hence # of detectors in a cycle = lind+1. */
  
  rcode +=
    dst_unpacki4_ (&tlmon_.xxyy[0], &nobj, bank, &tlmon_blen, &tlmon_maxlen);

  nobj = TLMON_NMONCHAN;
  for (i = 0; i <= tlmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_unpacki4_ (&tlmon_.hmip[i][j][0], &nobj, bank, &tlmon_blen,
			   &tlmon_maxlen);
	}
    }



  nobj = TLMON_NMONCHAN / 2;		/* Ped histograms have half as many channels */
  
  for (i = 0; i <= tlmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_unpacki4_ (&tlmon_.hped[i][j][0], &nobj, bank,
			   &tlmon_blen, &tlmon_maxlen);

	}
    }



  
  nobj = TLMON_NMONCHAN / 4;		/* Lin. histograms have 1/4 as many channels */

  for (i = 0; i <= tlmon_.lind; i++)
    {
      for (j = 0; j < 2; j++)
	{
	  rcode +=
	    dst_unpacki4_ (&tlmon_.hpht[i][j][0], &nobj, bank,
			   &tlmon_blen, &tlmon_maxlen);
	  rcode +=
	    dst_unpacki4_ (&tlmon_.hpcg[i][j][0], &nobj, bank,
			   &tlmon_blen, &tlmon_maxlen);
	}
    }



  nobj = 2;			/* For upper and lower counters */
  for (i = 0; i <= tlmon_.lind; i++)
    {
      rcode +=
	dst_unpacki4_ (&tlmon_.pchmip[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);

      rcode +=
	dst_unpacki4_ (&tlmon_.pchped[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);

      rcode +=
	dst_unpacki4_ (&tlmon_.lhpchmip[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);

      rcode +=
	dst_unpacki4_ (&tlmon_.lhpchped[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);


      rcode +=
	dst_unpacki4_ (&tlmon_.rhpchmip[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);

      rcode +=
	dst_unpacki4_ (&tlmon_.rhpchped[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
    }


  /* Status variables */

  for (i = 0; i <= tlmon_.lind; i++)
    {
      
      nobj = 600;
      
      rcode +=
	dst_unpacki4_ (&tlmon_.tgtblnum[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mclkcnt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen); 
      nobj = 10;

      /* CC */

      rcode +=
	dst_unpacki4_ (&tlmon_.ccadcbvt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.blankvl1[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.ccadcbct[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.blankvl2[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.ccadcrvt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.ccadcbtm[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.ccadcsvt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.ccadctmp[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);


      /* Mainboard */

      rcode +=
	dst_unpacki4_ (&tlmon_.mbadcgnd[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadcsdt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadc5vt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadcsdh[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadc33v[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadcbdt[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadc18v[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.mbadc12v[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);


      /* Rate monitoring */

      rcode +=
	dst_unpacki4_ (&tlmon_.crminlv2[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpacki4_ (&tlmon_.crminlv1[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
    }

  /* GPS Monitoring */
  nobj = tlmon_.lind + 1;
  rcode +=
    dst_unpacki4_ (&tlmon_.gpsyymmdd[0], &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmon_.gpshhmmss[0], &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmon_.gpsflag[0], &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmon_.curtgrate[0], &nobj, bank, &tlmon_blen,
		   &tlmon_maxlen);
  if(bankversion >= 1)
    {
      rcode +=
        dst_unpacki4_ (&tlmon_.num_sat[0], &nobj, bank, &tlmon_blen,
                       &tlmon_maxlen);
    }
  else
    memset(&tlmon_.num_sat[0],0,nobj*sizeof(integer4));
  
  /* Results for 1MIP fitting */
  
  nobj=2;
  for(i=0;i<=tlmon_.lind;i++)
    {
      rcode +=
	dst_unpacki4_ (&tlmon_.mftndof[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpackr8_ (&tlmon_.mip[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      rcode +=
	dst_unpackr8_ (&tlmon_.mftchi2[i][0], &nobj, bank, &tlmon_blen,
		       &tlmon_maxlen);
      nobj=4;
      for(j=0;j<2;j++)
	{
	  rcode +=
	    dst_unpackr8_ (&tlmon_.mftp[i][j][0], &nobj, bank, &tlmon_blen,
			   &tlmon_maxlen);
	  rcode +=
	    dst_unpackr8_ (&tlmon_.mftpe[i][j][0], &nobj, bank, &tlmon_blen,
			   &tlmon_maxlen);
	  
	}
      if(bankversion >= 1)
	{
	  nobj = 3;
	  rcode +=
	    dst_unpackr8_ (&tlmon_.lat_lon_alt[i][0], &nobj, bank, &tlmon_blen,
			   &tlmon_maxlen);
	  rcode +=
	    dst_unpackr8_ (&tlmon_.xyz_cor_clf[i][0], &nobj, bank, &tlmon_blen,
			   &tlmon_maxlen);
	}
      else
	{
	  nobj = 3;
	  memset(&tlmon_.lat_lon_alt[i][0],0,3*sizeof(real8));
	  memset(&tlmon_.xyz_cor_clf[i][0],0,3*sizeof(real8));
	}
    }  
  return rcode;
}

integer4 tlmon_common_to_dump_ (integer4 * long_output)
{
  return tlmon_common_to_dumpf_ (stdout, long_output);
}


static char* tlmon_binrep(int n, int nbits)
{
  int i=0;
  static char buf[8*sizeof(int)];
  if (nbits > (int)(8*sizeof(int)-1))
    {
      sprintf(buf,"error: nbits > %d (max)",(int)(8*sizeof(int)-1));
      return buf;
    }
  buf[nbits] = '\0';
  for (i=0; i < nbits; i++)
    buf[nbits-1-i] = (n & (1<<i) ? '1' : '0');
  return buf;
}

integer4 tlmon_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i, j;
  fprintf (fp,"event_num %d site %s",tlmon_.event_num,tlmon_binrep(tlmon_.site,TALEX00_NCT));
  fprintf (fp, " run_id: %d err_code %d yymmddb %06d hhmmssb %06d yymmdde %06d hhmmsse %06d lind %d\n",
	   tlmon_.run_id,
	   tlmon_.errcode, tlmon_.yymmddb, tlmon_.hhmmssb,
	   tlmon_.yymmdde, tlmon_.hhmmsse,tlmon_.lind);
  
  if (*long_output == 0)
    {

      fprintf (fp, "%s",
	       "ind xxyy pchmip    pchped  lhpchmip lhpchped  rhpchmip rhpchped gpsyymmdd gpshhmmss gpsflag curtgrate num_sat\n");
      for (i = 0; i <= tlmon_.lind; i++)
	{
	  fprintf (fp, "%03d %04d", i, tlmon_.xxyy[i]);
	  fprintf (fp, "%4d,%3d%5d,%3d%5d,%3d%5d,%3d %5d,%3d%5d,%3d %8.06d %9.06d %8d %6d %10d\n",
		   tlmon_.pchmip[i][0], tlmon_.pchmip[i][1],
		   tlmon_.pchped[i][0], tlmon_.pchped[i][1],
		   tlmon_.lhpchmip[i][0], tlmon_.lhpchmip[i][1],
		   tlmon_.lhpchped[i][0], tlmon_.lhpchped[i][1],
		   tlmon_.rhpchmip[i][0], tlmon_.rhpchmip[i][1],
		   tlmon_.rhpchped[i][0], tlmon_.rhpchped[i][1],
		   tlmon_.gpsyymmdd[i],tlmon_.gpshhmmss[i],
		   tlmon_.gpsflag[i],tlmon_.curtgrate[i],tlmon_.num_sat[i]);
	}
    }

  else if (*long_output == 1)
    {
      for (i = 0; i <= tlmon_.lind; i++)
	{
	  fprintf (fp, "%s",
		   "ind xxyy pchmip    pchped  lhpchmip lhpchped  rhpchmip rhpchped gpsyymmdd gpshhmmss gpsflag curtgrate num_sat\n");
	  fprintf (fp, "%03d %04d", i, tlmon_.xxyy[i]);
	  fprintf (fp, "%4d,%3d%5d,%3d%5d,%3d%5d,%3d %5d,%3d%5d,%3d %8.06d %9.06d %8d %6d %10d\n",
		   tlmon_.pchmip[i][0], tlmon_.pchmip[i][1],
		   tlmon_.pchped[i][0], tlmon_.pchped[i][1],
		   tlmon_.lhpchmip[i][0], tlmon_.lhpchmip[i][1],
		   tlmon_.lhpchped[i][0], tlmon_.lhpchped[i][1],
		   tlmon_.rhpchmip[i][0], tlmon_.rhpchmip[i][1],
		   tlmon_.rhpchped[i][0], tlmon_.rhpchped[i][1],
		   tlmon_.gpsyymmdd[i],tlmon_.gpshhmmss[i],
		   tlmon_.gpsflag[i],tlmon_.curtgrate[i],tlmon_.num_sat[i]);
	  fprintf(fp,"HIST:\n");
	  fprintf(fp,"%16s %25s %27s %25s\n","MIP","PED","PHLIN","CHLIN");

	  fprintf(fp,"%s %12s %12s %12s %12s %12s %12s %12s %12s\n","BIN","CH0","CH1",
		  "CH0","CH1","CH0","CH1","CH0","CH1");


	  for(j=0;j<TLMON_NMONCHAN;j++)
	    {
	      fprintf(fp,"%03d %12d %12d",j,tlmon_.hmip[i][0][j],tlmon_.hmip[i][1][j]);
	      
	      if (j < (TLMON_NMONCHAN/2))
		{
		  fprintf(fp,"%12d %12d",
			  tlmon_.hped[i][0][j],tlmon_.hped[i][1][j]);
		}


	      if (j < (TLMON_NMONCHAN/4))
		{
		  fprintf(fp," %12d %12d %12d %12d",
			  tlmon_.hpht[i][0][j],tlmon_.hpht[i][1][j],
			  tlmon_.hpcg[i][0][j],tlmon_.hpcg[i][1][j]);
		}

	      
	      

	      
	      fprintf(fp,"\n");
	    }
	  

	}
      





      fprintf (fp,
	       "---------- OTHER INFORMATION (DET. STATUS VARIABLES) ------------- \n");



      for (i = 0; i <= tlmon_.lind; i++)
	{
	  fprintf (fp, "/////////// ind %03d xxyy %04d ////////////////\n", i, tlmon_.xxyy[i]);
	  
	  /* CC */
	  fprintf (fp, "ccadcbvt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.ccadcbvt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "blankvl1\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.blankvl1[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcbct\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.ccadcbct[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "blankvl2\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.blankvl2[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcrvt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.ccadcrvt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcbtm\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.ccadcbtm[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadcsvt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.ccadcsvt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "ccadctmp\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.ccadctmp[i][j]);
	  fprintf (fp, "\n");




	  /* MB */
	  fprintf (fp, "mbadcgnd\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadcgnd[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadcsdt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadcsdt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc5vt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadc5vt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadcsdh\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadcsdh[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc33v\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadc33v[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadcbdt\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadcbdt[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc18v\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadc18v[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "mbadc12v\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.mbadc12v[i][j]);
	  fprintf (fp, "\n");



	  /* RM */
	  fprintf (fp, "crminlv2\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.crminlv2[i][j]);
	  fprintf (fp, "\n");
	  fprintf (fp, "crminlv1\n");
	  for (j = 0; j < 10; j++)
	    fprintf (fp, " %d", tlmon_.crminlv1[i][j]);
	  fprintf (fp, "\n");

	  /* GPS */
	  fprintf(fp,"gpsyymmdd %06d gpshhmmss %06d gpsflag %d curtgrate %d\n",
		  tlmon_.gpsyymmdd[i],
		  tlmon_.gpshhmmss[i],
		  tlmon_.gpsflag[i],
		  tlmon_.curtgrate[i]
		  );
	}
      
      
    }

  return 0;
}
