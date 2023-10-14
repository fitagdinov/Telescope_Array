/*
 * C functions for sdgealib
 * Dmitri Ivanov, ivanov@physics.rutgers.edu
 * Mar 17, 2010
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "sdgealib_dst.h"

/* allocate memory for sdgealib structures */
sdgealib_head_struct sdgealib_head_;
sdgealib_pinf_struct sdgealib_pinf_;
sdgealib_hist_struct sdgealib_hist_;

static integer4 sdgealib_blen = 0;

// Using largest possible structures size used in the readout 
static integer4 sdgealib_maxlen = sizeof(integer4) * 2 + sizeof(sdgealib_head_struct) + sizeof(sdgealib_hist_struct);
static integer1 *sdgealib_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* sdgealib_bank_buffer_ (integer4* sdgealib_bank_buffer_size)
{
  (*sdgealib_bank_buffer_size) = sdgealib_blen;
  return sdgealib_bank;
}



static const char* sdgealib_corid2name(int corid)
{
  switch (corid)
    {
    // gamma
    case 1:
      return "gamma";
      break;

      // e+
    case 2:
      return "eplus";
      break;

      // -e
    case 3:
      return "eminus";
      break;

      // mu+
    case 5:
      return "muplus";
      break;

      // mu-
    case 6:
      return "muminus";
      break;

      // pi0
    case 7:
      return "pi0";
      break;

      // pi+
    case 8:
      return "piplus";
      break;

      // pi-
    case 9:
      return "piminus";
      break;

      // n
    case 13:
      return "neutron";
      break;

      // p
    case 14:
      return "proton";
      break;

      // pbar
    case 15:
      return "pbar";
      break;

      // nbar
    case 25:
      return "nbar";
      break;

    default:
      fprintf(stderr, "particle corid=%d is not supported\n", corid);
      return "";
      break;
    }
}

static void sdgealib_bank_init()
{
  sdgealib_bank = (integer1 *) calloc(sdgealib_maxlen, sizeof(integer1));
  if (sdgealib_bank == NULL)
    {
      fprintf(stderr, "sdgealib_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    } /* else fprintf ( stderr,"sdgealib_bank allocated memory %d\n",sdgealib_maxlen); */
}

// To pack objects of sdgealib_bin_struct
static integer4 pack_sdgealib_bins(sdgealib_bin_struct sdgealib_bin_obj[], integer4 *Nobj, integer1 Bank[],
    integer4 *LenBank, integer4 *MaxLen)
{
  integer4 i, nobj, rcode;
  rcode = 0;
  nobj = 1;
  for (i = 0; i < (*Nobj); i++)
    {
      rcode += dst_packi2_(&sdgealib_bin_obj[i].ix, &nobj, Bank, LenBank, MaxLen);
      rcode += dst_packi2_(&sdgealib_bin_obj[i].iy, &nobj, Bank, LenBank, MaxLen);
      rcode += dst_packi2_(&sdgealib_bin_obj[i].w, &nobj, Bank, LenBank, MaxLen);
    }
  return rcode;
}

// To unpack objects of sdgealib_bin_struct
static integer4 unpack_sdgealib_bins(sdgealib_bin_struct sdgealib_bin_obj[], integer4 *Nobj, integer1 Bank[],
    integer4 *LenBank, integer4 *MaxLen)
{
  integer4 i, nobj, rcode;
  rcode = 0;
  nobj = 1;
  for (i = 0; i < (*Nobj); i++)
    {
      rcode += dst_unpacki2_(&sdgealib_bin_obj[i].ix, &nobj, Bank, LenBank, MaxLen);
      rcode += dst_unpacki2_(&sdgealib_bin_obj[i].iy, &nobj, Bank, LenBank, MaxLen);
      rcode += dst_unpacki2_(&sdgealib_bin_obj[i].w, &nobj, Bank, LenBank, MaxLen);
    }
  return rcode;
}

integer4 sdgealib_common_to_bank_()
{
  static integer4 id = SDGEALIB_BANKID, ver = SDGEALIB_BANKVERSION;
  integer4 rcode, nobj, i, j;

  if (sdgealib_bank == NULL)
    sdgealib_bank_init();

  rcode = dst_initbank_(&id, &ver, &sdgealib_blen, &sdgealib_maxlen, sdgealib_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = 1;

  rcode += dst_packi4_(&sdgealib_head_.corid, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
  rcode += dst_packi4_(&sdgealib_head_.itype, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);

  if (sdgealib_head_.itype == SDGEALIB_PINF)
    {
      nobj = 1;

      rcode += dst_packi4_(&sdgealib_pinf_.nke, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_packi4_(&sdgealib_pinf_.nsectheta, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_packr8_(&sdgealib_pinf_.log10kemin, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_packr8_(&sdgealib_pinf_.log10kemax, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_packr8_(&sdgealib_pinf_.secthetamin, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_packr8_(&sdgealib_pinf_.secthetamax, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);

      for (i = 0; i < sdgealib_pinf_.nke; i++)
        {
          nobj = sdgealib_pinf_.nsectheta;
          rcode += dst_packr8_(&sdgealib_pinf_.peloss_kepc[i][0], &nobj, sdgealib_bank, &sdgealib_blen,
              &sdgealib_maxlen);
          rcode += dst_packr8_(&sdgealib_pinf_.peloss_sepc[i][0], &nobj, sdgealib_bank, &sdgealib_blen,
              &sdgealib_maxlen);
          nobj = 2;
          for (j = 0; j < sdgealib_pinf_.nsectheta; j++)
            {
              rcode += dst_packr8_(&sdgealib_pinf_.pkeloss_kepc[i][j][0], &nobj, sdgealib_bank, &sdgealib_blen,
                  &sdgealib_maxlen);
              rcode += dst_packr8_(&sdgealib_pinf_.pkeloss_sepc[i][j][0], &nobj, sdgealib_bank, &sdgealib_blen,
                  &sdgealib_maxlen);
            }
        }

    }
  else if (sdgealib_head_.itype == SDGEALIB_HIST)
    {
      nobj = 1;
      rcode += dst_packi4_(&sdgealib_hist_.ike, &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      nobj = sdgealib_pinf_.nsectheta;
      rcode += dst_packi4_(&sdgealib_hist_.nn0bins[0], &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_packr8_(&sdgealib_hist_.peloss[0], &nobj, sdgealib_bank, &sdgealib_blen, &sdgealib_maxlen);
      for (i = 0; i < sdgealib_pinf_.nsectheta; i++)
        {
          nobj = sdgealib_hist_.nn0bins[i];
          rcode += pack_sdgealib_bins(&sdgealib_hist_.bins[i][0], &nobj, sdgealib_bank, &sdgealib_blen,
              &sdgealib_maxlen);
        }
    }
  else
    {
      fprintf(stderr, "sdgealib: information type = %d is not recognized\n", sdgealib_head_.itype);
      rcode++;
    }

  return rcode;
}

integer4 sdgealib_bank_to_dst_(integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_(NumUnit, &sdgealib_blen, sdgealib_bank);
  free(sdgealib_bank);
  sdgealib_bank = NULL;
  return rcode;
}

integer4 sdgealib_common_to_dst_(integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = sdgealib_common_to_bank_()))
    {
      fprintf(stderr, "sdgealib_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  if ((rcode = sdgealib_bank_to_dst_(NumUnit)))
    {
      fprintf(stderr, "sdgealib_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  return 0;
}

integer4 sdgealib_bank_to_common_(integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj, i, j;
  sdgealib_blen = 2 * sizeof(integer4); /* skip id and version  */

  nobj = 1;

  rcode += dst_unpacki4_(&sdgealib_head_.corid, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
  rcode += dst_unpacki4_(&sdgealib_head_.itype, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);

  if (sdgealib_head_.itype == SDGEALIB_PINF)
    {
      nobj = 1;

      rcode += dst_unpacki4_(&sdgealib_pinf_.nke, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_unpacki4_(&sdgealib_pinf_.nsectheta, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_unpackr8_(&sdgealib_pinf_.log10kemin, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_unpackr8_(&sdgealib_pinf_.log10kemax, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_unpackr8_(&sdgealib_pinf_.secthetamin, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_unpackr8_(&sdgealib_pinf_.secthetamax, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);

      for (i = 0; i < sdgealib_pinf_.nke; i++)
        {
          nobj = sdgealib_pinf_.nsectheta;
          rcode += dst_unpackr8_(&sdgealib_pinf_.peloss_kepc[i][0], &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
          rcode += dst_unpackr8_(&sdgealib_pinf_.peloss_sepc[i][0], &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
          nobj = 2;
          for (j = 0; j < sdgealib_pinf_.nsectheta; j++)
            {
              rcode += dst_unpackr8_(&sdgealib_pinf_.pkeloss_kepc[i][j][0], &nobj, bank, &sdgealib_blen,
                  &sdgealib_maxlen);
              rcode += dst_unpackr8_(&sdgealib_pinf_.pkeloss_sepc[i][j][0], &nobj, bank, &sdgealib_blen,
                  &sdgealib_maxlen);
            }
        }

    }
  else if (sdgealib_head_.itype == SDGEALIB_HIST)
    {
      nobj = 1;
      rcode += dst_unpacki4_(&sdgealib_hist_.ike, &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      nobj = sdgealib_pinf_.nsectheta;
      rcode += dst_unpacki4_(&sdgealib_hist_.nn0bins[0], &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      rcode += dst_unpackr8_(&sdgealib_hist_.peloss[0], &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
      for (i = 0; i < sdgealib_pinf_.nsectheta; i++)
        {
          nobj = sdgealib_hist_.nn0bins[i];
          rcode += unpack_sdgealib_bins(&sdgealib_hist_.bins[i][0], &nobj, bank, &sdgealib_blen, &sdgealib_maxlen);
        }
    }
  else
    {
      fprintf(stderr, "sdgealib: information type = %d is not recognized\n", sdgealib_head_.itype);
      rcode++;
    }

  return rcode;
}

integer4 sdgealib_common_to_dump_(integer4 * long_output)
{
  return sdgealib_common_to_dumpf_(stdout, long_output);
}

integer4 sdgealib_common_to_dumpf_(FILE * fp, integer4 * long_output)
{
  int ike, isectheta, ib;
  static integer4 particle_corid = -1;
  fprintf(fp, "%s :\n", "sdgealib");
  fprintf(fp, "particle: %s\n", sdgealib_corid2name(sdgealib_head_.corid));
  fprintf(fp, "information type: ");
  if (sdgealib_head_.itype == SDGEALIB_PINF)
    {
      fprintf(fp, "particle information\n");
      fprintf(fp, "number of log10(K.E./MeV) bins: %d\n", sdgealib_pinf_.nke);
      fprintf(fp, "log10(K.E./MeV)_min = %.2f\n", sdgealib_pinf_.log10kemin);
      fprintf(fp, "log10(K.E./MeV)_max = %.2f\n", sdgealib_pinf_.log10kemax);
      fprintf(fp, "number of sec(theta) bins: %d\n", sdgealib_pinf_.nsectheta);
      fprintf(fp, "sec(theta)_min: %.2f\n", sdgealib_pinf_.secthetamin);
      fprintf(fp, "sec(theta)_max: %.2f\n", sdgealib_pinf_.secthetamax);
      if (sdgealib_head_.corid == particle_corid)
        fprintf(stderr, "Warning (sdgealib): particle information block was written twice\n");
      else
        particle_corid = sdgealib_head_.corid;
      
      if ((*long_output) == 1)
	{
	  fprintf(fp, "%s\t%s\t%s\t%s\t%20s\t%25s\n",
		  "ike","isectheta",
		  "peloss_kepc","peloss_sepc",
		  "pkeloss_kepc (L,U)","pkeloss_sepc (L,U)");
	  
	  for (ike = 0; ike < sdgealib_pinf_.nke; ike ++)
	    {
	      for ( isectheta = 0; isectheta < sdgealib_pinf_.nsectheta; isectheta++ ) 
		{
		  fprintf(fp,"%02d\t%6.02d\t%18.3e\t%10.3e\t%10.3e%12.3e\t%15.3e%12.3e\n",
			  ike,isectheta,
			  sdgealib_pinf_.peloss_kepc[ike][isectheta],sdgealib_pinf_.peloss_sepc[ike][isectheta],
			  sdgealib_pinf_.pkeloss_kepc[ike][isectheta][0],sdgealib_pinf_.pkeloss_kepc[ike][isectheta][1],
			  sdgealib_pinf_.pkeloss_sepc[ike][isectheta][0],sdgealib_pinf_.pkeloss_sepc[ike][isectheta][1]
			  );
		}
	    }
	}
    }
  else if (sdgealib_head_.itype == SDGEALIB_HIST)
    {
      if (sdgealib_head_.corid != particle_corid)
        fprintf(stderr, "Warning (sdgealib): particle histograms were written before the particle information block\n");
      fprintf(fp, "energy loss histograms ike = %d\n", sdgealib_hist_.ike);
      for (isectheta = 0; isectheta < sdgealib_pinf_.nsectheta; isectheta++)
        {
          fprintf(fp, "log10(K.E./MeV) = %.2f  sec(theta) = %.2f", sdgealib_pinf_.log10kemin
              + ((real8) sdgealib_hist_.ike + 0.5) * (sdgealib_pinf_.log10kemax - sdgealib_pinf_.log10kemin)
                  / ((real8) sdgealib_pinf_.nke), sdgealib_pinf_.secthetamin + ((real8) isectheta + 0.5)
              * (sdgealib_pinf_.secthetamax - sdgealib_pinf_.secthetamin) / ((real8) sdgealib_pinf_.nsectheta));
          fprintf(fp, " peloss = %.3e nn0bins = %d", sdgealib_hist_.peloss[isectheta],
              sdgealib_hist_.nn0bins[isectheta]);
	  fprintf(fp, "\n");
        }
      if ((*long_output) == 1)
        {
          for (isectheta = 0; isectheta < sdgealib_pinf_.nsectheta; isectheta++)
            {
              fprintf(fp, "ike = %d isectheta = %d", sdgealib_hist_.ike, isectheta);
	      fprintf(fp, " log10(K.E./MeV) = %.2f  sec(theta) = %.2f", sdgealib_pinf_.log10kemin
		      + ((real8) sdgealib_hist_.ike + 0.5) * (sdgealib_pinf_.log10kemax - sdgealib_pinf_.log10kemin)
		      / ((real8) sdgealib_pinf_.nke), sdgealib_pinf_.secthetamin + ((real8) isectheta + 0.5)
		      * (sdgealib_pinf_.secthetamax - sdgealib_pinf_.secthetamin) / ((real8) sdgealib_pinf_.nsectheta));
	      fprintf(fp, "\n");
              for (ib = 0; ib < sdgealib_hist_.nn0bins[isectheta]; ib++)
                {
                  fprintf(fp, "[%03d,%03d,%05d]", (int) sdgealib_hist_.bins[isectheta][ib].ix,
			  (int) sdgealib_hist_.bins[isectheta][ib].iy, (int) sdgealib_hist_.bins[isectheta][ib].w);
                  if ((ib + 1) % 10 == 0)
                    fprintf(fp, "\n");
                }
	      if ((ib+1) %10 != 1)
		fprintf(fp, "\n");
            }
        }
    }
  else
    fprintf(fp, "unknown\n");

  return 0;
}
