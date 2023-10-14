/*
 * $Source: /hires_soft/uvm2k/bank/ftrg1_dst.c,v $
 * $Log: ftrg1_dst.c,v $
 * Revision 1.12  2003/05/28 15:30:57  hires
 * update to nevis version to eliminate tar file dependency (boyer)
 *
 * Revision 1.10  1996/09/27  16:53:48  boyer
 * add discarded triggers to bank
 *
 * Revision 1.9  1996/08/20  20:31:56  boyer
 * change dump to V-trig and H-trig (from row and col)
 *
 * Revision 1.8  1996/06/07  15:41:42  boyer
 * new trig packet structure
 *
 * Revision 1.4  1996/02/23  18:26:18  boyer
 * change common block a bit to conform with real data
 *
 * Revision 1.3  1995/10/09  16:39:22  boyer
 * Add a missed variable to pack/unpack
 *
 * Revision 1.2  1995/10/04  19:43:00  boyer
 * reformat dump
 *
 * Revision 1.1  1995/08/09  20:17:01  boyer
 * Initial revision
 *
 * Created by CPH 11:15 7/19/95
 * *** empty log message ***
 *
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "ftrg1_dst.h"  

ftrg1_dst_common ftrg1_;  /* allocate memory to ftrg1_dst_common */
ftrg1_info_common ftrg1_info_;  /* allocate memory to ftrg1_info_common */

static integer4 ftrg1_blen = 0; 
static integer4 ftrg1_maxlen = sizeof(integer4) * 2 + sizeof(ftrg1_dst_common);
static integer1 *ftrg1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* ftrg1_bank_buffer_ (integer4* ftrg1_bank_buffer_size)
{
  (*ftrg1_bank_buffer_size) = ftrg1_blen;
  return ftrg1_bank;
}



static void ftrg1_bank_init()
{
  ftrg1_bank = (integer1 *)calloc(ftrg1_maxlen, sizeof(integer1));
  if (ftrg1_bank==NULL)
    {
      fprintf (stderr,"ftrg1_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}    

integer4 ftrg1_common_to_bank_()
{
  static integer4 id = FTRG1_BANKID, ver = FTRG1_BANKVERSION;
  integer4 i, rcode, nobj;
  
  if (ftrg1_bank == NULL) ftrg1_bank_init();
    
  rcode = dst_initbank_(&id, &ver, &ftrg1_blen, &ftrg1_maxlen, ftrg1_bank);
   /* Initialize ftrg1_blen, and pack the id and version to bank */

  nobj = 1;

  rcode += dst_packi2_(&ftrg1_.num_mir, &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
  rcode += dst_packi2_(&ftrg1_.num_discard, &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);

  nobj = ftrg1_.num_mir;

    rcode += dst_packi2_(&ftrg1_.trig_code[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.mir_num[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);  
    rcode += dst_packi2_(&ftrg1_.nhit_dsp[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);  
    rcode += dst_packi2_(&ftrg1_.t_pld_start[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.t_pld_end[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.row_pattern[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.col_pattern[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.delay[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);

  nobj = ftrg1_.num_discard;

    rcode += dst_packi2_(&ftrg1_.discard_code[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.discard_mirid[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi4_(&ftrg1_.discard_clkcnt[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.discard_vpattern[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.discard_hpattern[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.discard_nbyte[0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);

  for(i=0;i<ftrg1_.num_mir;i++) {
    nobj = ftrg1_.nhit_dsp[i];
    rcode += dst_packi2_(&ftrg1_.ichan_dsp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.it0_dsp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.nt_dsp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.tav_dsp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.sigt_dsp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_packi2_(&ftrg1_.nadc_dsp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
    nobj = 16;
    rcode += dst_packi2_(&ftrg1_.wp[i][0], &nobj, ftrg1_bank, &ftrg1_blen, &ftrg1_maxlen);
  }

  return rcode ;
}

integer4 ftrg1_bank_to_dst_ (integer4 *unit)
{
  return dst_write_bank_(unit, &ftrg1_blen, ftrg1_bank);
}

integer4 ftrg1_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = ftrg1_common_to_bank_()) )
    {
      fprintf(stderr, "ftrg1_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
    if ( (rcode = ftrg1_bank_to_dst_(unit) ))
    {
      fprintf(stderr, "ftrg1_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  return 0;
}

integer4 ftrg1_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0 ;
  integer4 i, nobj ;
  ftrg1_blen = 2 * sizeof(integer4);	/* skip id and version  */

  nobj = 1;

  rcode += dst_unpacki2_(&ftrg1_.num_mir, &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
  rcode += dst_unpacki2_(&ftrg1_.num_discard, &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);

  nobj = ftrg1_.num_mir;

    rcode += dst_unpacki2_(&ftrg1_.trig_code[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.mir_num[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.nhit_dsp[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.t_pld_start[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.t_pld_end[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.row_pattern[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.col_pattern[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.delay[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);

  nobj = ftrg1_.num_discard;

    rcode += dst_unpacki2_(&ftrg1_.discard_code[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.discard_mirid[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki4_(&ftrg1_.discard_clkcnt[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.discard_vpattern[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.discard_hpattern[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.discard_nbyte[0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);

  for(i=0;i<ftrg1_.num_mir;i++) {
    nobj = ftrg1_.nhit_dsp[i];
    rcode += dst_unpacki2_(&ftrg1_.ichan_dsp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.it0_dsp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.nt_dsp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.tav_dsp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.sigt_dsp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    rcode += dst_unpacki2_(&ftrg1_.nadc_dsp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
    nobj=16;
    rcode += dst_unpacki2_(&ftrg1_.wp[i][0], &nobj, bank, &ftrg1_blen, &ftrg1_maxlen);
  } 
  return rcode ;
}

integer4 ftrg1_common_to_dump_(integer4 *long_output)
{
  return ftrg1_common_to_dumpf_(stdout,long_output);
}

integer4 ftrg1_common_to_dumpf_(FILE* fp,integer4 *long_output)
{
  int i;
  int j;

  fprintf (fp, "Ftrg1_");
  fprintf (fp, " num_mir %4d\n",ftrg1_.num_mir);
 
  if (*long_output==0) 
    {
    for (i = 0; i < ftrg1_.num_mir; i++)
      {
	fprintf (fp, "mirror %2d  # hits %4d\n",
	       ftrg1_.mir_num[i], ftrg1_.nhit_dsp[i]);

      	fprintf (fp, "it_pld_start %4d  end %4d",ftrg1_.t_pld_start[i], ftrg1_.t_pld_end[i]);
        fprintf (fp, " V_pattern %4.4hX  H_pattern %4.4hX",ftrg1_.row_pattern[i], ftrg1_.col_pattern[i]);
        fprintf (fp, " trig_code %4.4hX\n",ftrg1_.trig_code[i]); 
        if (ftrg1_.num_discard == 1)
          fprintf (fp, " intersite trigger nsec %d\n",ftrg1_.discard_clkcnt[0]);
        
        fprintf (fp, "Write pointers (Slaves 0-15):\n");
        for (j=0;j<16;j++) {
          fprintf (fp," %4.4hX",ftrg1_.wp[i][j]);
        }
        fprintf (fp,"\n");
      }
    }
  else if (*long_output==1)
  {
    for (i = 0; i < ftrg1_.num_mir; i++)
      {
	fprintf (fp, "mirror %2d  # hits %4d\n",
	       ftrg1_.mir_num[i], ftrg1_.nhit_dsp[i]);

      	fprintf (fp, "it_pld_start %4d  end %4d",ftrg1_.t_pld_start[i], ftrg1_.t_pld_end[i]);
        fprintf (fp, " V_pattern %4.4hX  H_pattern %4.4hX",ftrg1_.row_pattern[i], ftrg1_.col_pattern[i]);
        fprintf (fp, " trig_code %4.4hX\n",ftrg1_.trig_code[i]); 
        if (ftrg1_.num_discard == 1)
          fprintf (fp, " intersite trigger nsec %d\n",ftrg1_.discard_clkcnt[0]);
        /*
        fprintf (fp, " Num. discarded triggers %d\n",ftrg1_.num_discard);
        if (ftrg1_.num_discard > 0) {
          fprintf (fp, " Code  Mirid  Clkcnt  Vpattern  Hpattern  Nbyte\n");
        }
        for (j=0;j<ftrg1_.num_discard;j++) {
          fprintf (fp, " %4.4hX  %6d %8d   %4.4hX      %4.4hX     %d\n",
                   ftrg1_.discard_code[j],ftrg1_.discard_mirid[j],ftrg1_.discard_clkcnt[j],
                   ftrg1_.discard_vpattern[j],ftrg1_.discard_hpattern[j],ftrg1_.discard_nbyte[j]);
        }
        */
	
        fprintf (fp, "\nWrite pointers (Slaves 0-15):\n");
        for (j=0;j<16;j++) {
          fprintf (fp," %4.4hX",ftrg1_.wp[i][j]);
        }
        fprintf (fp,"\n \n");
	for (j = 0; j < ftrg1_.nhit_dsp[i]; j++)
	  {
	    fprintf (fp, "hit# %3d  chan# %3d  it0 %4d  nt %3d  itav %4d  stddev %3d  cts %4d\n", j+1, ftrg1_.ichan_dsp[i][j], ftrg1_.it0_dsp[i][j], ftrg1_.nt_dsp[i][j] , ftrg1_.tav_dsp[i][j], ftrg1_.sigt_dsp[i][j], ftrg1_.nadc_dsp[i][j]);
	  }
      }      
  }
  return 0;
} 
