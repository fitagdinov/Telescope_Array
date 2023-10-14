// Created 2009/03/24 LMS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdped_dst.h"
#include "caldat.h"

fdped_dst_common fdped_;

integer4 fdped_blen = 0; /* not static because it needs to be accessed by the c files of the derived banks */
static integer4 fdped_maxlen = sizeof(integer4) * 2 + sizeof(fdped_dst_common);
static integer1 *fdped_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdped_bank_buffer_ (integer4* fdped_bank_buffer_size)
{
  (*fdped_bank_buffer_size) = fdped_blen;
  return fdped_bank;
}



static void fdped_abank_init(integer1* (*pbank) ) {
  *pbank = (integer1 *)calloc(fdped_maxlen, sizeof(integer1));
  if (*pbank==NULL) {
      fprintf (stderr,"fdped_abank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

static void fdped_bank_init() {fdped_abank_init(&fdped_bank);}

integer4 fdped_common_to_bank_() {
  if (fdped_bank == NULL) fdped_bank_init();
  return fdped_struct_to_abank_(&fdped_, &fdped_bank, FDPED_BANKID, FDPED_BANKVERSION);
}
integer4 fdped_bank_to_dst_ (integer4 *unit) {return fdped_abank_to_dst_(fdped_bank, unit);}
integer4 fdped_common_to_dst_(integer4 *unit) {
  if (fdped_bank == NULL) fdped_bank_init();
  return fdped_struct_to_dst_(&fdped_, fdped_bank, unit, FDPED_BANKID, FDPED_BANKVERSION);
}
integer4 fdped_bank_to_common_(integer1 *bank) {return fdped_abank_to_struct_(bank, &fdped_);}
integer4 fdped_common_to_dump_(integer4 *opt) {return fdped_struct_to_dumpf_(&fdped_, stdout, opt);}
integer4 fdped_common_to_dumpf_(FILE* fp, integer4 *opt) {return fdped_struct_to_dumpf_(&fdped_, fp, opt);}

integer4 fdped_struct_to_abank_(fdped_dst_common *fdped, integer1 *(*pbank), integer4 id, integer4 ver) {
  integer4 rcode, nobj, i, j;
  integer1 *bank;

  if (*pbank == NULL) fdped_abank_init(pbank);

  bank = *pbank;
  rcode = dst_initbank_(&id, &ver, &fdped_blen, &fdped_maxlen, bank);

// Initialize fdped_blen and pack the id and version to bank

  nobj = 1;
  rcode += dst_packi4_(&fdped->julian_start,   &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_packi4_(&fdped->jsecond_start,  &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_packi4_(&fdped->jsecfrac_start, &nobj, bank, &fdped_blen, &fdped_maxlen);

  rcode += dst_packi4_(&fdped->julian_end,     &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_packi4_(&fdped->jsecond_end,    &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_packi4_(&fdped->jsecfrac_end,   &nobj, bank, &fdped_blen, &fdped_maxlen);

  rcode += dst_packi4_(&fdped->siteid,         &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_packi4_(&fdped->part,           &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_packi4_(&fdped->num_minutes,    &nobj, bank, &fdped_blen, &fdped_maxlen);

  for ( i=0; i<FDPED_NCAM; i++ ) {
    for ( j=0; j<FDPED_NPMT; j++ ) {
      nobj = fdped->num_minutes;
      rcode += dst_packi4_(&fdped->pedestal[i][j][0], &nobj, bank, &fdped_blen, &fdped_maxlen);
    }
  }

  for ( i=0; i<FDPED_NCAM; i++ ) {
    for ( j=0; j<FDPED_NPMT; j++ ) {
      nobj = fdped->num_minutes;
      rcode += dst_packi4_(&fdped->pedrms[i][j][0],   &nobj, bank, &fdped_blen, &fdped_maxlen);
    }
  }

  for ( i=0; i<FDPED_NCAM; i++ ) {
    for ( j=0; j<FDPED_NPMT; j++ ) {
      nobj = fdped->num_minutes;
      rcode += dst_packi4_(&fdped->liveflag[i][j][0], &nobj, bank, &fdped_blen, &fdped_maxlen);
    }
  }

  return rcode;
}

integer4 fdped_abank_to_dst_(integer1 *bank, integer4 *unit) {
  return dst_write_bank_(unit, &fdped_blen, bank);
}

integer4 fdped_struct_to_dst_(fdped_dst_common *fdped, integer1 *bank, integer4 *unit, integer4 id, integer4 ver) {
  integer4 rcode;
  if ( (rcode = fdped_struct_to_abank_(fdped, &bank, id, ver)) ) {
      fprintf(stderr, "fdped_struct_to_abank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  if ( (rcode = fdped_abank_to_dst_(bank, unit)) ) {
      fprintf(stderr, "fdped_abank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  return 0;
}

integer4 fdped_abank_to_struct_(integer1 *bank, fdped_dst_common *fdped) {
  integer4 rcode = 0 ;
  integer4 nobj, i, j;
  fdped_blen = 2 * sizeof(integer4);   /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_(&fdped->julian_start,   &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_unpacki4_(&fdped->jsecond_start,  &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_unpacki4_(&fdped->jsecfrac_start, &nobj, bank, &fdped_blen, &fdped_maxlen);

  rcode += dst_unpacki4_(&fdped->julian_end,     &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_unpacki4_(&fdped->jsecond_end,    &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_unpacki4_(&fdped->jsecfrac_end,   &nobj, bank, &fdped_blen, &fdped_maxlen);

  rcode += dst_unpacki4_(&fdped->siteid,         &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_unpacki4_(&fdped->part,           &nobj, bank, &fdped_blen, &fdped_maxlen);
  rcode += dst_unpacki4_(&fdped->num_minutes,    &nobj, bank, &fdped_blen, &fdped_maxlen);

  for ( i=0; i<FDPED_NCAM; i++ ) {
    for ( j=0; j<FDPED_NPMT; j++ ) {
      nobj = fdped->num_minutes;
      rcode += dst_unpacki4_(&fdped->pedestal[i][j][0], &nobj, bank, &fdped_blen, &fdped_maxlen);
    }
  }

  for ( i=0; i<FDPED_NCAM; i++ ) {
    for ( j=0; j<FDPED_NPMT; j++ ) {
      nobj = fdped->num_minutes;
      rcode += dst_unpacki4_(&fdped->pedrms[i][j][0],   &nobj, bank, &fdped_blen, &fdped_maxlen);
    }
  }

  for ( i=0; i<FDPED_NCAM; i++ ) {
    for ( j=0; j<FDPED_NPMT; j++ ) {
      nobj = fdped->num_minutes;
      rcode += dst_unpacki4_(&fdped->liveflag[i][j][0], &nobj, bank, &fdped_blen, &fdped_maxlen);
    }
  }

  return rcode;
}

integer4 fdped_struct_to_dump_(fdped_dst_common *fdped, integer4 *long_output) {
  return fdped_struct_to_dumpf_(fdped, stdout, long_output);
}

integer4 fdped_struct_to_dumpf_(fdped_dst_common *fdped, FILE* fp, integer4 *long_output) {
  int i, j, k;
  integer4 yr0=0, mo0=0, day0=0;
  integer4 yr1=0, mo1=0, day1=0;
  integer4 hr0, min0, sec0, nano0;
  integer4 hr1, min1, sec1, nano1;
  double realmin;

  hr0 = fdped->jsecond_start / 3600 + 12;
  hr1 = fdped->jsecond_end   / 3600 + 12;

  if (hr0 >= 24) {
    caldat((double)fdped->julian_start+1., &mo0, &day0, &yr0);
    hr0 -= 24;
  }
  else
    caldat((double)fdped->julian_start, &mo0, &day0, &yr0);

  if (hr1 >= 24) {
    caldat((double)fdped->julian_end+1., &mo1, &day1, &yr1);
    hr1 -= 24;
  }
  else
    caldat((double)fdped->julian_end, &mo1, &day1, &yr1);

  min0 = ( fdped->jsecond_start / 60 ) % 60;
  sec0 = fdped->jsecond_start % 60;
  nano0 = fdped->jsecfrac_start;

  min1 = ( fdped->jsecond_end / 60 ) % 60;
  sec1 = fdped->jsecond_end % 60;
  nano1 = fdped->jsecfrac_end;

  realmin = (((double)fdped->julian_end +
              (double)fdped->jsecond_end / 86400.0 +
	      (double)fdped->jsecfrac_end / 8.64e13
             ) -
             ((double)fdped->julian_start +	
       	      (double)fdped->jsecond_start / 86400.0 +
              (double)fdped->jsecfrac_start / 8.64e13
             )
            ) * 1440.0;

  if (fdped->siteid == BR)
    fprintf (fp, "\n\nBRPED bank (TA pedestal information for Black Rock FD)\n");
  else if (fdped->siteid == LR)
    fprintf (fp, "\n\nLRPED bank (TA pedestal information for Long Ridge FD)\n");
  fprintf (fp, "%4d/%02d/%02d %02d:%02d:%02d.%09d to ", 
    yr0, mo0, day0, hr0, min0, sec0, nano0);
  fprintf (fp, "%4d/%02d/%02d %02d:%02d:%02d.%09d : ",
    yr1, mo1, day1, hr1, min1, sec1, nano1);
  if (fdped->num_minutes == 1)
    fprintf (fp, "Part %2d [ %3d minute  %6.2f ]\n\n", fdped->part, fdped->num_minutes,
             realmin);
  else
    fprintf (fp, "Part %2d [ %3d minutes %6.2f ]\n\n", fdped->part, fdped->num_minutes,
             realmin); 

// Tube info
  if ( (*long_output) == 1) {
    for (i=0; i<fdped->num_minutes; i++) {
      fprintf(fp, "minute %3d\n", i);

      for (j=0; j<FDPED_NCAM; j++)
        for (k=0; k<FDPED_NPMT; k++)
          fprintf(fp, "  p %2d min %3d | m %02d t %03d : %6d +/- %6d | %d\n", 
            fdped->part, i, j, k,
            fdped->pedestal[j][k][i], fdped->pedrms[j][k][i], fdped->liveflag[j][k][i]);

    }
  }
  else
    fprintf (fp, "Tube information not displayed in short output\n");

  fprintf (fp, "\n\n");

  return 0;
}
