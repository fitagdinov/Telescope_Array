/*
 * hspec_dst.c 
 *
 * $Source: /hires_soft/cvsroot/bank/hspec_dst.c,v $
 * $Log: hspec_dst.c,v $
 * Revision 1.1  1999/05/25 15:45:01  ben
 * Initial revision
 *
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "hspec_dst.h"  

hspec_dst_common hspec_;  /* allocate memory to hspec_common */

static integer4 hspec_blen = 0; 
static integer4 hspec_maxlen = sizeof(integer4) * 2 + sizeof(hspec_dst_common);
static integer1 *hspec_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hspec_bank_buffer_ (integer4* hspec_bank_buffer_size)
{
  (*hspec_bank_buffer_size) = hspec_blen;
  return hspec_bank;
}



static void hspec_bank_init(void)
{
  hspec_bank = (integer1 *)calloc(hspec_maxlen, sizeof(integer1));
  if (hspec_bank==NULL)
    {
      fprintf(stderr, 
	      "hspec_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 hspec_common_to_bank_(void)
{	
  static integer4 id = HSPEC_BANKID, ver = HSPEC_BANKVERSION;
  integer4 rcode, nobj;

  if (hspec_bank == NULL) hspec_bank_init();

  /* Initialize hspec_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &hspec_blen, &hspec_maxlen, hspec_bank)))
    return rcode;
  
  if ((rcode = dst_packr8_( &hspec_.jdate, (nobj=1, &nobj), hspec_bank,
                            &hspec_blen, &hspec_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &hspec_.year, (nobj=1, &nobj), hspec_bank,
                            &hspec_blen, &hspec_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &hspec_.month, (nobj=1, &nobj), hspec_bank,
                            &hspec_blen, &hspec_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &hspec_.day, (nobj=1, &nobj), hspec_bank,
                            &hspec_blen, &hspec_maxlen))) return rcode;

  if ((rcode = dst_packi1_( &hspec_.creation_flag, (nobj=1, &nobj), hspec_bank, 
			    &hspec_blen, &hspec_maxlen))) return rcode;
  
  if ((rcode = dst_packi1_( &hspec_.spectrum_flag, (nobj=1, &nobj), hspec_bank, 
			    &hspec_blen, &hspec_maxlen))) return rcode;
  
  if ((rcode = dst_packr8_( hspec_.spectrum, (nobj=HSPEC_BINS, &nobj), 
			    hspec_bank, &hspec_blen, &hspec_maxlen))) 
    return rcode;
  
  if ((rcode = dst_packi1_( (integer1 *)hspec_.description, (nobj=HSPEC_DL*HSPEC_DC, &nobj),
			    hspec_bank, &hspec_blen, &hspec_maxlen))) 
    return rcode;

  return SUCCESS;
}


integer4 hspec_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &hspec_blen, hspec_bank );
}

integer4 hspec_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = hspec_common_to_bank_()))
    {
      fprintf (stderr,"hspec_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = hspec_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"hspec_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 hspec_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  
  hspec_blen = 2 * sizeof(integer4); /* skip id and version  */

  if ((rcode = dst_unpackr8_( &hspec_.jdate, (nobj=1, &nobj), bank,
			      &hspec_blen, &hspec_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &hspec_.year, (nobj=1, &nobj), bank,
			      &hspec_blen, &hspec_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &hspec_.month, (nobj=1, &nobj), bank,
			      &hspec_blen, &hspec_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &hspec_.day, (nobj=1, &nobj), bank,
			      &hspec_blen, &hspec_maxlen))) return rcode;
  
  if ((rcode = dst_unpacki1_( &hspec_.creation_flag, (nobj=1, &nobj), 
			      bank, &hspec_blen, &hspec_maxlen))) 
    return rcode;
  
  if ((rcode = dst_unpacki1_( &hspec_.spectrum_flag, (nobj=1, &nobj), 
			      bank, &hspec_blen, &hspec_maxlen))) 
    return rcode;
  
  if ((rcode = dst_unpackr8_( hspec_.spectrum, (nobj=HSPEC_BINS, &nobj), bank,
			      &hspec_blen, &hspec_maxlen))) return rcode;
  
  if ((rcode = dst_unpacki1_( (integer1 *)hspec_.description, 
			      (nobj=HSPEC_DL*HSPEC_DC, &nobj), 
			      bank, &hspec_blen, &hspec_maxlen))) 
    return rcode;
  
  return SUCCESS;
}

integer4 hspec_common_to_dump_(integer4 *long_output)
{
  return hspec_common_to_dumpf_(stdout, long_output);
}

integer4 hspec_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  int i;

  fprintf(fp,"\nHSPEC bank. \n\n");

  for (i = 0; i < HSPEC_DL; i++) {
    if (hspec_.description[i][0] != '\0')
      fprintf(fp, "%s\n", hspec_.description[i]);
  }

  fprintf(fp, "\n");
  fprintf(fp, "Calendar date of bank creation: %2d/%2d/%4d\n", hspec_.month,
	  hspec_.day, hspec_.year);
  fprintf(fp, "Julian date of bank creation: %lf\n", hspec_.jdate);
  
  switch (hspec_.creation_flag) {
  case HSPEC_CREATION_ARB:
    fprintf(fp, "HSPEC bank is of ARBITRARY creation.\n");
    break;
  case HSPEC_CREATION_ANALYSIS:
    fprintf(fp, "HSPEC bank is created from the ANALYSIS of data.\n");
    break;
  case HSPEC_CREATION_MEASURE:
    fprintf(fp, "HSPEC bank is created from lab/field MEASUREMENTS.\n");
    break;
  case HSPEC_CREATION_APPROX:
    fprintf(fp, "HSPEC bank is of APPROXIMATE/THEORETICAL creation.\n");
    break;
  default:
    fprintf(fp, "HSPEC bank is of UNKNOWN creation.\n");
  }

  switch (hspec_.spectrum_flag) {
  case HSPEC_SPECTYPE_ARB:
    fprintf(fp, "HSPEC spectrum is of ARBITRARY type.\n");
    break;
  case HSPEC_SPECTYPE_RXF:
    fprintf(fp, "HSPEC spectrum is of ROVING FLASHER type.\n");
    break;
  case HSPEC_SPECTYPE_YAG:
    fprintf(fp, "HSPEC spectrum is of YAG laser type.\n");
    break;
  case HSPEC_SPECTYPE_RXFNARROW:
    fprintf(fp, "HSPEC spectrum is of RXF/NARROW BAND type.\n");
    break;
  case HSPEC_SPECTYPE_RXFNARROWAPPROX:
    fprintf(fp, "HSPEC spectrum is of RXF/APPROXIMATE NARROW BAND type.\n");
    break;
  case HSPEC_SPECTYPE_LED:
    fprintf(fp, "HSPEC spectrum is of LED type.\n");
    break;
  case HSPEC_SPECTYPE_FILTER:
    fprintf(fp, "HSPEC spectrum is of FILTER type.\n");
    break;
  case HSPEC_SPECTYPE_QE:
    fprintf(fp, "HSPEC spectrum is of QUANTUM EFFICIENCY type.\n");
    break;
  default:
    fprintf(fp, "HSPEC spectrum is of UNKNOWN type.\n");
  }

  if ( *long_output == 1 ) {

    fprintf(fp, "  LAMBDA  EMISSION/TRANSMISSION\n");
    fprintf(fp, "  ------  ---------------------\n");
    for (i = 0; i < HSPEC_BINS; i++)
      fprintf(fp, "  %6.2lf          %6.4lf\n", 
	      (float)(i+200), hspec_.spectrum[i]);
  
  }

  fprintf(fp, "\n");

  return SUCCESS;
}
