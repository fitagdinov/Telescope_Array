#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "tale_db_uvled_dst.h"  

tale_db_uvled_dst_common tale_db_uvled_;  /* allocate memory to tale_db_uvled_common */

static integer4 tale_db_uvled_blen = 0; 
static integer4 tale_db_uvled_maxlen = sizeof(integer4) * 2 + sizeof(tale_db_uvled_dst_common);
static integer1 *tale_db_uvled_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tale_db_uvled_bank_buffer_ (integer4* tale_db_uvled_bank_buffer_size)
{
  (*tale_db_uvled_bank_buffer_size) = tale_db_uvled_blen;
  return tale_db_uvled_bank;
}



static void tale_db_uvled_bank_init()
{
  tale_db_uvled_bank = (integer1 *)calloc(tale_db_uvled_maxlen, sizeof(integer1));
  if (tale_db_uvled_bank==NULL) {
    fprintf(stderr, "tale_db_uvled_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

integer4 tale_db_uvled_common_to_bank_()
{	
  static integer4 id = TALE_DB_UVLED_BANKID, ver = TALE_DB_UVLED_BANKVERSION;
  integer4 rcode = 0;
  integer4 nobj;

  if (tale_db_uvled_bank == NULL) tale_db_uvled_bank_init();

  /* Initialize tale_db_uvled_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &tale_db_uvled_blen, &tale_db_uvled_maxlen, tale_db_uvled_bank))) return rcode;

  nobj = 4; // jday, idate, ipart, mirid
  if ((rcode = dst_packi4_(&tale_db_uvled_.jday, &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;

  nobj = TALE_DB_UVLED_NCHN;
  if ((rcode = dst_packi4asi2_(tale_db_uvled_.nshots,     &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tale_db_uvled_.nsaturated, &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tale_db_uvled_.nvalid,     &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packr4_    (tale_db_uvled_.qdc_mean,   &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packr4_    (tale_db_uvled_.qdc_sdev,   &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packr4_    (tale_db_uvled_.npe_mean,   &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packr4_    (tale_db_uvled_.npe_sdev,   &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packr4_    (tale_db_uvled_.tube_gain,  &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_packr4_    (tale_db_uvled_.tube_cfqe,  &nobj, tale_db_uvled_bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;

  return SUCCESS;
}

integer4 tale_db_uvled_bank_to_dst_(integer4 *unit)
{	
  return dst_write_bank_(unit, &tale_db_uvled_blen, tale_db_uvled_bank );
}

integer4 tale_db_uvled_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
  if ((rcode = tale_db_uvled_common_to_bank_())) {
    fprintf (stderr,"tale_db_uvled_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit(0);			 	
  }
  if ((rcode = tale_db_uvled_bank_to_dst_(unit))) {
    fprintf (stderr,"tale_db_uvled_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit(0);			 	
  }
  return SUCCESS;
}

integer4 tale_db_uvled_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  tale_db_uvled_blen = 2 * sizeof(integer4); /* skip id and version  */

  nobj = 4; // jday, idate, ipart, mirid
  if ((rcode = dst_unpacki4_(&(tale_db_uvled_.jday), &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;

  nobj = TALE_DB_UVLED_NCHN;
  if ((rcode = dst_unpacki2asi4_(tale_db_uvled_.nshots,     &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tale_db_uvled_.nsaturated, &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tale_db_uvled_.nvalid,     &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_    (tale_db_uvled_.qdc_mean,   &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_    (tale_db_uvled_.qdc_sdev,   &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_    (tale_db_uvled_.npe_mean,   &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_    (tale_db_uvled_.npe_sdev,   &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_    (tale_db_uvled_.tube_gain,  &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_    (tale_db_uvled_.tube_cfqe,  &nobj, bank, &tale_db_uvled_blen, &tale_db_uvled_maxlen))) return rcode;

  return SUCCESS;
}

integer4 tale_db_uvled_common_to_dump_(integer4 *long_output)
{
  return tale_db_uvled_common_to_dumpf_(stdout, long_output);
}

integer4 tale_db_uvled_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 ich;
  (void)(long_output);
  fprintf(fp, "\nTALE_DB_UVLED mirid %2d jday %10d idate %10d ipart %2d  nshots %d\n",
	  tale_db_uvled_.mirid, tale_db_uvled_.jday, tale_db_uvled_.idate, tale_db_uvled_.ipart, tale_db_uvled_.nshots[0]);

  for (ich=0; ich<TALE_DB_UVLED_NCHN; ++ich) {
    /* fprintf(fp, " mir %2d tube %3d  nshots %4d  nsat. %4d  nvalid %4d  qdc_mean %9.2f qdc_sdev %9.3f  tube_gain %9.5f tube_cfqe %9.5f\n", */
    /* 	    tale_db_uvled_.mirid, ich + 1, */
    /* 	    tale_db_uvled_.nshots[ich], tale_db_uvled_.nsaturated[ich], tale_db_uvled_.nvalid[ich], */
    /* 	    tale_db_uvled_.qdc_mean[ich], tale_db_uvled_.qdc_sdev[ich], */
    /* 	    tale_db_uvled_.tube_gain[ich], tale_db_uvled_.tube_cfqe[ich]); */
    fprintf(fp, " tube %3d  qdc_mean %8.2f qdc_sdev %6.3f  gain %8.5f cfqe %8.5f\n",
	    ich + 1,
	    tale_db_uvled_.qdc_mean[ich], tale_db_uvled_.qdc_sdev[ich],
	    tale_db_uvled_.tube_gain[ich], tale_db_uvled_.tube_cfqe[ich]);
  }
  
  fprintf(fp,"\n");
  return SUCCESS;
}

