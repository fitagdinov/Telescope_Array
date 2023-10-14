/*
 */
#include <stdio.h>
#include <stdlib.h>
#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"
#include "fdcalib_util.h"
#include "clfvaod_dst.h"

clfvaod_dst_common clfvaod_;

static integer4 clfvaod_blen = 0;
static integer4 clfvaod_maxlen = sizeof(integer4)*2 + sizeof(clfvaod_dst_common);

// Data block for read/write 
static integer1 *clfvaod_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* clfvaod_bank_buffer_ (integer4* clfvaod_bank_buffer_size)
{
  (*clfvaod_bank_buffer_size) = clfvaod_blen;
  return clfvaod_bank;
}



static void clfvaod_init()
{
  clfvaod_bank = (integer1 *) calloc(clfvaod_maxlen, sizeof(integer1));
  if (clfvaod_bank == NULL) {
    fprintf(stderr, "clfvaod_init: fail to assign memory to bank. Abort.\n");
    exit(1);
  }
  //  fprintf(stderr, "ALLOCATED the memory\n");
}

// Pack the data from struct to the data block
integer4 clfvaod_common_to_bank_()
{
  static integer4 id = CLFVAOD_BANKID, ver = CLFVAOD_BANKVERSION;
  integer4 rcode, nobj;


  if (clfvaod_bank == NULL) clfvaod_init();
  /* Initialize test_blen, and pack the id and version to bank */
  rcode = dst_initbank_(&id, &ver, &clfvaod_blen, &clfvaod_maxlen,
			clfvaod_bank);

  nobj = 1;

  /* Change here for each bank structure */
  rcode += dst_packi4_(&clfvaod_.uniqID, &nobj, clfvaod_bank, &clfvaod_blen,
		       &clfvaod_maxlen);
  rcode += dst_packi4_(&clfvaod_.dateFrom, &nobj, clfvaod_bank, &clfvaod_blen,
		       &clfvaod_maxlen);
  rcode += dst_packi4_(&clfvaod_.dateTo, &nobj, clfvaod_bank, &clfvaod_blen,
		       &clfvaod_maxlen);
  rcode += dst_packr4_(&clfvaod_.laserEnergy, &nobj, clfvaod_bank, &clfvaod_blen,
		       &clfvaod_maxlen);
  rcode += dst_packr4_(&clfvaod_.vaod, &nobj, clfvaod_bank, &clfvaod_blen,
		       &clfvaod_maxlen);
  rcode += dst_packr4_(&clfvaod_.vaodMin, &nobj, clfvaod_bank, &clfvaod_blen,
		       &clfvaod_maxlen);
  rcode += dst_packr4_(&clfvaod_.vaodMax, &nobj, clfvaod_bank, &clfvaod_blen,
			 &clfvaod_maxlen);
  return rcode;
}

// Unpack the data from the data block to struct
integer4 clfvaod_bank_to_common_(integer1 *block)
{
  integer4 rcode = 0;
  integer4 nobj;

  clfvaod_blen = 2*sizeof(integer4); /* skip id and version */
  nobj = 1;

  /* Change here for each bank structure */
  rcode += dst_unpacki4_(&clfvaod_.uniqID, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);
  rcode += dst_unpacki4_(&clfvaod_.dateFrom, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);
  rcode += dst_unpacki4_(&clfvaod_.dateTo, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);
  rcode += dst_unpackr4_(&clfvaod_.laserEnergy, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);
  rcode += dst_unpackr4_(&clfvaod_.vaod, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);
  rcode += dst_unpackr4_(&clfvaod_.vaodMin, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);
  rcode += dst_unpackr4_(&clfvaod_.vaodMax, &nobj, block, &clfvaod_blen, &clfvaod_maxlen);

  return rcode;
}

integer4 clfvaod_common_to_dumpf_(FILE *fp, integer4 *long_output)
{
  /* Change here for each bank */
  (void)(long_output); /* if long_output flag is not used, void it so 
			  that the compiler doesn't complain about 
			  unused arguments */

   fprintf(fp,"clfvaod_dst\n");
   fprintf(fp,"uniq ID: %d\n",clfvaod_.uniqID);
   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(clfvaod_.dateFrom,dateFromLine);
   convertSec2DateLine(clfvaod_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
  
  fprintf(fp, "Laser Energy: %f\n", clfvaod_.laserEnergy);
  fprintf(fp, "VAOD at 5km: %f\n", clfvaod_.vaod);
  fprintf(fp, " Min: %f, Max: %f\n", clfvaod_.vaodMin, clfvaod_.vaodMax);
  return 0;
}


integer4 clfvaod_bank_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_(NumUnit, &clfvaod_blen, clfvaod_bank);
  free(clfvaod_bank);
  clfvaod_bank = NULL;
  //  fprintf(stderr, "FREED the memory\n");
  return rcode;
}

integer4 clfvaod_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = clfvaod_common_to_bank_())) {
    fprintf(stderr, "clfvaod_common_to_bank_ ERROR: %ld\n", (long) rcode);
    exit(1);
  } 
  if ((rcode = clfvaod_bank_to_dst_(NumUnit))) {
    fprintf(stderr, "clfvaod_bank_to_dst_ ERROR: %ld\n", (long) rcode);
    exit(2);
  }
  return 0;
}
