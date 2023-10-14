/*
 * mcsdd_dst.c 
 *
 * $Source:$
 * $Log:$
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
#include "mcsdd_dst.h"  

mcsdd_dst_common mcsdd_;  /* allocate memory to mcsdd_common */

static integer4 mcsdd_blen = 0; 
static integer4 mcsdd_maxlen = sizeof(integer4)*2 + sizeof(mcsdd_dst_common);
static integer1 *mcsdd_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* mcsdd_bank_buffer_ (integer4* mcsdd_bank_buffer_size)
{
  (*mcsdd_bank_buffer_size) = mcsdd_blen;
  return mcsdd_bank;
}



static void  mcsdd_bank_init(void)
{
  mcsdd_bank = (integer1 *)calloc(mcsdd_maxlen, sizeof(integer1));
  if (mcsdd_bank==NULL)
    {
      fprintf(stderr, 
	      "mcsdd_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 mcsdd_common_to_bank_(void)
{  
  static integer4 id = MCSDD_BANKID;
  static integer4 ver = MCSDD_BANKVERSION;

  integer4 rcode;
  integer4 nobj;

  if (mcsdd_bank == NULL) mcsdd_bank_init();

  /* Initialize mcsdd_blen, and pack the id and version to bank */

  if (( rcode = dst_initbank_(&id, &ver, &mcsdd_blen, &mcsdd_maxlen, mcsdd_bank))) return rcode;

  // Rcore, have_core
  if ((rcode = dst_packr8_(       mcsdd_.Rcore,     (nobj=3, &nobj), mcsdd_bank, &mcsdd_blen, &mcsdd_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_( &(mcsdd_.have_core),(nobj=1, &nobj), mcsdd_bank, &mcsdd_blen, &mcsdd_maxlen))) return rcode;

  return SUCCESS;
}


integer4 mcsdd_bank_to_dst_(integer4 *NumUnit)
{  
  return dst_write_bank_(NumUnit, &mcsdd_blen, mcsdd_bank );
}

integer4 mcsdd_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if (( rcode = mcsdd_common_to_bank_() ))
    {
      fprintf (stderr,"mcsdd_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);         
    }             
  if ((rcode = mcsdd_bank_to_dst_(NumUnit) ))
    {
      fprintf (stderr,"mcsdd_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);         
    }
  return SUCCESS;
}

integer4 mcsdd_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  mcsdd_blen = 2 * sizeof(integer4); /* skip id and version  */

  if ((rcode = dst_unpackr8_(       mcsdd_.Rcore,     (nobj=3, &nobj), bank, &mcsdd_blen, &mcsdd_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_( &(mcsdd_.have_core), (nobj=1, &nobj), bank, &mcsdd_blen, &mcsdd_maxlen))) return rcode;

  return SUCCESS;
}

integer4 mcsdd_common_to_dump_(integer4 *long_output)
{
  return mcsdd_common_to_dumpf_(stdout, long_output);
}

integer4 mcsdd_common_to_dumpf_(FILE* fp, integer4 *long_output) {
  (void)(long_output);
  fprintf(fp, "\nMCSDD______________________________________\n");
  if ( mcsdd_.have_core == 1 ) {
    fprintf(fp, "core location known\n");
    fprintf(fp, "core vector: %10.2f  %10.2f  %10.2f\n",
	    mcsdd_.Rcore[0], mcsdd_.Rcore[1], mcsdd_.Rcore[2] ); 
  }
  else {
    fprintf(fp, "core location unknown\n");
  }
  return SUCCESS;
}




