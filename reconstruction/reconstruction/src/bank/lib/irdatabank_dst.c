/*!
 * Bank for IR data
 * Y. Tsunesada 2008/Dec/17
 */
#include "irdatabank_dst.h"

irdatabank_dst_common irdatabank_;

static integer4 irdatabank_blen = 0;
static integer4 irdatabank_maxlen = sizeof(integer4)*2 + sizeof(irdatabank_dst_common);

// Data block for read/write 
static integer1 *irdatabank_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* irdatabank_bank_buffer_ (integer4* irdatabank_bank_buffer_size)
{
  (*irdatabank_bank_buffer_size) = irdatabank_blen;
  return irdatabank_bank;
}



static void irdatabank_init()
{
  irdatabank_bank = (integer1 *) calloc(irdatabank_maxlen, sizeof(integer1));
  if (irdatabank_bank == NULL) {
    fprintf(stderr, "irdatabank_init: fail to assign memory to bank. Abort.\n");
    exit(1);
  }
  //  fprintf(stderr, "ALLOCATED the memory\n");
}

// Pack the data from struct to the data block
integer4 irdatabank_common_to_bank_()
{
  static integer4 id = IRDATABANK_BANKID, ver = IRDATABANK_BANKVERSION;
  integer4 rcode, nobj;
  //  int i;
  //  integer4 slen = sizeof(irdatabank_.name);

  if (irdatabank_bank == NULL) irdatabank_init();
  /* Initialize irdata_blen, and pack the id and version to bank */
  rcode = dst_initbank_(&id, &ver, &irdatabank_blen, &irdatabank_maxlen,
			irdatabank_bank);


  nobj = 1;

  /* Change here for each bank structure */
  /*  rcode += dst_packi1_(irdatabank_.name, &slen, 
		       irdatabank_bank, &irdatabank_blen,
		       &irdatabank_maxlen);*/
  rcode += dst_packi2_(&irdatabank_.iSite, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi4_(&irdatabank_.dateFrom, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi4_(&irdatabank_.dateTo, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);

  rcode += dst_packi2_(&irdatabank_.iy, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.im, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.id, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.iH, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.iM, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.iS, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);

  nobj = 14;

  rcode += dst_packi2_(&irdatabank_.status[0], &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.DA[0], &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  /*  rcode += dst_packr4_(&irdatabank_.TA, &nobj, irdatabank_bank, 
      &irdatabank_blen, &irdatabank_maxlen);*/
  /*  for (i = 0; i < 5; i++) {
    rcode += dst_packi2_(&irdatabank_.D50[i], &nobj, irdatabank_bank, 
			 &irdatabank_blen, &irdatabank_maxlen);
			 }*/
  nobj = 70;
  rcode += dst_packi2_(&irdatabank_.D50[0][0], &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  /*  rcode += dst_packi2_(&irdatabank_.idir, &nobj, irdatabank_bank, 
	&irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_packi2_(&irdatabank_.icloud, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
    */
  rcode += dst_packi2_(&irdatabank_.score[0][0], &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  nobj = 1;
  rcode += dst_packi2_(&irdatabank_.totalscore, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);

  /* Added 2010/Feb/08 */
  nobj = 70;
  rcode += dst_packr4_(&irdatabank_.prob[0][0], &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  nobj = 1;
  rcode += dst_packr4_(&irdatabank_.totalprob, &nobj, irdatabank_bank, 
		       &irdatabank_blen, &irdatabank_maxlen);
  return rcode;
}

// Unpack the data from the data block to struct
integer4 irdatabank_bank_to_common_(integer1 *block)
{
  integer4 rcode = 0;
  integer4 nobj;
  //  integer4 slen = sizeof(irdatabank_.name);
  //  int i;
  irdatabank_blen = 2*sizeof(integer4); /* skip id and version */
  nobj = 1;

  /* Change here for each bank structure */
  /*  rcode += dst_unpacki1_(irdatabank_.name, &slen, block, &irdatabank_blen, &irdatabank_maxlen);*/
  rcode += dst_unpacki2_(&irdatabank_.iSite, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki4_(&irdatabank_.dateFrom, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki4_(&irdatabank_.dateTo, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);

  rcode += dst_unpacki2_(&irdatabank_.iy, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.im, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.id, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.iH, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.iM, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.iS, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);

  nobj = 14;
  rcode += dst_unpacki2_(&irdatabank_.status[0], &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.DA[0], &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  //  rcode += dst_unpackr4_(&irdatabank_.TA, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  /*  for (i = 0; i < 5; i++) {
    rcode += dst_unpacki2_(&irdatabank_.D50[i], &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
    }*/
  nobj = 70;
  rcode += dst_unpacki2_(&irdatabank_.D50[0][0], &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  //  rcode += dst_unpacki2_(&irdatabank_.idir, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  //  rcode += dst_unpacki2_(&irdatabank_.icloud, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  rcode += dst_unpacki2_(&irdatabank_.score[0][0], &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  nobj = 1;
  rcode += dst_unpacki2_(&irdatabank_.totalscore, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);

  /* Added 2010/Feb/08 */
  nobj = 70;
  rcode += dst_unpackr4_(&irdatabank_.prob[0][0], &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  nobj = 1;
  rcode += dst_unpackr4_(&irdatabank_.totalprob, &nobj, block, &irdatabank_blen, &irdatabank_maxlen);
  return rcode;
}
#include <time.h>
integer4 irdatabank_common_to_dumpf_(FILE *fp, integer4 *long_output)
{
  time_t t;
  char buf[256];
  (void)(long_output);
  /* Change here for each bank */
  /*  fprintf(fp, "%04d/%02d/%02d %02d:%02d:%02d\n",
	  irdatabank_.iy, irdatabank_.im, irdatabank_.id,
	  irdatabank_.iH, irdatabank_.iM, irdatabank_.iS);*/
  t = (time_t) irdatabank_.dateFrom;
  strftime(buf, sizeof(buf), "%Y/%m/%d %H:%M:%S", gmtime(&t));
  fprintf(fp, "%s\n", buf);
  return 0;
}

integer4 irdatabank_bank_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_(NumUnit, &irdatabank_blen, irdatabank_bank);
  free(irdatabank_bank);
  irdatabank_bank = NULL;
  //  fprintf(stderr, "FREED the memory\n");
  return rcode;
}

integer4 irdatabank_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = irdatabank_common_to_bank_())) {
    fprintf(stderr, "irdatabank_common_to_bank_ ERROR: %ld\n", (long) rcode);
    exit(1);
  } 
  if ((rcode = irdatabank_bank_to_dst_(NumUnit))) {
    fprintf(stderr, "irdatabank_bank_to_dst_ ERROR: %ld\n", (long) rcode);
    exit(2);
  }
  return 0;
}
