/*
  K.Hayashi 2010/Jan/06
 */
#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdctd_clock_dst.h"

fdctd_clock_dst_common fdctd_clock_;

static integer4 fdctd_clock_blen = 0;
static integer4 fdctd_clock_maxlen = sizeof(integer4)*2 + sizeof(fdctd_clock_dst_common);

// Data block for read/write
static integer1 *fdctd_clock_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdctd_clock_bank_buffer_ (integer4* fdctd_clock_bank_buffer_size)
{
  (*fdctd_clock_bank_buffer_size) = fdctd_clock_blen;
  return fdctd_clock_bank;
}



static void fdctd_clock_init()
{
  fdctd_clock_bank = (integer1 *) calloc(fdctd_clock_maxlen, sizeof(integer1));
  if (fdctd_clock_bank == NULL) {
    fprintf(stderr, "fdctd_clock_init: fail to assign memory to bank. Abort.\n");
    exit(1);
  }
  //  fprintf(stderr, "ALLOCATED the memory\n");
}

// Pack the data from struct to the data block
integer4 fdctd_clock_common_to_bank_()
{
  static integer4 id = FDCTD_CLOCK_BANKID, ver = FDCTD_CLOCK_BANKVERSION;
  integer4 rcode, nobj;

  if (fdctd_clock_bank == NULL) fdctd_clock_init();
  /* Initialize test_blen, and pack the id and version to bank */
  rcode = dst_initbank_(&id, &ver, &fdctd_clock_blen, &fdctd_clock_maxlen,
			fdctd_clock_bank);

  nobj = 1;

  /* Change here for each bank structure */
  rcode += dst_packi2_(&fdctd_clock_.stID, &nobj, fdctd_clock_bank,
		       &fdctd_clock_blen, &fdctd_clock_maxlen);
  rcode += dst_packi4_(&fdctd_clock_.unixtime, &nobj, fdctd_clock_bank,
		       &fdctd_clock_blen, &fdctd_clock_maxlen);
  rcode += dst_packr8_(&fdctd_clock_.clock, &nobj, fdctd_clock_bank,
		       &fdctd_clock_blen, &fdctd_clock_maxlen);
  return rcode;
}

// Unpack the data from the data block to struct
integer4 fdctd_clock_bank_to_common_(integer1 *block)
{
  integer4 rcode = 0;
  integer4 nobj;

  fdctd_clock_blen = 2*sizeof(integer4); /* skip id and version */
  nobj = 1;

  /* Change here for each bank structure */
  rcode += dst_unpacki2_(&fdctd_clock_.stID, &nobj, block, &fdctd_clock_blen, &fdctd_clock_maxlen);
  rcode += dst_unpacki4_(&fdctd_clock_.unixtime, &nobj, block, &fdctd_clock_blen, &fdctd_clock_maxlen);
  rcode += dst_unpackr8_(&fdctd_clock_.clock, &nobj, block, &fdctd_clock_blen, &fdctd_clock_maxlen);

  return rcode;
}

integer4 fdctd_clock_common_to_dumpf_(FILE *fp, integer4 *long_output)
{
  /* Change here for each bank */
  (void)(long_output);
  fprintf(fp, "stID: %hd\n", fdctd_clock_.stID);
  fprintf(fp, "unixtime: %d\n", fdctd_clock_.unixtime);
  fprintf(fp, "clock: %d\n", (int)fdctd_clock_.clock);
  return 0;
}


integer4 fdctd_clock_bank_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_(NumUnit, &fdctd_clock_blen, fdctd_clock_bank);
  free(fdctd_clock_bank);
  fdctd_clock_bank = NULL;
  //  fprintf(stderr, "FREED the memory\n");
  return rcode;
}

integer4 fdctd_clock_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = fdctd_clock_common_to_bank_())) {
    fprintf(stderr, "fdctd_clock_common_to_bank_ ERROR: %ld\n", (long) rcode);
    exit(1);
  } 
  if ((rcode = fdctd_clock_bank_to_dst_(NumUnit))) {
    fprintf(stderr, "fdctd_clock_bank_to_dst_ ERROR: %ld\n", (long) rcode);
    exit(2);
  }
  return 0;
}
