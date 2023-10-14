#include <stdio.h>
#include "event.h"  /* includes all bank structures */

/* if you would like to tailor your include declarations, you can also use:

#include "univ_dst.h"
#include "dst_std_types.h"

#include "yourbank1_dst.h"
#include "yourbank2_dst.h"
#include "yourbankN_dst.h"

*/

/*
 *  Compile Command:
 *
 *  gcc -o simplesample simplesample.c -Iinc -lm -lc -lz -lbz2 -Llib -ldst2k
 *
 *
 *  This simple example counts the number of events in a DST file.  Each call 
 *    to 'eventRead()' sets the fields in the bank structures (declared in 
 *    inc/"bankname"_dst.h) included in the input file.
 *
 *  Sean Stratton -- 8.25.08
 */
int main(int argc, char *argv[]) {
  int n;  /* a counter */

  /* the following are variables typically needed for DST file I/O */
  int rc, inunit=1, inmode=MODE_READ_DST;
  int wantbanks, havebanks, size=100, event;
  char *dstpath;

  if ( argc != 2 ) {
    fprintf(stderr, "usage: %s dstpath\n", argv[0]);
    return 1;
  }

  dstpath = argv[1];

  /* I'm actually not sure exactly what these functions do, but they are 
   *   essential for 'eventRead()' to function. */
  wantbanks = newBankList(size);
  havebanks = newBankList(size);

  /* open the dst file.  If dstOpenUnit() returns non-zero code, then an error 
   *   occurred, you can see the names of the codes in inc/dst_err_codes.h. */
  rc = dstOpenUnit(inunit, dstpath, inmode);
  if ( rc != 0 ) {
    fprintf(stderr, "unable to open %s for reading.\n", dstpath);
    return rc;
  }

  n = 0;

  /* loop through events in input file.  eventRead() returns '-1' when end of 
   *   dst file has been reached. */
  while ( eventRead(inunit, wantbanks, havebanks, &event) >= 0 ) {
    n++;
  }

  /* close the dst file */
  dstCloseUnit(inunit);

  printf("file '%s' contains %d events\n", dstpath, n);

  return 0;
}
