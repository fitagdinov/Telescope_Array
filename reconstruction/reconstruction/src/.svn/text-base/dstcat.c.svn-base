/*
 * dstcat.c 
 *
 * Concatenates several DST files into one output DST file
 *
 * $Source: /hires_soft/uvm2k/pro/dstcat.c,v $
 * $Log: dstcat.c,v $
 * Last modified: DI 20171206
 *
 * Revision 1.2  1996/05/30 18:00:26  mjk
 * Fixed bug - had a ; instead of a , in a variable list
 *
 * Revision 1.1  1996/05/30  00:11:35  mjk
 * Initial revision
 *
 *
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "univ_dst.h"
#include "event.h"
#include "filestack.h"

/* Prototypes */
integer4 copy_file(integer1 *);

/* Command line option flags */
integer4 flag_o=0;
integer1 *outfile;

integer4 dst_unit_in = 1, dst_unit_out = 2;
integer4 banklist_want, banklist_got;

int main(int argc, char **argv) {

  integer4 rcode, arg;
  integer4 dst_mode = MODE_WRITE_DST;
  integer1 *filename;
  
  if (argc == 1 ) {
    fprintf(stderr,"\n  Usage: %s [-o output] dst_file [dst_file...]\n\n", 
	    argv[0]);
    fprintf(stderr,"     -o : output dst filename\n");
    fprintf(stderr,"\n  Concatenates several DST files into one output\n\n");
    fputs("\nCurrently recognized banks:",stderr);
    dscBankList((rcode=newBankList(nBanksTotal()),eventAllBanks(rcode),rcode),stderr);
    fprintf(stderr,"\n\n");
    exit(1);
  }

  /* Otherwise, scan the arguments first */
  
  for (arg = 1; arg < argc; ++arg) {
    if (argv[arg][0] != '-') 
      pushFile( argv[arg] );

    else {
      switch (argv[arg][1]) {
      case 'o': 
	flag_o = 1; arg++; outfile = argv[arg]; break;

      default: 
	fprintf(stderr,"Warning: unknown option: %s\n",argv[arg]); 
	break;
      }
    }
  }

  if ( !flag_o ) {
    fprintf(stderr, "\n  Error: No output file given\n\n");
    exit(1);
  }

  if ( (rcode = dst_open_unit_(&dst_unit_out, outfile, &dst_mode)) ) { 
    fprintf(stderr,"\n  Unable to open/write file: %s\n\n", outfile);
    exit(1);
  }

  banklist_want = newBankList(nBanksTotal());
  banklist_got  = newBankList(nBanksTotal());
  eventAllBanks(banklist_want);

  /* Now process input file(s) */
  
  while ( (filename = pullFile()) )
    copy_file( filename );
  

  /* close DST unit */
  dst_close_unit_(&dst_unit_out);
  return SUCCESS;
}


integer4 copy_file(integer1 *dst_filename) {

  integer4 rcode, ssf, dst_mode = MODE_READ_DST;
  
  if ( (rcode = dst_open_unit_(&dst_unit_in, dst_filename, &dst_mode)) ) {
    fprintf(stderr,"\n  Warning: Unable to open/read file: %s\n", 
	    dst_filename);
    return(-1);
  }

  else {
    printf("  Concatenating: %s\n", dst_filename);

    
    for (;;) {
      rcode = eventRead(dst_unit_in, banklist_want, banklist_got, &ssf);
      if ( rcode < 0 ) break ;
      if ( eventWrite( dst_unit_out, banklist_got, ssf) < 0) {
	fprintf(stderr, "  Failed to write event\n");
	dst_close_unit_(&dst_unit_in);
	return -2;
      }
    }
      
    dst_close_unit_(&dst_unit_in);
    if ( rcode != END_OF_FILE ) {
      fprintf(stderr,"  Error reading file\n");
      return -3;
    }

    return rcode;
  }
} 

