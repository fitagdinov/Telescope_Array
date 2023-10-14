/*
 * dstlist lists all DST banks in the given event, with one line per event.
 * start and stop banks are ignored, and banks outside of an event are
 * starred.
 *
 * Created: DRB 20081006
 * Last modified: DI 20171206
 *
 */

#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#include "dst_err_codes.h"

#include "event.h"
#include "filestack.h"

void listBanks(integer4 banks, FILE *fp, int split_string);

int main(int argc, char *argv[]) {

  integer1 *name;
  integer4 type, rc, event;
  integer4 gotBanks,wantBanks,dontBanks;
  integer4 unit = 1, mode = MODE_READ_DST;
  integer4 evtread=0,evtskip=0,evtnum=0;
  integer4 vmode = 0;
  
  /* initialize bank lists */
  rc = nBanksTotal();
  gotBanks = newBankList(rc);
  wantBanks = newBankList(rc);
  dontBanks = newBankList(rc);
  eventAllBanks(wantBanks);
  
  /* print help message if no arguments given */
  if (argc == 1) {
    fprintf(stderr, "\nUsage: %s [Flags] dstfile ...\n\n", argv[0]);
    fputs("Flags:\n", stderr);
    fputs("  +NNN:  Skip the first NNN events\n", stderr);
    fputs("  -NNN:  List NNN events (after skipping)\n", stderr);
    fputs("  -name: Do not show the named bank\n", stderr);
    fputs("  -v: (verbose) Inform about reading a new dst file, etc\n",stderr);
    fputs("\nCurrently recognized banks:", stderr);
    dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
    fprintf(stderr,"\n\n");
    return 2;
  }

  /* Loop through argument */
  for (rc = 1; rc < argc; ++rc) {
    if (argv[rc][0] == '+') {
      sscanf(argv[rc]+1,"%d",&evtskip);
      printf(" skipping first %d events...\n",evtskip);
      fflush(stdout);
    }    
    // verbose mode
    else if (strcmp("-v", argv[rc]) == 0) {
      vmode = 1;
    }
    else if (argv[rc][0] == '-') {
      if (isdigit(argv[rc][1])) {
	sscanf(argv[rc]+1,"%d",&evtnum);
	printf(" displaying %d events.\n",evtnum);
	fflush(stdout);
      }
      else if ( (type=eventIdFromName(argv[rc]+1)) )
	addBankList(dontBanks, type);
      else
	fprintf(stderr, "%s: Ignoring unrecognized bank name: %s\n", argv[0], argv[rc]);
    }
    else
      pushFile(argv[rc]);
  }

  /* Loop through files on command line */
  while ( (name=pullFile()) ) {
    if (evtnum && (evtread-evtskip > evtnum)) break;

    if ( (rc=dstOpenUnit(unit, name, mode)) ) {
      fprintf(stderr, "%s: Error %d: failed to open for reading dst file: %s\n", argv[0], rc, name);
      return 1;
    }
    if(vmode)
      printf("Reading DST file: %s\n", name);

    /* Loop through events in file */
    while ((rc = eventRead(unit, wantBanks, gotBanks, &event)) > 0) {
      evtread++;
      if (evtread > evtskip) {
	if (evtnum && (evtread-evtskip > evtnum)) break;
	if (event) {
	  difBankList(gotBanks,dontBanks);
	  if (cntBankList(gotBanks))
	    listBanks(gotBanks,stdout,0);
	  else
	    fprintf(stdout,"--");
	  fprintf(stdout,"\n");
	}
	else
	  fprintf(stdout,"DST Banks not in an event.\n");
      }
    }
    dstCloseUnit(unit);
  }
  return SUCCESS;
}

// Function to create a string of bank names from list
void listBanks(integer4 banks, FILE *fp, int split_string) {
  integer4 i;
  integer4 type;
  char bankName[32];
  integer4 bankNameSize = sizeof(bankName);
  integer4 bankNameLen;
  integer4 line;
  const integer4 lineLength=78;

  // Loop over bank list.
  for (i=0,line=0; (type = itrBankList(banks,&i))>0;) {
    eventNameFromId(type, bankName, bankNameSize);
    bankNameLen = strlen(bankName);

    if(split_string)
      {
	if ((line+bankNameLen+2 > lineLength)) 
	  {
	    fprintf(fp,",\n");
	    line = 0;
	  }
	else if(i>1)
	  {
	    fprintf(fp,", ");
	    line += 2;
	  }
      }
    else if (i>1)
      {
	fprintf(fp,",");
	line += 1;
      }
 
    fprintf(fp,"%s",bankName);
    line += bankNameLen;
  }
}
