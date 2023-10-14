/*
 * $Source: /hires_soft/uvm2k/pro/dstdump.c,v $
 * $Log: dstdump.c,v $
 *
 * Last modified: DI 20171206
 *
 * Revision 1.18  2002/01/29 00:53:43  ben
 * Fixed the problem with evtskip being un-initialized!
 *
 * Revision 1.17  2002/01/21 19:16:55  hires
 * Add option[#NNN] to skip NNN events (boyer)
 *
 * Revision 1.16  2001/04/18 22:49:13  bellido
 * Already exist more than 100 banks created.
 * actually the bank list was initialized with rc=100
 * now it is initialized with 'rc=150'. This change
 * allows to dump all banks and not only the first
 * 100 banks.
 *
 * Revision 1.15  1996/02/15 16:14:09  mjk
 * Fix a bug by changing stdout to stderr in one of the fputs statements.
 * Before the fix the recognized bank name part of the help message was
 * messed up on the decstations. Also added <string.h> include.
 *
 * Revision 1.14  96/01/22  22:27:12  jeremy
 * fixed bug.
 * 
 * Revision 1.13  1996/01/17  05:14:55  mjk
 * Moved pushFile and pullFile into filestack.c utility routines
 * Replace file return with SUCCESS instead of 0
 *
 * Revision 1.12  1995/07/21  16:26:29  jeremy
 * Prettified help message.
 *
 * Revision 1.11  1995/07/20  23:54:39  jeremy
 * Changed option syntax for more flexability.
 *
 * Revision 1.10  1995/07/14  18:27:02  jeremy
 * Changed to use new event functions. It should no longer be necessary to edit
 * this file to add new bank types. Just recompile with new uti library.
 *
 * Revision 1.9  1995/07/13  18:20:52  jeremy
 * Added PFIT1 & PFIT2 bank types.
 *
 * Revision 1.8  1995/07/06  20:12:55  mjk
 * Add switch -t for short from of TIM1 bank
 *
 * Revision 1.7  1995/06/21  11:29:40  mjk
 * more CRNN stuff
 *
 * Revision 1.6  95/06/20  02:17:57  mjk
 * Updated to handle crnn bank
 * 
 * Revision 1.5  95/06/16  18:04:43  jeremy
 * update for all current bank types
 * 
 * Revision 1.4  1995/06/15  20:36:29  jeremy
 * added casa bank types
 *
 * Revision 1.3  1995/05/24  18:56:48  jeremy
 * Rewrote program to use event read and event write functions.
 *
 * Revision 1.2  1995/04/21  22:14:41  jeremy
 * dstdump now uses wrapper program in uti library.
 *
*/

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#include "dst_err_codes.h"

#include "event.h"
#include "filestack.h"


int main(int argc, char *argv[]) {

  integer1 *name;
  integer4 type, rc, event;
  integer4 wantBanks, gotBanks;
  integer4 unit = 1, mode = MODE_READ_DST;
  integer4 evtread=0,evtskip=0;

  /* initialize bank lists */
  rc = n_banks_total_();
  wantBanks = new_bank_list_(&rc);
  gotBanks = new_bank_list_(&rc);

  /* print help message if no arguments given */
  if (argc == 1)
    {
      fprintf(stderr, "\nUsage: %s [flags] dstfile ...\n\n", argv[0]);
      fputs("Flags:\n", stderr);
      fputs("  -name: Show short form of bank 'name' (if supported, otherwise long)\n",
	    stderr);
      fputs("  +name: Show long form of bank 'name'\n", stderr);
      fputs("  -all:  Show short form of all banks\n", stderr);
      fputs("  +all:  Show long form of all banks\n", stderr);
      fputs("  #NNN:  Skip the first NNN events\n", stderr);
      fputs("\nDefault (no flags): Show long form of all banks\n", stderr);
      fputs("\nExamples:\n", stderr);
      fprintf(stderr,
	      "  %s file.dst             - dump all banks in long format.\n", argv[0]);
      fprintf(stderr,
	      "  %s -raw1 +pho1 file.dst - dump raw1 in short & pho1 in long format.\n",
	      argv[0]);
      fputs("\nCurrently recognized bank names:", stderr);
      event_all_banks_(&gotBanks);
      dsc_bank_list_  (&gotBanks,stderr);
      fprintf(stderr, "\n\n");
      return 2;
    }
  for (rc = 1; rc < argc; ++rc)
    {
      if (argv[rc][0] == '#') {
        sscanf(argv[rc]+1,"%d",&evtskip);
        printf(" skipping first %d events...\n",evtskip);
        fflush(stdout);
      }
      else if (argv[rc][0] != '-' && argv[rc][0] != '+')
	pushFile(argv[rc]);
      else
	{
	  if (strcmp(argv[rc] + 1, "all") == 0)
	    {
	      event_all_banks_(&wantBanks);
	      if (argv[rc][0] == '-')
		event_all_banks_(&gotBanks);
	      else
		clr_bank_list_(&gotBanks);
	    }
	  else if ( (type = event_id_from_name_(argv[rc] + 1)) )
	    {
	      add_bank_list_(&wantBanks, &type);
	      if (argv[rc][0] == '-')
		add_bank_list_(&gotBanks, &type);
	      else
		rem_bank_list_(&gotBanks, &type);
	    }
	  else
	    fprintf(stderr, "%s: Ignoring unrecognized bank name: %s\n", argv[0], argv[rc]);
	}
    }
  event_set_dump_format_(&wantBanks, (type = 1, &type)); /* default long format */
  if (cnt_bank_list_(&gotBanks))
    event_set_dump_format_(&gotBanks, (type = 0, &type)); /* short format */

  while ( (name = pullFile()) )
    {
      if ( (rc = dst_open_unit_(&unit, name, &mode)) )
	{
	  fprintf(stderr, "%s: Error %d: failed to open for reading dst file: %s\n",
		  argv[0], rc, name);
	  return 1;
	}
      printf("Reading DST file: %s\n", name);
      while ((rc = event_read_(&unit, &wantBanks, &gotBanks, &event)) > 0)
	{
          evtread++;
          if (evtread > evtskip) {
            if (event)
              puts("START OF EVENT ***********************************************************");
            event_dump_(&gotBanks);
            if (event)
              puts("END OF EVENT *************************************************************");
          } else if (evtread%1000 == 0) {
            printf("evtnum %d\n",evtread);
            fflush(stdout);
          }
	}
      if (rc != END_OF_FILE)
	{
	  fprintf(stderr, "%s: Error %d: failed to read bank from dst file: %s\n",
		  argv[0], rc, name);
	  return 1;
	}
      dst_close_unit_(&unit);
    }
  return SUCCESS;
}

