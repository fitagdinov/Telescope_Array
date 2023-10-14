/* Last modified: DI 20171206 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#include "dst_err_codes.h"
#include "event.h"
#include "bank_list.h"

static integer4 inUnit = 1;
static integer4 inMode = MODE_READ_DST;
static integer4 outUnit = 2;
static integer4 outMode = MODE_WRITE_DST;

char infile[0x400];
char outbname[0x400];
char outfile[0x400+0x20];
int nevents;

void parseCmdLine (int argc, char **argv);

int main (int argc, char *argv[])
{
  integer4 rc, size, event;
  integer4 wantBanks, gotBanks;
  int ievent,ifile;
  FILE *fl;
  
  parseCmdLine (argc, argv);
  

  if (dst_open_unit_ (&inUnit,infile,&inMode)) 
    {
      fprintf (stderr, "can't open DST input file %s\n",infile);
      exit(3);
    }

  /* initialize bank lists */
  size = n_banks_total_ ();
  wantBanks = new_bank_list_ (&size);
  event_all_banks_(&wantBanks);
  gotBanks = new_bank_list_ (&size);
  ievent = 0;
  ifile = 0;
  while ((rc = event_read_ (&inUnit, &wantBanks, &gotBanks, &event)) > 0) 
    {
      if (event) 
	{
	  // Start a new dst file
	  if (ievent % nevents == 0)
	    {
	      ifile++;
	      sprintf (outfile,"%s_%03d.dst.gz",outbname,ifile);
	      if (!(fl=fopen(outfile,"w")))
		{
		  fprintf (stderr, "can't start %s\n",outfile);
		  exit (3);
		} 
	      if (ievent > 0) 
		dst_close_unit_ (&outUnit);
	      
	      if (dst_open_unit_ (&outUnit,outfile,&outMode)) 
		{
		  fprintf (stderr, "can't start %s\n",outfile);
		  exit(3);
		} 
	      fprintf (stdout, "started: %s\n",outfile);
	      fflush(stdout);
	    }
	  event_write_(&outUnit, &gotBanks, &event);
	  ievent ++;
	}
    }
  
  dst_close_unit_ (&inUnit);
  
  if (ievent > 0)
    dst_close_unit_ (&outUnit);

  fprintf (stdout, "events: %d files: %d\n",ievent,ifile);
  fflush(stdout);
  
  return 0;
}
  
void parseCmdLine (int argc, char **argv)
{
  integer4 i;
  FILE *fl;
  infile[0] = '\0';
  nevents=0;
  if (argc != 7) 
    {
      integer4 rc;
      fprintf (stderr,"\nusage: %s -i [input_dst_name] -o [output_file_base] -n [nevents/file]\n", argv[0]);
      fprintf (stderr, "Partition a dst file into smaller parts\n");
      fprintf (stderr,"  -i:   input dst file\n");
      fprintf (stderr,"  -o:   output file base (will add '_???.dst.gz' 3-digit suffixes)\n");
      fprintf (stderr,"  -n:   number of events per dst file\n\n");
      fputs("\nCurrently recognized banks:", stderr);
      dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
      fprintf(stderr,"\n\n");
      exit (2);
    }
  
  for (i = 1; i < argc; ++i) {
    if (strcmp (argv[i], "-i") == 0) 
      {
	if (++i >= argc || argv[i][0] == '-') 
	  {
	    fprintf (stderr, "specify the input dst file\n");
	    exit (1);
	  }
	else 
	  {
	    strcpy(infile,argv[i]);
	    // dst opener doesn't always catch the i/o errors
	    if (!(fl=fopen(infile,"r")))
	      {
		fprintf(stderr,"can't open %s\n",infile);
		exit(1);
	      }
	    else
	      {
		fclose(fl);
	      }
	  }
      }
    else if(strcmp (argv[i], "-o") == 0) 
      {
	if(++i >= argc || argv[i][0] == '-') 
	  {
	    fprintf (stderr, "specify the output file base\n");
	    exit (1);
	  }
	strcpy(outbname, argv[i]);
      }
    else if(strcmp (argv[i], "-n") == 0) 
      {
	if(++i >= argc || argv[i][0] == '-') 
	  {
	    fprintf (stderr, "specify the number of events\n");
	    exit (1);
	  }
	nevents = atoi(argv[i]);
	if(nevents <= 0) {
	  fprintf (stderr, "invalid number of events given\n");
	  exit(1);
	}
      }
    else 
      {
	fprintf (stderr, "%s: unrecognized option\n", argv[i]);
	exit (1);
      }
  }
  if (infile[0] == '\0') 
    {
      fprintf (stderr, "input dst file not specified\n");
      exit (1);
    }
  if (nevents == 0)
    {
      fprintf(stderr,"number of events not specified\n");
      exit(1);
    }
}
