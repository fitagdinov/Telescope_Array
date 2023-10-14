#define MAIN
#include "nuf.h"


/*************************************************************
nuf - neutrino finder by Grigory Rubtsov, grisha@ms2.inr.ac.ru
      based on the sditerator by Dmitry Ivanov      
**************************************************************/

int main(int argc, char **argv)
{
  sddstio_class sdrun;              // to handle the run DST files
  char *dInFile;                 // pass1 dst input files
  FILE *fl;                      // dummy file pointer, for checking if the files exist

  // parses cmd line
  if(!cmd_opt.getFromCmdLine(argc,argv)) 
    return -1;

  if(cmd_opt.isset("dtf")) {
    mkdir(OUTDIR, 0755);
  }

  
  // Go over each run DST files and analyze the events/monitoring cycles in them.
  while((dInFile=pullFile()))
    {
      if(cmd_opt.isset("v")) {
	fprintf(stderr,"%s\n", "DATA FILE:");
	fprintf(stderr,"%s\n",dInFile);
      }

      // Make sure that the input files exist, since right now, DST opener 
      // doesn't check it, in the case of .gz files.
      if((fl=fopen(dInFile,"r"))==NULL)
	{
	  pIOerror;
	  fprintf(stderr,"Can't open %s\n",dInFile);
	  return -1;
	}
      fclose(fl);
      
      //      int old_stdout = dup(1); // suppress printout from openDSTfile()
      //      freopen ("/dev/null", "w", stdout); 
      if(!sdrun.openDSTinFile(dInFile)) return -1;       // Open event DST file
      //      fclose(stdout);
      // stdout = fdopen(old_stdout, "w"); 

      while(sdrun.readEvent())	{ // Go over all the events in the run.
	  iter(sdrun);
      }       
      sdrun.closeDSTinFile();  // close pass1 event DST files.
      
    }

  if(cmd_opt.isset("v")) {
    fprintf(stderr,"\n\nDone\n");
  }
  return 0;
}
