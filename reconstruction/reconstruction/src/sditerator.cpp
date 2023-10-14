#include "sditerator.h"

int main(int argc, char **argv)
{
  listOfOpt opt;                 // to handle the program arguments.
  sddstio_class dstio;           // to handle the DST files 
  char *dInFile;                 // dst input files
  FILE *outFl;                   // output file pointer
  
  
  // parses the command line
  if(!opt.getFromCmdLine(argc,argv)) 
    return -1;
  
  opt.printOpts();


  // Prepare the outputs
  if ((outFl=fopen(opt.outFile,"w"))==NULL)
    {
      fprintf(stderr,"Can't start %s\n",opt.outFile);
      exit(2);
    }
  
  // Go over each run DST files and analyze the events/monitoring cycles in them.
  while((dInFile=pullFile()))
    {
      
      // Open DST files
      if(!dstio.openDSTinFile(dInFile)) 
	return -1;
      
      // Go over all the events in the run.
      while(dstio.readEvent())
	{
	  cppanalysis(outFl); // Carry out an analysis in c++ 
	} // while(dstio.readEvent ...
      
      dstio.closeDSTinFile();  // close pass1 event DST files.
      
    }
  fclose(outFl);
  fprintf(stdout,"\n\nDone\n");
  return 0;
}
