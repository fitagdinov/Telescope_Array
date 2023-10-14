#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sddstio.h"
#include "filestack.h"


static bool parseCmdLine(int argc, char **argv);

// check and make sure that the waveform is not corrupted
static bool chk_wfm(SDEventSubData *wf);

int main(int argc, char **argv)
{
  sddstio_class *dstio;
  char *infile;
  char igwf_trigp[tasdevent_ndmax];
  int xxyy[16];
  int usec[16];
  char sdfound[16];
  char wffound[16];
  char wfigood[16];
  int npat;
  int ipat;
  int iwf;
  int usec1;
  int usec2;
  int i_not_found[16];
  int n_not_found;
  int i_bad_wf[16];
  int n_bad_wf;
  FILE *fp;
  if (!parseCmdLine(argc,argv))
    return -1; 
  dstio = new sddstio_class();
  fp = stdout;
  while((infile=pullFile()))
    {
      if(!dstio->openDSTinFile(infile))
	return -1;
      while(dstio->readEvent())
	{	  
	  if(!dstio->haveBank(TASDEVENT_BANKID))
	    {
	      fprintf(stderr,"tasdevent_ is absent\n");
	      continue;
	    }
	  fprintf(fp,"eventRead: %d %06d %06d.%06d\n",  
		  tasdevent_.site,tasdevent_.date,
		  tasdevent_.time,tasdevent_.usec);
	  npat=0;
	  for (ipat=0; ipat<16; ipat++)
	    {
	      if (tasdevent_.pattern[ipat] == 0)
		break;
	      xxyy[npat] = (((tasdevent_.pattern[ipat]>>20)&0x3f)+(100*((tasdevent_.pattern[ipat]>>26)&0x3f)));
	      usec[npat] = (tasdevent_.pattern[ipat]&0xfffff);
	      npat++;
	    }
	 
	  memset( sdfound,     0,   sizeof (sdfound)       );
	  memset( wffound,     0,   sizeof (wffound)       );
	  memset( wfigood,     1,   sizeof (wfigood)       );
	  memset( igwf_trigp,  1,   sizeof (igwf_trigp)    );
	  // Go over each counter in the pattern & make sure they are present in the 
	  // event readout. Also check for corrupted waveforms for SDs that are int trigger pattern
	  for (ipat=0; ipat<npat; ipat++)
	    {
	      for (iwf=0; iwf<tasdevent_.num_trgwf; iwf++)
	        {     
		  if (tasdevent_.sub[iwf].lid == xxyy[ipat])
		    {
		      sdfound[ipat] = 1;
		      usec1 = (int)
			floor(((double)tasdevent_.sub[iwf].clock)/
			      ((double)tasdevent_.sub[iwf].max_clock)*1.0e6);
		      usec2 = (int)
			ceil(((double)tasdevent_.sub[iwf].clock)/
			     ((double)tasdevent_.sub[iwf].max_clock)*1.0e6 + 2.56);
		      if(usec1 <= (double)usec[ipat] && (double)usec[ipat] <= usec2)
			{
			  wffound[ipat] = 1;
			  wfigood[ipat] = (chk_wfm(&tasdevent_.sub[iwf]) ? 1 : 0);
			  if(wfigood[ipat]==0) 
			    igwf_trigp[iwf] = 0;
			  break;
			}
		    }
	        }
	    }
	  
	  
	  ///// CHECK THAT THE PATTERNS HAVE BEEN FOUND ///////////
	  n_not_found = 0;
	  n_bad_wf = 0;
	  for (ipat=0; ipat<npat; ipat++)
	    {
	      if(sdfound[ipat]==0 || wffound[ipat]==0)
		{
		  i_not_found[n_not_found] = ipat;
		  n_not_found++;
		}
	      if (wfigood[ipat] == 0)
		{
		  i_bad_wf[n_bad_wf] = ipat;
		  n_bad_wf ++;
		}
	    }
	  if (n_not_found > 0 || n_bad_wf > 0)
	    {
	      fprintf(stdout, "\n********************* EVENT START ************************\n");
	      fprintf(fp, "TRIGP_PROBLEMS: %d %06d %06d.%06d",
		      tasdevent_.site,tasdevent_.date,
		      tasdevent_.time,tasdevent_.usec);
	      if (n_not_found > 0)
		{
		  fprintf(fp, " NOT_FOUND:");
		  for (int ii=0; ii<n_not_found; ii++)
		    {
		      ipat=i_not_found[ii];
		      fprintf(fp," %04d:%06d",xxyy[ipat],usec[ipat]);
		    }
		  if (n_bad_wf == 0)
		    fprintf(fp,"\n");
		}
	      if (n_bad_wf > 0)
		{
		  fprintf(fp," BAD_WAVEFORMS:");
		  for (int ii=0; ii<n_bad_wf; ii++)
		    {
		      ipat=i_bad_wf[ii];
		      fprintf(fp," %04d:%06d",xxyy[ipat],usec[ipat]);
		    }
		  fprintf(fp,"\n");
		}
	      for (iwf=0; iwf<tasdevent_.num_trgwf; iwf++)
		{     
		  usec1 = (int)
		    floor(((double)tasdevent_.sub[iwf].clock)/
			  ((double)tasdevent_.sub[iwf].max_clock)*1.0e6);
		  usec2 = (int)
		    ceil(((double)tasdevent_.sub[iwf].clock)/
			 ((double)tasdevent_.sub[iwf].max_clock)*1.0e6 + 2.56);
		  fprintf(fp, "%04d:%06d-%06d\n",
			  tasdevent_.sub[iwf].lid,usec1,usec2);
		  if (igwf_trigp[iwf] == 0)
		    {
		      fprintf(fp, 
			      "wf_id = %d lid %04d clock %08d max_clock %08d usum %06d lsum %06d\n",
			      tasdevent_.sub[iwf].wf_id,
			      tasdevent_.sub[iwf].lid,
			      tasdevent_.sub[iwf].clock,
			      tasdevent_.sub[iwf].max_clock,
			      tasdevent_.sub[iwf].usum,
			      tasdevent_.sub[iwf].lsum);
		      fprintf(fp,"uwf[%03d]:\n",iwf);
		      for (int ii=0; ii<128; ii++)
			{
			  fprintf(fp,"%6d ",tasdevent_.sub[iwf].uwf[ii]);
			  if((ii+1)%16 == 0)
			    fprintf(fp,"\n");
			}
		      fprintf(fp,"lwf[%03d]:\n",iwf);
		      for (int ii=0; ii<128; ii++)
			{
			  fprintf(fp,"%6d ",tasdevent_.sub[iwf].lwf[ii]);
			  if((ii+1)%16 == 0)
			    fprintf(fp,"\n");
			}
		    }
		}
	      fprintf(stdout, "\n********************* EVENT END ************************\n");
	    }
	}
      dstio->closeDSTinFile();
    }
  fprintf(stdout,"\nDone\n");
  return 0;
}

bool parseCmdLine(int argc, char **argv)
{

  int i;
  char *line;
  char inBuf[0x400];
  FILE *fp;
  
  if(argc == 1) 
    {
      fprintf(stdout, "\nUsage: %s [in_file1 ...] and/or -i [list file] \n", argv[0]);
      fprintf(stderr, "Pass input dst file names as arguments without any prefixes\n");
      fprintf(stderr, "-i:    Specify the want file (with dst files)\n");
      fprintf(stderr, "--tts: Or get input dst file names from stdin\n");
      return false;
    }
  for (i = 1; i < argc; i++)
    {
      if (strcmp("-i", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              printf("Specify the list file!\n");
              return false;
            }
          else
            {
              if ((fp = fopen(argv[i], "r")) == NULL)
                {
                  fprintf(stderr, "can't open %s\n", argv[i]);
                  return false;
                }
              else
                {
                  while (fgets(inBuf, 0x400, fp))
                    {
                      if (((line = strtok(inBuf, " \t\r\n"))) && (strlen(line) > 0))
                        {
                          if (pushFile(line) != SUCCESS)
                            return false;
                        }
                    }
                  fclose(fp);
                }
            }
        }
      else if (strcmp("--tts", argv[i]) == 0)
        {
          while (fgets(inBuf, 0x400, stdin))
            {
              if (((line = strtok(inBuf, " \t\r\n"))) && (strlen(line) > 0))
                {
                  if (pushFile(line) != SUCCESS)
                    return false;
                }
            }
        }
      else if (argv[i][0] != '-')
        {
          if (pushFile(argv[i]) != SUCCESS)
            return false;
        }
      else
        {
          fprintf(stderr, "'%s': unrecognized option\n", argv[i]);
          return false;
        }
    }
  if (!countFiles())
    {
      fprintf(stderr,"no input files\n");
      return false;
    }
  return true;
}

bool chk_wfm(SDEventSubData *wf)
{
  int i;
  for (i=0; i<128; i++)
    {
      if (wf->uwf[i]<0)
	return false;
      if (wf->lwf[i]<0)
	return false;
    }
  if (wf->wf_id < 0)
    return false;
  if (((wf->lid / 100) < 1 || (wf->lid % 100) < 1))
    return false;
  if (wf->clock < 1)
    return false;
  if (wf->max_clock < 1)
    return false;
  return true;
}
