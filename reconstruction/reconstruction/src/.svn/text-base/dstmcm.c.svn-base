/* Last modified: DI 20171206 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "event.h"

#define DSTMCM_NINFILES 4

static const int br = 0;
static const int lr = 1;
static const int md = 2;
static const int sd = 3;
static const char *dstmcm_names[DSTMCM_NINFILES] = {"BR", "LR", "MD", "SD"};
static char infile[DSTMCM_NINFILES][0x400];
static char outfile[0x400];
integer4 fMode;     // force-overwrite mode

/* buffers to store the trumpmc information from the BR, LR detectors */
static trumpmc_dst_common trumpmc_buffer[2]; // [0] - BR, [1] - LR

static int parseCmdLine(int argc, char **argv);

/* 
   Here, it is assumed that the trump MC has been thrown in the mono mode for each detector
   so that nSites in trumpmc for BR, LR is always 1 and one needs to use only the [0]'the element
   of every [MXS]-max-size array in the mono trumpmc banks.
   INPUTS: 
   pfdm is the pointer to the array of trumpmc_dst_common with 2 elements, for BR and LR mono trumpmc banks. 
   has_triggered is the array of integers with 2 elemtents to indicate which FD mono has triggered
   OUTPUT:
   pfds is a pointer to the trumpmc_dst_common variable which will contain the combined stereo trumpmc bank
*/
static void trumpmc_mono2str(trumpmc_dst_common *pfds, trumpmc_dst_common *pfdm, int *has_triggered);

int main(int argc, char **argv)
{
  integer4 ievent,ifile,size,rc,event,mode;
  integer4 wantBanks,writeBanks;
  integer4 inUnit[DSTMCM_NINFILES],gotBanks[DSTMCM_NINFILES];
  integer4 outUnit;
  const integer4 ninfiles=DSTMCM_NINFILES;
  FILE *fl;
  int has_triggered[2]; /* to determine if BR, LR have triggered, [0] - BR, [1] - LR */
  
  if (parseCmdLine(argc, argv) != 0)
    return 1;
  
  size = nBanksTotal();
  
  wantBanks = newBankList(size);
  eventAllBanks(wantBanks);
  
  for (ifile=0; ifile < ninfiles; ifile++)
    gotBanks[ifile] = newBankList(size);
  
  writeBanks = newBankList(size);
  
  for ( ifile=0; ifile < ninfiles; ifile++)
    inUnit[ifile] = ifile+1;
  outUnit = ninfiles+1;
  
  
  
  // open all the input files
  for (ifile=0; ifile < ninfiles; ifile ++ )
    {
      if (!(fl=fopen(infile[ifile],"r")))
	{
	  fprintf(stderr,"can't open %s\n",infile[ifile]);
	  return 2;
	}
    }
  fclose(fl);
  mode = MODE_READ_DST;
  for (ifile=0; ifile<ninfiles; ifile++)
    {
      if ((rc=dstOpenUnit(inUnit[ifile],&infile[ifile][0],mode)) != 0)
	{
	  fprintf(stderr,"can't dst-open %s for reading\n",infile[ifile]);
	  return 2;
	}
    }
  if ( ((fl=fopen(outfile,"r"))) && (!fMode))
    {
      fprintf(stderr,"%s: file exists\n",outfile);
      return 2;
    }
  if(fl)
    {
      fclose(fl);
      fl = 0;
    }
  if (!(fl=fopen(outfile,"w")))
    {
      fprintf(stderr,"can't start %s\n",outfile);
      return 2;
    }
  fclose(fl);
  mode = MODE_WRITE_DST;
  if ((rc=dstOpenUnit(outUnit, outfile, mode)) != 0)
    {
      fprintf(stderr,"can't dst-open %s for writing\n",outfile);
      return 2;
    }
  
  ievent=0;
  // read all events in the 1st input files ( BR FD )
  while ((rc = eventRead (inUnit[br], wantBanks, gotBanks[br], &event)) > 0) 
    {
      if (!event)
	{
	  fprintf(stderr,"corrupted event!\n");
	  return 3;
	}
      
      clrBankList(writeBanks);
      cpyBankList(writeBanks, gotBanks[br]);
      // Save the BR trumpmc bank into the buffer if BR has triggered
      has_triggered[br] = 0;
      if (tstBankList(gotBanks[br],BRRAW_BANKID))
	{
	  memcpy(&trumpmc_buffer[br],&trumpmc_, sizeof(trumpmc_dst_common));
	  has_triggered[br] = 1;
	}
      
      // read an LR FD event
      rc = eventRead (inUnit[lr], wantBanks, gotBanks[lr], &event);
      if ( (rc <= 0) || (!event)) 
	{
	  fprintf (stderr, "Error: '%s' has less events than '%s'\n", infile[lr],infile[br]);
	  return 3;
	}
      sumBankList(writeBanks, gotBanks[lr]);
      // Save the LR trumpmc bank into the buffer if LR has triggered
      has_triggered[lr] = 0;
      if (tstBankList(gotBanks[lr],LRRAW_BANKID))
	{
	  memcpy(&trumpmc_buffer[lr],&trumpmc_, sizeof(trumpmc_dst_common));
	  has_triggered[lr] = 1;
	}

      
      // looping over the remaining two detectors: MD and SD
      for (ifile=2; ifile<4; ifile++)
	{ 
	  rc = eventRead (inUnit[ifile], wantBanks, gotBanks[ifile], &event);
	  if ( (rc <= 0) || (!event)) 
	    {
	      fprintf (stderr, "Error: '%s' has less events than '%s'\n",
		       infile[ifile],infile[0]);
	      return 3;
	    }
	  sumBankList(writeBanks, gotBanks[ifile]);
	}


      /////////////// Sanity checks ///////////////////////////////

      if (ievent==0)
	{
	  if (!tstBankList(gotBanks[br],TRUMPMC_BANKID) || 
	      !tstBankList(gotBanks[lr],TRUMPMC_BANKID) ||
	      !tstBankList(gotBanks[md],MC04_BANKID)    ||
	      !tstBankList(gotBanks[sd],RUSDMC_BANKID)
	      )
	    {
	      fprintf(stderr,"input files mixed up (br/lr/md/sd) ?\n");
	      return 3;
	    }
	}
      if (tstBankList(gotBanks[br],LRRAW_BANKID) || tstBankList(gotBanks[lr], BRRAW_BANKID))
	{
	  fprintf(stderr,"br/lr are mixed up\n");
	  return 3;
	}
      
      
      /********** REMOVE ANY UNNECESSARY BANKS FROM THE COMBINED BANK LIST HERE *********/
      
      // Remove the trumpmc bank from the writeBanks only if both BR, LR did not trigger;
      // otherwise, combine the trumpmc into the proper stereo format (if one of the 
      // detectors did not trigger, it will have zeros for all the variables. That's
      // inefficient in terms of storage, since now one uses 2x the space, but it is more 
      // intuitive)
      if (!has_triggered[br] && !has_triggered[lr])
	remBankList(writeBanks,TRUMPMC_BANKID);
      else
	trumpmc_mono2str(&trumpmc_,trumpmc_buffer,has_triggered);
      
      // MD didn't trigger
      if (!tstBankList(gotBanks[md],MCRAW_BANKID))
	{
	  remBankList(writeBanks,MC04_BANKID);
	  remBankList(writeBanks,MCSDD_BANKID);
	}
	
      /**********************************************************************************/

      // write out the event which now should have the sum of all 
      // relevant banks from the events in the input files
      if ((rc=eventWrite(outUnit, writeBanks, TRUE)) < 0)
	{
	  fprintf(stderr,"failed to write an event \n");
	  return 3;
	}      
      ievent ++;
    }
  

  // Check the rest of the dst files.  
  // They must be on last event, if not - print an error message

  for ( ifile=1; ifile < ninfiles; ifile++)
    {
      rc = eventRead (inUnit[ifile], wantBanks, gotBanks[ifile], &event);
      if ( rc > 0 ) 
	{
	  fprintf (stderr, "Error: '%s' has more events than '%s'\n",
		   infile[ifile],infile[0]);
	  return 3;
	}
    }
  
  fprintf(stdout, "%d events\n",ievent);

  
  
  dstCloseUnit(outUnit);
  for (ifile=0; ifile < ninfiles; ifile++)
    dstCloseUnit(inUnit[ifile]);
  
  return 0;
}

int parseCmdLine(int argc, char **argv)
{

  int i;
  
  outfile[0] = '\0';
  fMode      = 0;
  
  for (i=0; i<DSTMCM_NINFILES;i++)
    infile[i][0] = '\0';
 
  
  if (argc==1)
    {
      integer4 rc;
      fprintf(stderr,"\nMerge and cleanup hybrid MC events\n");
      fprintf(stderr,"Also, merge the mono trumpmc into the proper stereo format, if relevant\n");
      fprintf(stderr,"\nUsage: %s -o [dst_file] -ibr [dst_file]  ...\n",argv[0]);
      fprintf(stderr,"-ibr:  BR MC file\n");
      fprintf(stderr,"-ilr:  LR MC file\n");
      fprintf(stderr,"-imd:  MD MC file\n");
      fprintf(stderr,"-isd:  SD MC file\n");
      fprintf(stderr,"-o:    Output dst file\n");
      fprintf(stderr,"-f:    (opt) don't check if the output file exists, just overwrite it\n\n");
      fputs("\nCurrently recognized banks:", stderr);
      dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
      fprintf(stderr,"\n\n");
      return -1;
    } 
    
  for(i=1; i<argc; i++)
    { 
      
      // output file
      if (strcmp ("-o", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"specify the output file!\n");
              return -1;
            }
          else
            {
              sscanf (argv[i], "%s", outfile);
            }
        }
      
      // br file
      else if (strcmp ("-ibr", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printf ("specify the BR MC file!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%s", infile[br]);
	    }
	}
      // lr file
      else if (strcmp ("-ilr", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printf ("specify the LR MC file!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%s", infile[lr]);
	    }
	}
      // md file
      else if (strcmp ("-imd", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printf ("specify the MD MC file!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%s", infile[md]);
	    }
	}
      // sd file
      else if (strcmp ("-isd", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printf ("specify the SD MC file!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%s", infile[sd]);
	    }
	}
      
      
      // fore overwrite mode ?
      else if (strcmp ("-f", argv[i]) == 0)
	fMode = 1;
      
      // everything else is junk
      else
	{
	  fprintf (stderr, "'%s': unrecognized option\n", argv[i]);
	  return -1;
	}
    }

  for (i=0; i<DSTMCM_NINFILES;i++)
    {
      if(infile[i][0] == '\0')
	{
	  fprintf(stderr,"%s input file not specified\n", dstmcm_names[i]);
	  return -1;
	}
    }
  if (outfile[0]=='\0')
    {
      fprintf(stderr,"output file not specified\n");
      return -1;
    }
  
  return 0;
}


static void trumpmc_mono2str(trumpmc_dst_common *pfds,trumpmc_dst_common *pfdm,int *has_triggered)
{
  int i, idepth, isite, imir, itube;
  int isite_notrig;  // site id that did not trigger
  int isite_trig;    // site id that did trigger
  
  
  // first, zero-out the entire bank
  memset (pfds, 0, sizeof(trumpmc_dst_common));
  
  // determine at least one site that did trigger, 
  // and a site that did not trigger (if any).  If both sites did not trigger,
  // then there is a problem: one can't call this routine unless at least one
  // site triggers.
  isite_notrig  =  -1;
  isite_trig    =  -1;
  for (isite = 0; isite < 2; isite ++)
    {
      if (has_triggered[isite])
	isite_trig   = isite;
      else
	isite_notrig = isite;
    }
  if (isite_trig == -1)
    {
      fprintf(stderr,"INTERNAL ERROR: trumpmc_mono2str is called but neither BR nor LR has triggered\n");
      exit (3);
    }
  
  /************* VARIABLES THAT DO NOT DEPEND ON THE SITE (BELOW) ************/
  
  /* since these variables don't depend on the FD site, we're using just one FD site ( but it must have triggered) */
  isite                                            =   isite_trig;
  
  for (i=0; i<3; i++)
    {
      pfds->impactPoint[i]                         =   pfdm[isite].impactPoint[i];
      pfds->showerVector[i]                        =   pfdm[isite].showerVector[i];
    }
  
  pfds->energy                                     =   pfdm[isite].energy;
  pfds->primary                                    =   pfdm[isite].primary;
  
  for (i=0; i<4; i++)
    pfds->ghParm[i]                                =   pfdm[isite].ghParm[i];
  
  pfds->nSites                                     =   2; /* we're combining two sites here: BR, LR */
  pfds->nDepths                                    =   pfdm[isite].nDepths;
  
  for (idepth=0; idepth < pfdm[isite].nDepths; idepth++)
    pfds->depth[idepth]                            =   pfdm[isite].depth[idepth];
  
  /************* VARIABLES THAT DO NOT DEPEND ON THE SITE (ABOVE) *************/
  
  /************* VARIABLES THAT DO DEPEND ON THE SITE (BELOW) *****************/  
  
  /* These two variables are filled for both BR,LR sites regardless of whether they've triggered or not */
  pfds->siteid[0]                                  =   0;  /* BR */
  pfds->siteid[1]                                  =   1;  /* LR */
  
  for (isite=0; isite < 2; isite++)
    {
      
      /* don't fill in any additional information for the site that did not trigger */
      if (isite == isite_notrig)
	continue;
      
      pfds->psi[isite]                             =   pfdm[isite].psi[0];
      
      for (i=0; i<3; i++)
	{
	  pfds->siteLocation[isite][i]             =   pfdm[isite].siteLocation[0][i];
	  pfds->rp[isite][i]                       =   pfdm[isite].rp[0][i];
	}
      
      pfds->nMirrors[isite]                        =   pfdm[isite].nMirrors[0];
      
      for (imir = 0; imir < pfds->nMirrors[isite]; imir ++)
	{
	  
	  pfds->mirror[isite][imir]                =   pfdm[isite].mirror[0][imir]; 
	  
	  for (idepth = 0; idepth < pfds->nDepths; idepth ++)
	    {
	      pfds->fluoFlux[isite][imir][idepth]  =   pfdm[isite].fluoFlux[0][imir][idepth];
	      pfds->aeroFlux[isite][imir][idepth]  =   pfdm[isite].aeroFlux[0][imir][idepth];
	      pfds->raylFlux[isite][imir][idepth]  =   pfdm[isite].raylFlux[0][imir][idepth];
	      pfds->dirCFlux[isite][imir][idepth]  =   pfdm[isite].dirCFlux[0][imir][idepth];
	    }
	  
	  pfds->totalNPEMirror[isite][imir]        =   pfdm[isite].totalNPEMirror[0][imir];
	  pfds->nTubes[isite][imir]                =   pfdm[isite].nTubes[0][imir];
	  
	  for (itube = 0; itube <  pfds->nTubes[isite][imir]; itube ++)
	    {
	      pfds->tube[isite][imir][itube]       =   pfdm[isite].tube[0][imir][itube];
	      pfds->aveTime[isite][imir][itube]    =   pfdm[isite].aveTime[0][imir][itube];
	      pfds->totalNPE[isite][imir][itube]   =   pfdm[isite].totalNPE[0][imir][itube];
	    }
	}
    }
  
  /************* VARIABLES THAT DO DEPEND ON THE SITE (ABOVE) ****************/
  

  
}
