#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <assert.h>
#include <libgen.h>
#include <errno.h>
#include "sdmc.h"
#include "sdmc_tadate.h"



#define DISTMAX 8400 /* Meters */
#define NX DISTMAX/3
#define NY DISTMAX/3
#define Z0 1430.e2 /* cm */
#define ERAD 6.3707e8
#define TMAX 500
#define DETDIST 25.e5
#define NSECT 3
#define DISTMIN 502
#define VEM2MeV 2.05
#define MeVPOISSON 9.
#define NWORD 273
#define FADC_TOT 4000
#define FADC_START 2000
#define ATMOSRATE 0.1225
#define TRIG_START 24
#define FADC_NONLIN1 2000.
#define FADC_NONLIN2 600.
#define FADC_MAX 4095.
#define ATTEN 0.05
#define INDXTHR -2.0
#define DRAWER 1.2589254
#define DRAWER_HF 1.122018
#define NATMOS 30297456
#define NEVENT 10000000
#define DT 20
#define CLFZ 1391.e2


int main(int argc, char **argv)
{
  
  
  FILE *f = 0;
  float calbuf[NCAL];
  int n_calib    = 0;
  int n_live     = 0;
  
  int i=0, j = 0;
  int NDATE1, NDATE2, dataset=0;
  char *calFile = 0;
  int sector[NDATE*144][NCOUNTERS];
  
  
  // print manual
  if (argc != 3)
    {
      fprintf(stderr,"\nUsage: %s [1] [2]\n", argv[0]);
      fprintf(stderr,"[1]: <int> calibration epoch number\n");
      fprintf(stderr,"[2]: <str> sdcalib_*.bin binary calibration file prepared for sdmc\n");
      fprintf(stderr,"\n");
      exit(1);
    }
  
  for ( i = 0; i < NDATE*144; i++ )
    for ( j = 0; j < NCOUNTERS; j++ )
      deadtime[i][j] = 1;
  sscanf(argv[1], "%d", &dataset);
  NDATE2=24 + dataset*30;
  NDATE1=NDATE2-30;
  calFile = argv[2];
  if ((f=fopen(calFile, "rb")) == NULL)
    {
      fprintf(stderr, "Error: can't open file '%s'\n",calFile);
      exit(2);
    }

  // Load the calibration file into memory
  
  while (fread(calbuf, sizeof(float),NCAL,f)==NCAL)
    {
      i=(int)calbuf[0]-NDATE1*144;
      j=(int)calbuf[1];
      if ( i < 0 || i >= 144*(NDATE2-NDATE1) || j < 0 || j >= NCOUNTERS ) 
	{
	  fprintf(stdout,"i=%d / %d\n", i, 144*(NDATE2-NDATE1));
	  continue;
	}
      n_calib ++;
      if ( (int)calbuf[2] != 0 || calbuf[3] <= 0. || calbuf[4] <= 0. || 
	   calbuf[5] <= 0. || calbuf[6] <= 0. || calbuf[7] <= 0. || 
	   calbuf[8] <= 0. || calbuf[9] <= 0. || calbuf[10] <= 0. || 
	   calbuf[11] <= 0. || calbuf[12] <= 0. || calbuf[13] <= 0. ||
	   calbuf[14] <= 0. || 
	   calbuf[17] <= 0. || calbuf[18] <= 0. || calbuf[19] <= 0. ||
	   calbuf[20] <= 0. || calbuf[21] <= 0. || calbuf[22] <= 0. ||
	   calbuf[24] <= 0. || calbuf[25] <= 0.) 
	{
	  deadtime[i][j] = 1;
	}
      else
	{
	  mevpoisson[i][j][0] = calbuf[3];
	  mevpoisson[i][j][1] = calbuf[4];
	  one_mev[i][j][0] = calbuf[5];
	  one_mev[i][j][1] = calbuf[6];
	  mip[i][j][0] = calbuf[7];
	  mip[i][j][1] = calbuf[8];
	  fadc_ped[i][j][0] = calbuf[9];
	  fadc_ped[i][j][1] = calbuf[10];
	  fadc_noise[i][j][0] = calbuf[11];
	  fadc_noise[i][j][1] = calbuf[12];
	  pchped[i][j][0] = calbuf[13];
	  pchped[i][j][1] = calbuf[14];
	  lhpchped[i][j][0] = calbuf[15];
	  lhpchped[i][j][1] = calbuf[16];
	  rhpchped[i][j][0] = calbuf[17];
	  rhpchped[i][j][1] = calbuf[18];
	  mftndof[i][j][0] = calbuf[19];
	  mftndof[i][j][1] = calbuf[20];
	  mftchi2[i][j][0] = calbuf[21];
	  mftchi2[i][j][1] = calbuf[22];
	  sector[i][j] = (int)calbuf[23];
	  if(sector[i][j] < 0)
	    sector[i][j] = 0;
	  deadtime[i][j] = 0;
	  sat[i][j][0] = FADC_NONLIN1;
	  sat[i][j][1] = FADC_NONLIN1;
	  n_live++;
	}
    }
  fclose(f);
  fprintf(stdout,"file=%s epoch=%d n_calib=%d n_live=%d dead_fraction=%.1f%%\n",
	  calFile,
	  dataset,
	  n_calib,
	  n_live,
	  100.0*(1.0-((double)n_live)/(144.0*(double)(NDATE2-NDATE1)*(double)NCOUNTERS))
	  );
  fprintf(stdout,"\nDone\n");
  return 0;
  
}
