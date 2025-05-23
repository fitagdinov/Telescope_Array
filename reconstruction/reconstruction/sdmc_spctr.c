

/* This variable tells how much memory to use per sdmc_spctr process.
   20 is the optimum choice.  One should pass -DNT=20 (or some other number)
   on the command line when compiling the program. */
#ifndef NT
#define NT 20
#endif


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
#include "event.h"
#include "sdmc_tadate.h"
#include "sdmc_bsd_bitf.h"




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
#define INDX01 -2.0
#define INDX02 -2.0
#define BRKPNT  5.6234132519
#define DRAWER 1.2589254
#define DRAWER_HF 1.122018
#define NATMOS 30297456
#define N_AZI  1000000
#define DT 20
#define CLFZ 1391.e2

void gaussf(float pair[]);
void gaussd(double pair[]);
float poissonf(float lambda);
double poissond(double lambda);
void put2rusdraw();

int main(int argc, char **argv)
{
  
  
  // Poisson mean of the number of events to genearte 
  // in this instance of the program.  This is taken from the command line.
  double nEvent_to_throw_poisson_mean = 0.0;

  FILE *f;
  short sdmap[NX][NY][NT];
  unsigned short buf[6];
  int i=0, j, k, m, n, p, q, qq, r, sampletime[NCOUNTERS][NT][TMAX];
  int NDATE1, NDATE2, dataset=0, eventdate[NT], sdcoor[NCOUNTERS][4];
  int trigtable[NCOUNTERS][NT][FADC_TOT/2], yymmdd, hhmmss, kmax;
  int dtrigtable[NCOUNTERS][NT][FADC_TOT/2], nEvent_global, pnum;
  int tmptime, tmin[NT], col[NCOUNTERS], dtrigtime_tmp, natmos; 
  int nofwf[NCOUNTERS], time_offset, nEvent, corecounter[NT], wfnum, wfnum2; 
  int row[NCOUNTERS], sector[NDATE*144][NCOUNTERS], secttime[NT][NSECT], randint;
  int smear_energies = 1, n_azi = 0;
  
  float sdht[NCOUNTERS][NT], randnum[4], sdtrans[3], sdrot[2], trigsum[2];
  float trigsum2[2];
  float mev[NCOUNTERS][NT][TMAX][2], fadc[NCOUNTERS][NT][FADC_TOT][2];
  float eventbuf[NWORD], corexyz[NT][3], distmin, pztmp, phi[NT];
  float spctrindx = INDX02, diff, encor[NT], energy, showertime[NT];
  float mev_tmp, poisson_tmp, atmos[NATMOS][2], calbuf[NCAL], phi_orig = 0;
  float azi[N_AZI]; // pre-loaded shower azimuthal angles
  int
    size,
    outUnit,
    outBanks,
    rc;
  
 
  // print manual
  if (argc < 8 || argc > 10)
    {
      fprintf(stderr,"\nUsage: %s [1] [2] [3] [4]\n", argv[0]);
      fprintf(stderr,"[1]: <str> DAT????XX_gea.dat sower library file (last 2 digits XX are the energy channel:\n");
      fprintf(stderr,"     XX=00-25: energy from 10^18.0 to 10^20.5 eV\n");
      fprintf(stderr,"     XX=26-39: energy from 10^16.6 to 10^17.9 eV\n");
      fprintf(stderr,"     XX=80-85: energy from 10^16.0 to 10^16.5 eV\n");
      fprintf(stderr,"[2]: output DST file\n");
      fprintf(stderr,"[3]: <float> Number of events (Poisson mean) to generate\n");
      fprintf(stderr,"[4]: <int> Random seed number\n");
      fprintf(stderr,"[5]: <int> Data set number (for sdcalib_[data set number].bin file, \n");
      fprintf(stderr,"[6]: <str> Calibration file (/full/path/to/sdcalib_[data set number].bin file\n");
      fprintf(stderr,"     this number is the TA Date (\"Epoch\") calculated by TADay2Date function\n");
      fprintf(stderr,"[7]: <str> Binary file with atmospheric muon data (/full/path/to/atmos.bin)\n");
      fprintf(stderr,"[8]: <int> (OPT) Smear energies in 0.1 log10 bin according to E^-2? 1=YES,0-NO; default: %d\n",
	      smear_energies);
      fprintf(stderr,"[9]: <str> (OPT) Provide an ASCII file with azimuthal angles to use\n");
      fprintf(stderr,"                 1 column ASCII file, each line is azimuthal angle in degrees\n");
      fprintf(stderr,"                 CCW from East, along the shower propagation direction\n");
      fprintf(stderr,"                 By default, azimuthal angles are sampled randomly\n");
      fprintf(stderr,"\n");
      exit(1);
    }

  /* initialize counter flags */
  for ( i = 0; i < NDATE*144; i++ )
    for ( j = 0; j < NCOUNTERS; j++ )
      {
	/* initialize all counters as not working, then when the calibration information 
	   is loaded, working counters will have a zero flag for deadtime */
	deadtime[i][j] = 1;
	/* initialize bit flags to label SDs that are in the calibration but 
	   not working properly */
	bsd_bitf[i][j] = 0xFFFF;
      }
  
  /* Start DST file for writing events */
  rusdraw_.event_num = 0;
  size = 100;
  outUnit = 1;
  outBanks = newBankList(size);
  addBankList(outBanks, RUSDMC_BANKID);
  addBankList(outBanks, RUSDRAW_BANKID);
  addBankList(outBanks, BSDINFO_BANKID);
  if ((rc=dstOpenUnit(outUnit,argv[2],MODE_WRITE_DST))< 0)
    {
      fprintf(stderr,"Can't start the output dst file %s\n", argv[2]);
      return -1;
    }
  sscanf(argv[3], "%lf", &nEvent_to_throw_poisson_mean);
  sscanf(argv[4], "%d",  &randint);
  sscanf(argv[5], "%d",  &dataset);

  /* Calculate the date range. 24 days (SDATE) added because SD data starts at May 11, 2008 but the
   current implementation of Date2TADay from sdmc_tadate.c uses April 17, 2008 00:00 UT as
   zero point. */
  NDATE1 = SDATE + (dataset-1)*NDATE;
  NDATE2 = SDATE + dataset*NDATE;
  
  /* loading counter positions that are appropriate for the date range */
  for ( j = 0; j < NCOUNTERS; j++)
    for ( k = 0; k < 4; k++)
      sdcoor[j][k] = sdcoor0[j][k];
  for (i = 0; i < NALT; i++)
    if ( altcoor[i][4] < NDATE1 )
      for ( j = 0; j < NCOUNTERS; j++)
	if ( sdcoor[j][0] == altcoor[i][0] )
	  for ( k = 1; k < 4; k++)
	    sdcoor[j][k] = altcoor[i][k];

  /* load the calibration information that is appropriate for the data set number*/
  if (!(f=fopen(argv[6],"rb")))
    {
      fprintf(stderr, "Error: file %s that should correspond to epoch %d can not be opened for reading\n",
	      argv[6],dataset);
      exit(2);
    }
  while ( fread ( calbuf, sizeof(float), NCAL, f) == NCAL)
    {
      i=(int)calbuf[0]-NDATE1*144;
      j=(int)calbuf[1];

      if ( i < 0 || i >= 144*(NDATE2-NDATE1))
	{
	  fprintf(stderr,"warning: counter's monitoring cycle %d in %s is inconsistent with the epoch number %d range [%d,%d)!\n",
		  (int)calbuf[0],argv[6],dataset,NDATE1*144,NDATE2*144);
	  continue;
	}

      if (j < 0 || j >= NCOUNTERS )
	{
	  fprintf(stderr,"warning: counter internal index %d in %s is outside of allowed range [0,%d)!\n",
		  j,argv[6],NCOUNTERS);
	  continue;
	}
      /* calculate the probelem description bit flags for counters that are or are not getting simulated */
      bsd_bitf[i][j] = failed_Ben_Stokes_D_Ivanov_Live_Counter_Criteria(calbuf);
      
      /* if the counter has a non-zero bit flag and it is not working well enough to be simulated, then
	 the MC dead time flag is set. */
      if(bsd_bitf[i][j] != 0 && (is_icrr_cal_mc(bsd_bitf[i][j]) || is_ru_cal_mc(bsd_bitf[i][j])))
	deadtime[i][j] = 1;
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
	  deadtime[i][j] = 0;
	  /*	  sat[i][j][0] = calbuf[24];
		  sat[i][j][1] = calbuf[25]; */
	  sat[i][j][0] = FADC_NONLIN1;
	  sat[i][j][1] = FADC_NONLIN1;
	}
    }
  
  fclose(f);

  /* Read atmospheric muon simulation file */
  f=fopen(argv[7], "rb");
  if(!f)
    {
      fprintf(stderr,"error: file %s for atmospheric muon simulation not found!\n",argv[7]);
      exit(2);
    }
  i=0;
  while (fread(atmos[i],sizeof(float),2,f)==2 && i<NATMOS) i++;
  fclose(f);
  natmos=i;

  // Smear energies?
  if(argc > 8)
    sscanf(argv[8], "%d",  &smear_energies);
  if(abs(smear_energies) > 1)
    smear_energies = 1;
  
  if(argc > 9)
    {
      if(!(f=fopen(argv[9],"r")))
	{
	  fprintf(stderr,"error: file %s for shower azimuthal angles could not be read!\n",argv[9]);
	  exit(2);
	}
      char in_buf[1024];
      while(fgets(in_buf,1024,f))
	{
	  if(n_azi >= N_AZI)
	    {
	      fprintf(stderr,"error: %s has more than the maximum %d allowed pre-set azimuthal values\n", argv[9],N_AZI);
	      exit(2);
	    }
	  if(sscanf(in_buf,"%f",&azi[n_azi]))
	    n_azi++;
	}
      fclose(f);
    }

  srand48(randint);
  nEvent_global = (int)floor(poissond(nEvent_to_throw_poisson_mean)+0.5);
  fprintf(stdout, "Input DAT file: %s\n", argv[1]);
  fprintf(stdout, "Output DST file: %s\n", argv[2]);
  fprintf(stdout, "Number of Events Sampled Using Poisson Mean: %.9e\n", nEvent_to_throw_poisson_mean);
  fprintf(stdout, "Number of Events Thrown: %d\n", nEvent_global);
  fprintf(stdout, "Random Seed: %d\n", randint);
  fprintf(stdout, "Epoch number: %d\n", dataset);
  fprintf(stdout, "Calibration file: %s\n",argv[6]);
  fprintf(stdout, "Atmospheric muon file: %s\n",argv[7]);
  fprintf(stdout, "Smear energies in 0.1 log10 bins according to E^-2? %s\n", (smear_energies >= 1 ? "YES" : "NO"));
  fprintf(stdout, "Sampling shower azimuthal angles: ");
  if(n_azi)
    fprintf(stdout,"%d unique values from %s\n",n_azi,argv[10]);
  else
    fprintf(stdout,"randomly in 0 .. 360 degree range\n");
  fprintf(stdout, "\n");
  fflush(stdout);


  while ( nEvent_global > 0 )
    {
      if ( nEvent_global > NT )
	{
	  nEvent = NT;
	  nEvent_global -= NT;
	}
      else
	{
	  nEvent = nEvent_global;
	  nEvent_global = 0;
	}
      
      fprintf(stdout,"Throwing %d out of %d remaining ...\n", nEvent, nEvent_global+nEvent);
      fflush(stdout);
      
      for (i = 0; i < NT;  i++)
	for ( j = 0; j < NSECT; j++)
	  secttime[i][j] = 1000000;
      
      for (i = 0; i < NCOUNTERS; i++)
	{
	  col[i] = (int)sdcoor[i][0]/100;
	  row[i] = (int)sdcoor[i][0] - 100*col[i];
	  for ( j = 0; j < nEvent;  j++)
	    {
	      for (k = 0; k < TMAX; k++)
		sampletime[i][j][k] = -1000000;
	      for (k = 0; k < FADC_TOT/2; k++)
		{
		  trigtable[i][j][k] = 0;
		  dtrigtable[i][j][k] = 0;
		}
	      for (k = 0; k < FADC_TOT; k++)
		{
		  fadc[i][j][k][0] = 0.;
		  fadc[i][j][k][1] = 0.;
		}
	    }
	}
      for (i = 0; i < NX; i++)
	for ( j = 0; j < NY;  j++)
	  for (k = 0; k < nEvent; k++)
	    {
	      sdmap[i][j][k] = 0;
	      tmin[k] = 1000000;
	    }


      // must know event's oringinal azimuthal angle before adjusting the geometry of the event
      // if using predefined azimuthal angles from an ASCII file.
      if(n_azi)
	{
	  f=fopen(argv[1], "rb");
	  if(!f)
	    {
	      fprintf(stderr,"Error: file '%s' not found\n",argv[1]);
	      exit(2);
	    }
	  if(!fread ( eventbuf, sizeof(float), NWORD, f))
	    {
	      fprintf(stderr,"error: failed to read header from %s\n",argv[1]);
	      exit(2);
	    }
	  fclose(f);
	  phi_orig = eventbuf[11];
	}
      

      for (i = 0; i < nEvent; i++)
	{
	  randnum[0] = sqrtf((float)drand48()*DETDIST*DETDIST);
	  randnum[1] = 2.*PI*(float)drand48();
	  // if using predefined values of azimuthal angles, n_azi will be greater than zero
	  if(n_azi)
	    {
	      // take a random angle from the list of predefined values that have
	      // been stored in the ASCII file
	      // calculate the addition to phi_orig so that the overall azimuthal angle
	      // is the one taken from the ASCII file
	      randnum[2] = (PI/180.0)*azi[(int)floor(-0.5+drand48()*(double)n_azi)]-phi_orig;
	      if(randnum[2] < 0.0)
		randnum[2] += 2.0 * PI;
	    }
	  else
	    randnum[2] = 2.*PI*(float)drand48(); // do a random addition to phi_orig
	  phi[i]=randnum[2];
	  randnum[3] = (float)(NDATE2-NDATE1)*144.*(float)drand48();
	  showertime[i]=randnum[3];
	  eventdate[i]=(int)(showertime[i]/144.)+NDATE1;
	  distmin = 30.e5;
	  corecounter[i] = 0;
	  corexyz[i][0] = randnum[0]*cosf(randnum[1]);
	  corexyz[i][1] = randnum[0]*sinf(randnum[1]);
	  corexyz[i][2] = Z0-CLFZ-randnum[0]*randnum[0]/2./ERAD;
	  for ( j = 0; j < NCOUNTERS; j++ )
	    {
	      sdtrans[0] = (float)sdcoor[j][1]-randnum[0]*cosf(randnum[1]);
	      sdtrans[1] = (float)sdcoor[j][2]-randnum[0]*sinf(randnum[1]);
	      if ( hypotf(sdtrans[0],sdtrans[1]) < distmin)
		{
		  distmin = hypotf(sdtrans[0],sdtrans[1]);
		  corecounter[i] = (int)sdcoor[j][0];
		}
	      
	      sdtrans[2] = (float)sdcoor[j][3]-
		(sdtrans[0]*sdtrans[0]+sdtrans[1]*sdtrans[1])/2./ERAD;
	      sdrot[0] = (cosf(randnum[2])*sdtrans[0])+(sinf(randnum[2])*
							sdtrans[1]);
	      sdrot[1] = (-sinf(randnum[2])*sdtrans[0])+(cosf(randnum[2])*
							 sdtrans[1]);
	      if ( hypotf(sdrot[0], sdrot[1])/100. < (float)DISTMAX)
		{
		  m = (int)((sdrot[0]+(float)(100*DISTMAX))/600.);
		  n = (int)((sdrot[1]+(float)(100*DISTMAX))/600.);
		  sdmap[m][n][i] = j;
		  sdht[j][i] = sdtrans[2] - Z0;
		}
	    }  
	}
      for (i = 0; i < nEvent; i++)
	{
	  for ( j = 0; j < NCOUNTERS; j++)
	    {
	      kmax=(int)poissonf(ATMOSRATE);
	      if ( kmax > 0 )
		for (k = 0; k < 2; k++)
		  {
		    NPE2FADCE[k] = 6.08607+
		      log10(one_mev[(int)showertime[i]][j][k]/
			    mevpoisson[(int)showertime[i]][j][k]);
		    NPE2FADCE[k] = 1.93348 - (0.47694*NPE2FADCE[k]) + 
		      (0.0493108*pow(NPE2FADCE[k],2.)) - 
		      (0.00194568*pow(NPE2FADCE[k],3.));
		  }
	      for (k = 0; k < kmax; k++)
		{
		  if ( k < TMAX )
		    {
		      pnum=(int)(drand48()*(double)natmos);
		      gaussf(randnum);
		      mev_tmp = atmos[pnum][0];
		      poisson_tmp = poissonf(mev_tmp*
					     mevpoisson[(int)showertime[i]][j][0]);
		      if (poisson_tmp > 0.)
			poisson_tmp *= 1.+ NPE2FADCE[0]/sqrtf(poisson_tmp)*
			  randnum[0];
		      mev[j][i][k][0] = poisson_tmp/
			mevpoisson[(int)showertime[i]][j][0];
		      mev_tmp = atmos[pnum][1];
		      poisson_tmp = poissonf(mev_tmp*
					     mevpoisson[(int)showertime[i]][j][1]);
		      if (poisson_tmp > 0.)
			poisson_tmp *= 1.+ NPE2FADCE[1]/sqrtf(poisson_tmp)*
			  randnum[1];
		      mev[j][i][k][1] = poisson_tmp/
			mevpoisson[(int)showertime[i]][j][1];
		      
		      sampletime[j][i][k] = (int)((double)FADC_TOT*drand48())-FADC_START; 
		    }
		}
	    }
	}
      f=fopen(argv[1], "rb");
      if(!f)
	{
	  fprintf(stderr,"Error: file '%s' not found\n",argv[1]);
	  exit(2);
	}
      if(!fread ( eventbuf, sizeof(float), NWORD, f))
	{
	  fprintf(stderr,"error: failed to read header from %s\n",argv[1]);
	  exit(2);
	}
      rusdmc_.parttype = (int)eventbuf[2];
      energy = eventbuf[3]/1.e9;
      if ( energy < BRKPNT ) spctrindx = INDX01;
      for (i = 0; i < nEvent; i++)
	{
	  diff     = 0.0;
	  encor[i] = 1.0;
	  if(smear_energies)
	    {
	      while (drand48() > diff )
		{
		  encor[i] = (DRAWER - 1.)*drand48() + 1.;
		  diff = powf(encor[i], spctrindx);
		}
	      encor[i] = encor[i]/DRAWER_HF;
	    }
	}
      rusdmc_.height = eventbuf[6];
      rusdmc_.theta = eventbuf[10];
      phi_orig = eventbuf[11];
      while ( fread ( buf, sizeof(short), 6, f) == 6)
	{
	  m = (int)buf[0];
	  n = (int)buf[1];
	  for (i = 0; i < nEvent; i++)
	    {
	      if ( sdmap[m][n][i] > 0 )
		{
		  for (k = 0; k < 2; k++)
		    {
		      NPE2FADCE[k] = 6.08607+
			log10(one_mev[(int)showertime[i]][sdmap[m][n][i]][k]/
			      mevpoisson[(int)showertime[i]]
			      [sdmap[m][n][i]][k]);
		      NPE2FADCE[k] = 1.93348 - (0.47694*NPE2FADCE[k]) + 
			(0.0493108*pow(NPE2FADCE[k],2.)) - 
			(0.00194568*pow(NPE2FADCE[k],3.));
		    }
		  j = 0;
		  while (sampletime[sdmap[m][n][i]][i][j] > -1000000 && j < TMAX) j++;
		  if ( j < TMAX )
		    {
		      gaussf(randnum);
		      mev_tmp = (float)buf[2]/100.*VEM2MeV*encor[i];
		      poisson_tmp = poissonf(mev_tmp*
					     mevpoisson[(int)showertime[i]]
					     [sdmap[m][n][i]][0]);
		      if (poisson_tmp > 0.)
			poisson_tmp *= 1.+ NPE2FADCE[0]/sqrtf(poisson_tmp)*
			  randnum[0];
		      mev[sdmap[m][n][i]][i][j][0] = poisson_tmp/
			mevpoisson[(int)showertime[i]][sdmap[m][n][i]][0];
		      mev_tmp = (float)buf[3]/100.*VEM2MeV*encor[i];
		      poisson_tmp = poissonf(mev_tmp*
					     mevpoisson[(int)showertime[i]]
					     [sdmap[m][n][i]][1]);
		      if (poisson_tmp > 0.)
			poisson_tmp *= 1.+ NPE2FADCE[1]/sqrtf(poisson_tmp)*
			  randnum[1];
		      mev[sdmap[m][n][i]][i][j][1] = poisson_tmp/
			mevpoisson[(int)showertime[i]][sdmap[m][n][i]][1];
		      tmptime = (int)buf[4];
		      pztmp=(float)buf[5]/((float)buf[2]+(float)buf[3])*2.;
		      if (pztmp <= 0.) pztmp = cos(rusdmc_.theta);
		      if ( buf[4] > 32768 ) tmptime -= 65537;
		      
		      sampletime[sdmap[m][n][i]][i][j] = (tmptime*DT - 
						    (int)(sdht[sdmap[m][n][i]][i]*
							  pztmp/CSPEED2))/DT;
		      if ( sampletime[sdmap[m][n][i]][i][j] < tmin[i] )
			tmin[i] = sampletime[sdmap[m][n][i]][i][j];
		    }
		}
	    }
	}
     fclose(f);
     for ( i = 0; i < nEvent; i++)
	{
	  for ( j = 0; j < NCOUNTERS; j++)
	    {
	      if (deadtime[(int)showertime[i]][j] == 0 && 
		  sampletime[j][i][0] > -1000000 )
		{
		  for ( m = 0; m < FADC_TOT; m++)
		    {
		      gaussf(randnum);
		      fadc[j][i][m][0] += fadc_ped[(int)showertime[i]][j][0] + 
			fadc_noise[(int)showertime[i]][j][0]*randnum[0];
		      fadc[j][i][m][1] += fadc_ped[(int)showertime[i]][j][1] + 
			fadc_noise[(int)showertime[i]][j][1]*randnum[1];
		    }
		  k=0;
		  while ( sampletime[j][i][k] > -1000000 && k < TMAX)
		    {
		      for ( m=0; m < RESP_WIDTH; m++)
			{
			  if ( m+sampletime[j][i][k]+FADC_START > 0 &&
			       m+sampletime[j][i][k]+FADC_START < FADC_TOT)
			    {
			      fadc[j][i][m+sampletime[j][i][k]+FADC_START][0] += 
				mev[j][i][k][0]*one_mip[m]*
				one_mev[(int)showertime[i]][j][0];
			      fadc[j][i][m+sampletime[j][i][k]+FADC_START][1] += 
				mev[j][i][k][1]*one_mip[m]*
				one_mev[(int)showertime[i]][j][1];
			    }
			}
		      k++;
		    }
		  //if (k > 100) printf("%d %d\n", sdcoor[j][0], k);
		  //fflush(stdout);
		  for ( m = 0; m < FADC_TOT; m++)
		    {
		      if (fadc[j][i][m][0] > sat[(int)showertime[i]][j][0])
			{			
			  //printf("%g %g ", fadc[j][i][m][0], sat[(int)showertime[i]][j][0]); 
			  fadc[j][i][m][0] = sat[(int)showertime[i]][j][0] + 
			  ((fadc[j][i][m][0]-
					sat[(int)showertime[i]][j][0])*
				expf(-(ATTEN*
				       (fadc[j][i][m][0]-
					       sat[(int)showertime[i]][j][0])/
				       FADC_NONLIN2)));
			  //printf("%g\n", fadc[j][i][m][0]);			
			}
		      if (fadc[j][i][m][1] > sat[(int)showertime[i]][j][1])
			fadc[j][i][m][1] = sat[(int)showertime[i]][j][1] + 
			  ((fadc[j][i][m][1]-
					sat[(int)showertime[i]][j][1])*
				expf(-(ATTEN*
				       (fadc[j][i][m][1]-
					       sat[(int)showertime[i]][j][1])/
				       FADC_NONLIN2)));
		      if (fadc[j][i][m][0] > FADC_MAX) fadc[j][i][m][0] = 
							 FADC_MAX;
		      if (fadc[j][i][m][1] > FADC_MAX) fadc[j][i][m][1] = 
							 FADC_MAX;
		    }
		  trigsum[0] = 0.;
		  trigsum[1] = 0.;
		  wfnum=0;
		  wfnum2=0;
		  for( m = 22; m < 30; m++)
		    for ( n = 0; n < 2; n++)
		      trigsum[n] += rint(fadc[j][i][m][n]);
		  while ( m < FADC_TOT-136 )
		    {
		      //if ( wfnum >= FADC_TOT/2 && wfnum2 >= FADC_TOT/2) printf ("%d %d\n", wfnum, wfnum2);
		      if ( trigsum[0] > 
			   (float)pchped[(int)showertime[i]][j][0]+
			   0.3*ONE_MIP_TH &&
			   trigsum[1] > 
			   (float)pchped[(int)showertime[i]][j][1]+
			   0.3*ONE_MIP_TH )
			
			{
			  trigtable[j][i][wfnum] = m;
			  if ( wfnum > 0 )
			    if ( trigtable[j][i][wfnum] < 
				 trigtable[j][i][wfnum-1]+128)
			      trigtable[j][i][wfnum] =
				trigtable[j][i][wfnum-1]+128;
			  trigsum2[0] = 0.;
			  trigsum2[1] = 0.;
			  /*			  printf("%d\t%d\t%d\t%d\n", sdcoor[j][0], wfnum, m,
						  trigtable[j][i][wfnum]); */
			  for ( m = trigtable[j][i][wfnum] - 30; 
				m < trigtable[j][i][wfnum] + 98; m++ )
			    for ( n = 0; n < 2; n++)
			      trigsum2[n] += rint(fadc[j][i][m][n]);
			  if ( trigsum2[0] > 
			       (float)pchped[(int)showertime[i]][j][0]*16.+
			       3.*ONE_MIP_TH &&
			       trigsum2[1] > 
			       (float)pchped[(int)showertime[i]][j][1]*16.+
			       3.*ONE_MIP_TH )
			    {
			      dtrigtable[j][i][wfnum2] = trigtable[j][i][wfnum];
			      /*			      printf("\t%d\t%d\t%d\n",wfnum2, m,
							      dtrigtable[j][i][wfnum2]); */ 
			      wfnum2++;
			    }
			  trigsum[0] = 0.;
			  trigsum[1] = 0.;
			  for( m = trigtable[j][i][wfnum] + 97; 
			       m < trigtable[j][i][wfnum] + 105; m++)
			    for ( n = 0; n < 2; n++)
			      trigsum[n] += rint(fadc[j][i][m][n]);
			  wfnum++;
			}
		      for ( n = 0; n < 2; n++)
			{
			  trigsum[n] -= rint(fadc[j][i][m-8][n]);
			  trigsum[n] += rint(fadc[j][i][m][n]);
			}
		      m++;
		    }
		  /*		  if ( wfnum > 0 || wfnum2 > 0 )
				  printf("\t\t%d\t%d\t%d\n", sdcoor[j][0], wfnum, wfnum2); */
		}
	    }
 	  for ( j = 0; j < NCOUNTERS; j++)
	    {
	      p = 0;
	      while ( dtrigtable[j][i][p] > 0 && p <  FADC_TOT/2)
		{
		  for ( k = 0; k < NCOUNTERS; k++)
		    {
		      q = 0;
		      while ( dtrigtable[k][i][q] > 0 && q <  FADC_TOT/2)
			{
			  if ( j != k && 
			       ((abs(row[j]-row[k]) == 1 && 
				 col[j] == col[k]) ||
				(abs(col[j]-col[k]) == 1 &&
				 row[j] == row[k])) &&
			       (sector[(int)showertime[i]][j] == sector[(int)showertime[i]][k] || eventdate[i] > 207) &&
			       abs(dtrigtable[j][i][p]/50 - 
				   dtrigtable[k][i][q]/50) < 8)
			    {
			      for (m = 0; m < NCOUNTERS; m++)
				{
				  r = 0;
				  while ( dtrigtable[m][i][r] > 0 && r <  FADC_TOT/2)
				    {
				      if (((abs(row[j]-row[m]) == 1 && 
					    col[j] == col[m]) ||
					   (abs(col[j]-col[m]) == 1 &&
					    row[j] == row[m])) &&
					  m != k && m != j &&
					  (sector[(int)showertime[i]][j] == sector[(int)showertime[i]][m] || eventdate[i] > 207) &&
					  abs(dtrigtable[j][i][p]/50 - 
					      dtrigtable[m][i][r]/50) < 8 &&
					  abs(dtrigtable[k][i][q]/50 - 
					      dtrigtable[m][i][r]/50) < 8)
					{
					  dtrigtime_tmp = (dtrigtable[j][i][p]
							   + dtrigtable[k][i][q]
							   + dtrigtable[m][i][r]
							   )/3.;
					  if (dtrigtime_tmp < 
					      secttime[i][sector[(int)showertime[i]][j]])
					    secttime[i][sector[(int)showertime[i]][j]] = 
					      dtrigtime_tmp;
					}
				      r++;
				    }
				}
			    }
			  q++;
			}
		    }
		  p++;
		}
	      //	      if (p > 0 ) printf ("%d %d\n", sdcoor[j][0], p);
	      //	      fflush(stdout);
	    }
	  if ( eventdate[i] > 207 )
	    {
	      for ( j = 0; j < NSECT; j++ )
		for ( k = 0; k < NSECT; k++)
		  if ( secttime[i][j] > secttime[i][k] )
		    secttime[i][j] = secttime[i][k];
	    }
	  
	}
      rusdraw_.nofwf = 0;
      for ( i = 0; i < nEvent; i++)
	{
	    
	  /*	  printf ("%g ", showertime[i]/144.);
	  for ( j = 0; j < NCOUNTERS; j++ )
	    printf("%d ", deadtime[(int)showertime[i]][j]);
	    printf("\n\n"); */
	  nofwf[i] = 0;
	  for ( j = 0; j < NCOUNTERS; j++)
	    {
	      q = 0;
	      while ( trigtable[j][i][q] > 0 &&  q < FADC_TOT/2)
		{
		  if ( abs(secttime[i][sector[(int)showertime[i]][j]]/50 - trigtable[j][i][q]/50) < 
		       32)
		    {
		      nofwf[i]++;
		    }
		  q++;
		}
	    }
	}
      for ( i = 0; i < nEvent; i++)
	{
	  rusdraw_.nofwf = nofwf[i];
	  put2rusdraw();	   
	  rusdraw_.event_num = (int)lrand48();
	  TADay2Date(showertime[i]/144.+(float)NDATE1, &yymmdd, &hhmmss);
	  rusdraw_.yymmdd = yymmdd;
	  rusdraw_.hhmmss = hhmmss;  
	  rusdraw_.monyymmdd = yymmdd;
	  rusdraw_.monhhmmss = hhmmss;  
	  rusdmc_.event_num = rusdraw_.event_num;
	  time_offset = (int)(drand48()*5.e7);
	  //time_offset = 24;
	  rusdmc_.tc = time_offset + FADC_START - TOFFSET;
	  rusdraw_.usec = (int) (rusdmc_.tc/50.); 
	  if ( secttime[i][0] < 1000000 || 
	       secttime[i][1] < 1000000 || 
	       secttime[i][2] < 1000000 )
	    {
	      if ( secttime[i][0] < 1000000 )
		{ 
		  rusdraw_.site = RUSDRAW_BR;
		  rusdraw_.run_id[0] = 111;
		  rusdraw_.trig_id[0] = 1212;
		  if ( secttime[i][1] < 1000000 )
		    {
		      rusdraw_.site = RUSDRAW_BRLR;
		      rusdraw_.run_id[1] = 111;
		      rusdraw_.trig_id[1] = 1212;
		      if ( secttime[i][2] < 1000000 )
			{
			  rusdraw_.site = RUSDRAW_BRLRSK;
			  rusdraw_.run_id[2] = 111;
			  rusdraw_.trig_id[2] = 1212;
			}
		    }
		  else if (secttime[i][2] < 1000000 )
		    {
		      rusdraw_.site = RUSDRAW_BRSK;
		      rusdraw_.run_id[2] = 111;
		      rusdraw_.trig_id[2] = 1212;
		    }
		}
	      else if ( secttime[i][1] < 1000000 ) 
		{
		  rusdraw_.site = RUSDRAW_LR;
		  rusdraw_.run_id[1] = 111;
		  rusdraw_.trig_id[1] = 1212;
		  if ( secttime[i][2] < 1000000 )
		    {
		      rusdraw_.site = RUSDRAW_LRSK;
		      rusdraw_.run_id[2] = 111;
		      rusdraw_.trig_id[2] = 1212;
		    }
		}
	      else if ( secttime[i][2] < 1000000 )
		{
		  rusdraw_.site = RUSDRAW_SK;
		  rusdraw_.run_id[2] = 111;
		  rusdraw_.trig_id[2] = 1212;
		}
	      k = 0;
	      for ( j = 0; j < NCOUNTERS; j++)
		{
		  q = 0;
		  qq = 0;
		  while ( trigtable[j][i][q] > 0 && q < FADC_TOT/2 )
		    {
		      if ( abs(secttime[i][sector[(int)showertime[i]][j]]/50 - trigtable[j][i][q]/50) < 
			   32)
			{
			  rusdraw_.xxyy[k] = sdcoor[j][0];
			  rusdraw_.clkcnt[k] = trigtable[j][i][q]-30 + time_offset;
			  for ( m = 0; m < FADC_LENGTH; m++)
			    {
			      for ( n = 0; n < 2; n++ )
				{
				  rusdraw_.fadc[k][1-n][m] = 
				    (int)rintf(fadc[j][i]
					       [m+trigtable[j][i][q]-30][n]);
				  rusdraw_.fadcti[k][1-n] += 
				    rusdraw_.fadc[k][1-n][m];
				}
			    }
			  rusdraw_.wf_id[k] = qq;
			  rusdraw_.fadcav[k][0] = (int)rintf
			    (8.*fadc_ped[(int)showertime[i]][j][1]);
			  rusdraw_.fadcav[k][1] = (int)rintf
			    (8.*fadc_ped[(int)showertime[i]][j][0]);
			  rusdraw_.mip[k][0] = mip[(int)showertime[i]][j][1];
			  rusdraw_.mip[k][1] = mip[(int)showertime[i]][j][0];
			  rusdraw_.pchped[k][0] = 
			    (int)pchped[(int)showertime[i]][j][1];
			  rusdraw_.pchped[k][1] = 
			    (int)pchped[(int)showertime[i]][j][0];
			  rusdraw_.lhpchped[k][0] = 
			    (int)lhpchped[(int)showertime[i]][j][1];
			  rusdraw_.lhpchped[k][1] =
			    (int)lhpchped[(int)showertime[i]][j][0]; 
			  rusdraw_.rhpchped[k][0] = 
			    (int)rhpchped[(int)showertime[i]][j][1];
			  rusdraw_.rhpchped[k][1] =
			    (int)rhpchped[(int)showertime[i]][j][0]; 
			  rusdraw_.mftndof[k][0] = 
			    (int)mftndof[(int)showertime[i]][j][1];
			  rusdraw_.mftndof[k][1] =
			    (int)mftndof[(int)showertime[i]][j][0]; 
			  rusdraw_.mftchi2[k][0] = 
			    mftchi2[(int)showertime[i]][j][1];
			  rusdraw_.mftchi2[k][1] =
			    mftchi2[(int)showertime[i]][j][0]; 
			  rusdraw_.pchmip[k][0] = (int)rintf
			    (12.*fadc_ped[(int)showertime[i]][j][1]+
			     mip[(int)showertime[i]][j][1]);
			  rusdraw_.pchmip[k][1] = (int)rintf
			    (12.*fadc_ped[(int)showertime[i]][j][0]+
			     mip[(int)showertime[i]][j][0]);
			  rusdraw_.lhpchmip[k][0] = (int)rintf
			    (12.*fadc_ped[(int)showertime[i]][j][1]+
			     0.5*mip[(int)showertime[i]][j][1]);
			  rusdraw_.lhpchmip[k][1] = (int)rintf
			    (12.*fadc_ped[(int)showertime[i]][j][0]+
			     0.5*mip[(int)showertime[i]][j][0]);
			  rusdraw_.rhpchmip[k][0] = (int)rintf
			    (12.*fadc_ped[(int)showertime[i]][j][1]+
			     1.5*mip[(int)showertime[i]][j][1]);
			  rusdraw_.rhpchmip[k][1] = (int)rintf
			    (12.*fadc_ped[(int)showertime[i]][j][0]+
			     1.5*mip[(int)showertime[i]][j][0]);
			  k++;
			  qq++;
			}
		      q++;
		    }
		}
	    }

	  /* Filling information on counters that were not working properly. 
	     Note. In the TA SD Monte Carlo they mostly do not participate in the event trigger 
	     and event readout, but they are useful in the reconstruction later */
	  bsdinfo_.yymmdd = yymmdd;
	  bsdinfo_.hhmmss = hhmmss;
	  bsdinfo_.usec = (int) (rusdmc_.tc/50.);
	  bsdinfo_.nbsds = 0;
	  bsdinfo_.nsdsout = 0;
	  for ( j = 0; j < NCOUNTERS; j++)
	    {
	      /* cases where a poorly working counter can show up in the MC event */ 
	      if(bsd_bitf[(int)showertime[i]][j] != 0 && deadtime[(int)showertime[i]][j] == 0)
		{
		  /* check if a poorly working counter showed up in the event read out 
		     (because it did not fail MC simulation wellness requirements but it failed
		     the reconstruction wellness requirement) */
		  for (k=0; k < rusdraw_.nofwf; k++)
		    if(rusdraw_.xxyy[k] == sdcoor[j][0])
		      break;
		  if(k != rusdraw_.nofwf)
		    {
		      bsdinfo_.xxyy[bsdinfo_.nbsds] = sdcoor[j][0];
		      bsdinfo_.bitf[bsdinfo_.nbsds] = bsd_bitf[(int)showertime[i]][j];
		      bsdinfo_.nbsds ++;
		    }
		}
	      /* cases where a poorly working counter does not show up in the MC event trigger and readout */
	      if(deadtime[(int)showertime[i]][j])
		{
		  bsdinfo_.xxyyout[bsdinfo_.nsdsout] = sdcoor[j][0];
		  bsdinfo_.bitfout[bsdinfo_.nsdsout] = bsd_bitf[(int)showertime[i]][j];
		  bsdinfo_.nsdsout ++;
		}
	    }
	  
	  rusdmc_.corecounter = corecounter[i];
	  rusdmc_.corexyz[0] = corexyz[i][0];
	  rusdmc_.corexyz[1] = corexyz[i][1];
	  rusdmc_.corexyz[2] = corexyz[i][2];
	  rusdmc_.phi = phi[i]+phi_orig;
	  if (rusdmc_.phi >= 2.*PI) rusdmc_.phi -= 2.*PI; 
	  rusdmc_.energy = energy*encor[i];
	  /*	  fprintf(stdout, "Event %d: Core = %d\nBR trigtime = %d\nLR trigtime = %d\nSK trigtime = %d\n\n", rusdraw_.event_num,
		  rusdmc_.corecounter, secttime[i][0], secttime[i][1],
		  secttime[i][2]); */
	  if ( secttime[i][0] < 1000000 || 
	       secttime[i][1] < 1000000 || 
	       secttime[i][2] < 1000000 )
	    {
	      if((rc = eventWrite(outUnit, outBanks, TRUE))<0)
		{
		  fprintf(stderr,"Can't  write to dst file\n");
		  exit(EXIT_FAILURE);
		}
	    }
	  else
	    {
	      if((rc = eventWrite(outUnit, outBanks, TRUE))<0)
		{
		  fprintf(stderr,"Can't  write to dst file\n");
		  exit(EXIT_FAILURE);
		}
	    }
	}
    }
  fprintf(stdout,"\nDone\n");
  dstCloseUnit(outUnit);  
  exit(EXIT_SUCCESS);
}

void gaussf(float pair[])
{
  double r, phi;
  r = sqrt(-2.0*log(drand48()));
  phi = 2*PI*drand48();
  pair[0] = (float)(r*cos(phi));
  pair[1] = (float)(r*sin(phi));
  return;
}

void gaussd(double pair[])
{
  double r, phi;
  r = sqrt(-2.0*log(drand48()));
  phi = 2*PI*drand48();
  pair[0] = (double)(r*cos(phi));
  pair[1] = (double)(r*sin(phi));
  return;
}

float poissonf(float lambda)
{
  int k=0;
  double L=exp(-(double) lambda), p=1;
  float pair[2];
  if (lambda > 20.)
    {
      gaussf(pair);
      lambda += sqrtf(lambda)*pair[0];
      return rintf(lambda);
    }
  else
    {
      while ( p >= L )
	{
	  p *= drand48();
	  k++;
	}
      return (float)(k-1);
    }
}

double poissond(double lambda)
{
  int k=0;
  double L=exp(-(double) lambda), p=1;
  double pair[2];
  if (lambda > 20.)
    {
      gaussd(pair);
      lambda += sqrt(lambda)*pair[0];
      return rint(lambda);
    }
  else
    {
      while ( p >= L )
	{
	  p *= drand48();
	  k++;
	}
      return (double)(k-1);
    }
}

void put2rusdraw()
{
  integer4 iwf,j,k;
  // Indicate that this is a MC event
  rusdraw_.event_code = 0;
  // Let it be a BR event
  rusdraw_.site = RUSDRAW_BR;
  // Run ID corresponding to BR (the 322 part of BR000322.Y2008, for example). Make one up for MC.
  rusdraw_.run_id[0] = -1;
  rusdraw_.run_id[1] = -1;   // This is a BR only event, so the other numbers are irrelevant.
  rusdraw_.run_id[2] = -1;
  // Trig. id corresponding to BR.  Useful for finding events in real data, but not here. Make one up.
  rusdraw_.trig_id[0] = -1;
  rusdraw_.trig_id[1] = -1;
  rusdraw_.trig_id[2] = -1;
  rusdraw_.errcode = 0; // Don't want our event to get cut out.
  // make these up
  //  rusdraw_.yymmdd = 80916; // Event date year = 08, month = 09, day = 16
  //  rusdraw_.hhmmss = 1354;  // Event time, hour=00, minute=13, second = 54
  // Reported by DAQ as time of the 1st signal in the triple that caused the triggger.
  // From now on, everyhting is relative to hhmmss.  Not useful in the event reconstruction.
  //  rusdraw_.usec = 111111;
  // Date and time of the monitoring cycle for this event. In an ideal situation, a 10min
  // monitoring cycle that was recorded right after the event trigger time should be used, i.e.
  // want the calib. information collected in the time window when the event is collected.
  // For MC, just make up an ideal time, for now.
  //  rusdraw_.monyymmdd = 80916;
  //  rusdraw_.monhhmmss = 2000; // hour = 00, minute = 20, second=00
  // Cumulative number of 128 channel FADC windows. It doesn't equal to the number of counters
  // in the event b/c some counters have more than 1 waveform in them (10 waveforms is max.).
  // It's not very often that we get more than 1 waveform per SD, so rusdraw is 
  // a waveform-based list of variables, not counter-based (makes the analysis simpler).
  // CHANGE this.
  // Fill in the waveforms, their times, and some calib. information.
  for(iwf=0; iwf<rusdraw_.nofwf; iwf++)
    {
      // assume good wireless connection
      rusdraw_.nretry[iwf] = 0;
      // For multiple waveform SDs, this just counts the waveforms (and is reported by DAQ)
      // wf_id = 0 for the 1st one, wf_id = 1 for the 2nd one, and so on. For most SDs, there is only one
      // waveform, so this is set to zero. CHANGE this as needed.
      // Some kind of a code that tells if the waveform (in same SD) was recorded
      // shortly after the previous waveform (=2) or some time after it (usually a couple of uS) (=4)
      // I don't use it, so set it to the typical value.
      rusdraw_.trig_code[iwf] = 2;
      // Counter positions. CHANGE these.
      rusdraw_.xxyy[iwf] = 1809; 
      // CHANGE THIS, so that the time of the i'th FADC bin (with respect to current HHMMSS)
      // is time = ((double)clkcnt[iwf]/(double)mclkcnt[iwf]) + 20e-9 * i, where i runs from 0 to 127.
      rusdraw_.clkcnt[iwf] = (integer4)30e6;
      // Ideal value for max. clock count
      rusdraw_.mclkcnt[iwf] = (integer4)50e6;
      // k loops over lower(k=0) and upper(k=1) layers.
      for(k=0;k<2;k++)
	{
	  // CHANGE THIS, so that this variable equals to the integral of the 128 channel FADC histogram.
	  rusdraw_.fadcti[iwf][k] = 0;
	  // This is an ideal value. This variable is not the average of the actual FADC trace.
	  // It is the average noise in 8 fadc channels.
	  // CHANGE this. Arrange your FADC bins so that the first signal in the SD starts after about 20 FADC
	  // channels of pedestals (that's how the real data is).
	  for(j=0;j<128;j++)
	    {
	      if((rusdraw_.wf_id[iwf]==0) && (j<20))
		{
		  rusdraw_.fadc[iwf][k][j] = 6;
		}
	      else
		{
		  rusdraw_.fadc[iwf][k][j] = 0.0;
		}
	    }

	  /*
	    CALIBRATION INFORMATION. SOME OF THESE ARE IMPORTANT, OR ELSE THE COUNER WILL GET CUT OUT 
	    IN THE ANALYSIS.
	    Because pedestals  (and 1MIP values) vary a lot with time, I'm going to give only their 'usual'
	    values here.
	  */
	  
	  //	  rusdraw_.pchmip[iwf][k] = 119;     /* peak channel of 1MIP histograms */
	  //	  rusdraw_.pchped[iwf][k] = 48;     /* peak channel of pedestal histograms */
	  //	  rusdraw_.lhpchmip[iwf][k] = 96;   /* left half-peak channel for 1mip histogram */
	  //	  rusdraw_.lhpchped[iwf][k] = 45;   /* left half-peak channel of pedestal histogram */
	  //	  rusdraw_.rhpchmip[iwf][k] = 141;   /* right half-peak channel for 1mip histogram */
	  //	  rusdraw_.rhpchped[iwf][k] = 52;   /* right half-peak channel of pedestal histograms */
	  

	  // Pedestal histograms are very sharp and there is not point of fitting them into anything. 
	  // We calculate the pedestals as
	  // (ped per fadc channel) = ((double)pchped) / 8.0 = 6.25 
	  // fwhm = rhpchmip-lhpchped, (ped. err per fadc channel) = ((double)fwhm)/8.0/2.33
	  // typical value for fwhm of ped histogram is 6, which gives a typical
	  // ped. err per fadc channel = 6/8/2.33 = 0.32
	  // (the 2.33 number is to get the "gauss sigma" from the FWHM)

	  
	  
	  /* Results from fitting 1MIP histograms */
	  /* number of degrees of freedom in 1MIP fit, using generic values, such that the counter
	     is not cut out */
	  rusdraw_.mftndof[iwf][k] = 25; 
	  /* 1MIP value.  Here, we calculate it as
	     pchmip - 12/8 * pchped.  In real data analysis, I fit the 1MIP histogram to find a better
	     peak value and get chi2, ndof which describe the goodness of the counter.*/
	  // chi2 of the 1MIP fit, (ideal situation ...)
	  rusdraw_.mftchi2[iwf][k] = 25.0; 
	  //    Don't worry about these variables, they are not used in the analysis.
	  memset(rusdraw_.mftp[iwf][k],  (char)0, 4*sizeof(integer4));
	  memset(rusdraw_.mftpe[iwf][k], (char)0, 4*sizeof(integer4));
	}
    }
}
