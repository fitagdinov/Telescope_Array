#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "univ_dst.h"
#include "event.h"
#include "filestack.h"
#include "time.h"

// AVAILABLE SPECTRUM OPTIONS
#define HIRES_PRL_2008_SPECTRUM_ID 1
#define TASD_ICRC2015_SPECTRUM_ID  2
#define E_MINUS_3_SPECTRUM_ID      3


// Spectral index with which the SD MC levent ibrary files has been thrown
static double INDX0 = 2.0;

// Energy scale correction factor to use before comparing to HiRes spectrum
// (energies changed by 1/ENCOR)
static double ENCOR = 1.27;

// Default minimum energy in the set [EeV]
static double E0 = 0.316227766017;


// Processes each individual DST file
integer4 cull_file(integer1 *);

// write event into DST file and open DST file as needed
void write_out_event_to_dst();

// E^-3 spectrum everywhere, energy is in EeV units
// Result is normalized to 1 at energy_norm (in EeV)
real8 e_minus_3_spctr(real8 energy, real8 energy_norm);

// HiRes PRL 2008 spectrum, energy is in EeV units
// Result is normalized to 1 at energy_norm (in EeV)
real8 hires_spctr_prl2008(real8 energy, real8 energy_norm);

// TA SD 2013 spectrum (7 years of data), energy is in EeV units
// Result is normalized to 1 at energy_norm (in EeV)
real8 tasd_spctr_icrc2015(real8 energy, real8 energy_norm);

// Pointer to the selected energy spectrum function
real8 (*used_spctr)(real8 energy,real8 energy_norm) = 0;

integer1 *outfile = 0;

integer4 dst_unit_out_open = 0;
integer4 dst_unit_in = 1, dst_unit_out = 2;
integer4 banklist_want, banklist_got;
real8 frac=1.0;
integer4 spctr_id = E_MINUS_3_SPECTRUM_ID;

int main(int argc, char **argv) {

  integer4 arg;
  integer4 randnum = (integer4)(time(NULL) % 4294967296);
  integer1 *filename = 0;
  if (argc == 1 ) 
    {
      fprintf(stderr,"Program to convert an SD shower library element (0.1 log10E/eV bin) into HiRes spectrum\n");
      fprintf(stderr,"Program assumes that events inside of the bin follow E^-2 power law\n");
      fprintf(stderr,"Aslo taking into account energy scale correction factor of 1.0/%.3f\n",ENCOR);
      fprintf(stderr,"\nUsage: %s [-o output] [-frac Fraction of events to retain (on statistical basis)]\n\n", 
	      argv[0]);
      fprintf(stderr,"     -s  <int>    : spectrum to make: \n");
      fprintf(stderr,"                         %d: HiRes PRL 2008 spectrum\n",HIRES_PRL_2008_SPECTRUM_ID);
      fprintf(stderr,"                         %d: TA SD ICRC-2015 spectrum\n",TASD_ICRC2015_SPECTRUM_ID);
      fprintf(stderr,"                         %d: E^-3 spectrum (DEFAULT)\n",E_MINUS_3_SPECTRUM_ID);
      fprintf(stderr,"     -o <string> : output dst filename\n");
      fprintf(stderr,"     -i <int>    : random seed number, default is system time\n");
      fprintf(stderr,"     -e <float>  : minimum energy [EeV](before energy scale correction), default %.4f EeV\n",
	      E0);
      fprintf(stderr,"     -f <float>  : fraction of events in the file to use for sampling, default %f\n",frac);
      fprintf(stderr,"     -g <float>  : starting index of the MC event library, default %f\n",INDX0);
      fprintf(stderr,"     -c <float>  : energy scale correction (reduction) factor, default %f\n\n", ENCOR);
      exit(1);
    }

  /* Otherwise, scan the arguments first */
  
  for (arg = 1; arg < argc; ++arg) {
    if (argv[arg][0] != '-') 
      pushFile( argv[arg] );

    else {
      switch (argv[arg][1]) {
      case 'o': 
	arg++; outfile = argv[arg]; 
	break;
      case 's':
	arg++; sscanf(argv[arg], "%d", &spctr_id);
	break;
      case 'f':
	arg++; sscanf(argv[arg], "%lf", &frac); 
	break;
      case 'r':
	arg++; sscanf(argv[arg], "%i", &randnum); 
	break;
      case 'e':
	arg++; sscanf(argv[arg], "%lf", &E0); 
	break;
      case 'g':
	arg++; sscanf(argv[arg], "%lf", &INDX0); 
	break;
      case 'c':
	arg++; sscanf(argv[arg], "%lf", &ENCOR); 
	break;
      default: 
	fprintf(stderr,"Warning: unknown option: %s\n",argv[arg]); 
	break;
      }
    }
  }
  
  if (!outfile ) 
    {
      fprintf(stderr, "\n  Error: No output file given\n\n");
      exit(1);
    }
  // output file is opened only if there are events to write out
  dst_unit_out_open = 0;
  
  fprintf(stdout,"SAMPLING: ");
  switch(spctr_id)
    {
    case HIRES_PRL_2008_SPECTRUM_ID:
      fprintf(stdout,"HIRES PRL 2008 SPECTRUM\n");
      used_spctr = hires_spctr_prl2008;
      break;
    case TASD_ICRC2015_SPECTRUM_ID:
      fprintf(stdout,"TASD ICRC 2015 SPECTRUM\n");
      used_spctr = tasd_spctr_icrc2015;
      break;
    case E_MINUS_3_SPECTRUM_ID:
      fprintf(stdout,"E^-3 SPECTRUM\n");
      used_spctr = e_minus_3_spctr;
      break;
    default:
      fprintf(stdout,"\n");
      fprintf(stderr,"Error: wrong spectrum id (%d): must be one of (%d %d %d)\n",
	      spctr_id,HIRES_PRL_2008_SPECTRUM_ID,TASD_ICRC2015_SPECTRUM_ID,E_MINUS_3_SPECTRUM_ID);
      exit(1);
      break;
    }
  fflush(stdout);


  srand48(randnum);
  banklist_want = newBankList(512);
  banklist_got  = newBankList(512);
  eventAllBanks(banklist_want);
  
  /* Now process input file(s) */
  
  while ((filename = pullFile()))
    cull_file( filename );
  

  /* close DST unit if it was opened*/
  if(dst_unit_out_open)
    dst_close_unit_(&dst_unit_out);
  return SUCCESS;
}


integer4 cull_file(integer1 *dst_filename) 
{
  integer4 rcode=0, ssf=0, dst_mode = MODE_READ_DST;
  if ((rcode=dst_open_unit_(&dst_unit_in, dst_filename, &dst_mode))) 
    
    {
      fprintf(stderr,"\n  Warning: Unable to open/read file: %s\n", 
	      dst_filename);
      return(-1);
    }
  else 
    fprintf(stdout, "  Culling: %s   \n", dst_filename);
  
  // Read events out of the DST file and write out successfully sampled events
  while ((rcode=eventRead(dst_unit_in,banklist_want,banklist_got,&ssf)) >= 0)
    {
      
      // First (statistically) reduce the number of events in the sample using the fraction
      // that was passed on the command line, this might be necessary to bring the
      // entire shower library elements into a common power law, before doing anything else
      if(frac < drand48())
	continue;
      
      // Make sure that the event meets the minimum energy requirement
      if (rusdmc_.energy < E0)
	continue;
      
      // Apply the spectrum sampling, which is normalized in such a way
      // that it is unity at the corrected minimum energy E0/ENCOR
      if(used_spctr(rusdmc_.energy/ENCOR,E0/ENCOR) < drand48())
	continue;
      
      // Write out the successes into the output DST file
      write_out_event_to_dst();
    }     
  dst_close_unit_(&dst_unit_in);
  if ( rcode != END_OF_FILE ) 
    {
      fprintf(stderr,"  Error reading file\n");
      return -3;
    }
  return rcode;
}


void write_out_event_to_dst()
{
  if(!dst_unit_out_open)
    {
      integer4 dst_mode = MODE_WRITE_DST;
      if (dst_open_unit_(&dst_unit_out,outfile,&dst_mode)) 
	{ 
	  fprintf(stderr,"\n  Unable to open/write file: %s\n\n", outfile);
	  exit(2);
	}
      dst_unit_out_open = 1;
    }
  if (eventWrite(dst_unit_out,banklist_got,TRUE)< 0) 
    {
      fprintf(stderr, "  Failed to write event\n");
      dst_close_unit_(&dst_unit_in);
      exit(2);
    }
}

real8 e_minus_3_spctr(real8 energy, real8 energy_norm)
{
  const real8 INDX1 = 3.0;
  const real8 p1    = INDX1-INDX0;
  return pow(energy/energy_norm, -p1);
}

real8 hires_spctr_prl2008(real8 energy, real8 energy_norm)
{
  const real8 INDX1 = 3.25;          // Spectral index before the Ankle
  const real8 E1    = 4.42788445031; // Energy of the Ankle
  const real8 INDX2 = 2.81;          // Spectral index after the ankle and before the GZK cutoff
  const real8 E2    = 56.234132519;  // Energy of the GZK cutoff
  const real8 INDX3 = 5.1;           // Spectral index after the GZK cutoff
  const real8 p1    = INDX1-INDX0;
  const real8 p2    = INDX2-INDX0;
  const real8 p3    = INDX3-INDX0;
  if(energy <= E1)
    {
      if(E1 < energy_norm)
	return 1.0;
      return pow(energy/energy_norm,-p1);
    }
  else if (E1 < energy && energy <= E2)
    {
      real8 a = 1.0;
      if(energy_norm < E1)
	a *= pow(E1/energy_norm,p2-p1);
      return a * pow(energy/energy_norm,-p2);
    }
  else
    {
      real8 a = 1.0;
      if(energy_norm < E1)
	a *= pow(E1/energy_norm,p2-p1);
      if(energy_norm < E2)
	a *= pow(E2/energy_norm,p3-p2);
      return a * pow(energy/energy_norm,-p3);
    }
}

real8 tasd_spctr_icrc2015(real8 energy, real8 energy_norm)
{
  const real8 INDX1 = 3.298;         // Spectral index before the Ankle
  const real8 E1    = 4.9888448746;  // Energy of the Ankle
  const real8 INDX2 = 2.677;         // Spectral index after the ankle and before the GZK cutoff
  const real8 E2    = 60.6736329589; // Energy of the GZK cutoff
  const real8 INDX3 = 4.548;         // Spectral index after the GZK cutoff
  const real8 p1    = INDX1-INDX0;
  const real8 p2    = INDX2-INDX0;
  const real8 p3    = INDX3-INDX0;
  if(energy <= E1)
    {
      if(E1 < energy_norm)
	return 1.0;
      return pow(energy/energy_norm,-p1);
    }
  else if (E1 < energy && energy <= E2)
    {
      real8 a = 1.0;
      if(energy_norm < E1)
	a *= pow(E1/energy_norm,p2-p1);
      return a * pow(energy/energy_norm,-p2);
    }
  else
    {
      real8 a = 1.0;
      if(energy_norm < E1)
	a *= pow(E1/energy_norm,p2-p1);
      if(energy_norm < E2)
	a *= pow(E2/energy_norm,p3-p2);
      return a * pow(energy/energy_norm,-p3);
    }
}
