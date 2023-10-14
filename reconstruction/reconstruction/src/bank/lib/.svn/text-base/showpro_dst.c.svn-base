#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"
 
#include "univ_dst.h"
#include "showpro_dst.h"

showpro_dst_common showpro_;  /* allocate memory to showpro_common */

static integer4 showpro_blen = 0;
static integer4 showpro_maxlen =  sizeof (integer4) * 2 + sizeof (showpro_dst_common);
static integer1 *showpro_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* showpro_bank_buffer_ (integer4* showpro_bank_buffer_size)
{
  (*showpro_bank_buffer_size) = showpro_blen;
  return showpro_bank;
}



static void showpro_bank_init() {
  showpro_bank = (integer1 *) calloc (showpro_maxlen, sizeof (integer1));
  if (showpro_bank == NULL){
    fprintf (stderr, "showpro_bank_init: fail to assign memory to bank. Abort.\n");
    exit (0);
  }
}

integer4 showpro_common_to_bank_() {
  static integer4 id = SHOWPRO_BANKID, ver = SHOWPRO_BANKVERSION;
  integer4 rcode, nobj;

  if(showpro_bank == NULL){
    showpro_bank_init ();
  }
  rcode =  dst_initbank_ (&id, &ver, &showpro_blen, &showpro_maxlen, showpro_bank);
  nobj = 1;
  
  rcode += dst_packi4_(&showpro_.event_num, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen); 
  rcode += dst_packi4_(&showpro_.parttype, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  
  rcode += dst_packr8_(&showpro_.eScale, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr8_(&showpro_.zen, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  /*
  rcode += dst_packr8_(&showpro_.energy, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr8_(&showpro_.zen, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr8_(&showpro_.azm, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr8_(&showpro_.xcore, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr8_(&showpro_.ycore, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr8_(&showpro_.height, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  */
  rcode += dst_packi4_(&showpro_.nslices, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  nobj = showpro_.nslices;  
  rcode += dst_packr4_(&showpro_.x[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.gammas[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.electrons[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.positrons[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.muonPlus[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.muonMinus[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.hadrons[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.charged[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.nuclei[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.cherenkov[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  
  nobj = 1;
  rcode += dst_packi4_(&showpro_.nDepSlices, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  
  nobj = showpro_.nDepSlices;
  rcode += dst_packr4_(&showpro_.xDep[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.gammaDep[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.emIoniz[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.emCut[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.muIoniz[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.hadrIoniz[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.hadrCut[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.neutrino[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.total[0], &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  
  nobj = 1;
  rcode += dst_packr4_(&showpro_.x0, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.xmax, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.nmax, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.lambda0, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.lambda1, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.lambda2, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.chi2, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_packr4_(&showpro_.aveDev, &nobj, showpro_bank, &showpro_blen, &showpro_maxlen);
  
  return rcode;
}

integer4 showpro_bank_to_dst_(integer4 * unit) {
  integer4 rcode;
  rcode = dst_write_bank_(unit, &showpro_blen, showpro_bank);
  free (showpro_bank);
  showpro_bank = NULL;
  
  return rcode;
}

integer4 showpro_common_to_dst_(integer4 * unit) {
  integer4 rcode;
  if ( (rcode = showpro_common_to_bank_()) ){
    fprintf (stderr, "showpro_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
  if ( (rcode = showpro_bank_to_dst_(unit) )){
    fprintf (stderr, "showpro_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
  return 0;
} 

integer4 showpro_bank_to_common_(integer1 * bank) {

  integer4 rcode = 0;
  integer4 nobj;
  showpro_blen = 2 * sizeof (integer4);  // skip id and version

  nobj = 1;
    nobj = 1;
  
  rcode += dst_unpacki4_ (&showpro_.event_num, &nobj, bank, &showpro_blen, &showpro_maxlen); 
  rcode += dst_unpacki4_ (&showpro_.parttype, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_ (&showpro_.eScale, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_ (&showpro_.zen, &nobj, bank, &showpro_blen, &showpro_maxlen);
  /*
  rcode += dst_unpackr8_(&showpro_.energy, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_(&showpro_.zen, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_(&showpro_.azm, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_(&showpro_.xcore, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_(&showpro_.ycore, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr8_(&showpro_.height, &nobj, bank, &showpro_blen, &showpro_maxlen);
  */
  rcode += dst_unpacki4_(&showpro_.nslices, &nobj, bank, &showpro_blen, &showpro_maxlen);
  nobj = showpro_.nslices;  
  rcode += dst_unpackr4_(&showpro_.x[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.gammas[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.electrons[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.positrons[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.muonPlus[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.muonMinus[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.hadrons[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.charged[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.nuclei[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.cherenkov[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  
  nobj = 1;
  rcode += dst_unpacki4_(&showpro_.nDepSlices, &nobj, bank, &showpro_blen, &showpro_maxlen);
  
  nobj = showpro_.nDepSlices;
  rcode += dst_unpackr4_(&showpro_.xDep[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.gammaDep[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.emIoniz[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.emCut[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.muIoniz[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.hadrIoniz[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.hadrCut[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.neutrino[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.total[0], &nobj, bank, &showpro_blen, &showpro_maxlen);
  
  nobj = 1;
  rcode += dst_unpackr4_(&showpro_.x0, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.xmax, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.nmax, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.lambda0, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.lambda1, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.lambda2, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.chi2, &nobj, bank, &showpro_blen, &showpro_maxlen);
  rcode += dst_unpackr4_(&showpro_.aveDev, &nobj, bank, &showpro_blen, &showpro_maxlen);
 
  return rcode;
}

integer4 showpro_common_to_dump_ (integer4 * long_output) {
  return showpro_common_to_dumpf_ (stdout, long_output);
}
 
integer4 showpro_common_to_dumpf_ (FILE * fp, integer4 * long_output) {
  fprintf(fp, "SHOWPRO BANK\n");
  fprintf(fp, "event_num: %d, particle type: %d\n", showpro_.event_num, showpro_.parttype);
  fprintf(fp, "Thrown Shower Parameters:\n");
  fprintf(fp, "Energy / CORSIKA Energy: %5.3g\n", showpro_.eScale);
  fprintf(fp, "zenith: %5.3g\n", showpro_.zen);
  /*
  fprintf(fp, "energy: %5.3g [EeV]\n", showpro_.energy);
  fprintf(fp, "(zen, azm) = (%5.3g, %5.3g)\n", showpro_.zen*R2D, showpro_.azm*R2D);
  fprintf(fp, "Core location (km from clf): (%5.3g, %5.3g)\n", showpro_.xcore/1000, showpro_.ycore/1000);
  fprintf(fp, "First Interaction from CORSIKA: %e [cm]", showpro_.height);
  */  

  fprintf(fp, "\nCORISIKA GH Fit Parameteres\n");
  fprintf(fp, "x0      = %5.3g [g/cm^2]\n", showpro_.x0);
  fprintf(fp, "xmax    = %5.3g [g/cm^2]\n", showpro_.xmax);
  fprintf(fp, "Nmax    = %5.3g [g/cm^2]\n", showpro_.nmax);
  fprintf(fp, "lambda0 = %5.3g [g/cm^2]\n", showpro_.lambda0);
  fprintf(fp, "lambda1 = %5.3g [g/cm^2]\n", showpro_.lambda1);
  fprintf(fp, "lambda2 = %5.3g [g/cm^2]\n", showpro_.lambda2);
  fprintf(fp, "chi2    = %5.3g [g/cm^2]\n", showpro_.chi2);
  fprintf(fp, "aveDev  = %5.3g [g/cm^2]\n", showpro_.aveDev);
  
  if(*long_output == 1){
    fprintf(fp, "slices = %d\n", showpro_.nslices);
      printf("x[g/cm^2]  gammas     electrons  postirons  muon-      muon+\n");
    int i;
    for(i=0;i<showpro_.nslices;i++){

      printf("%7.2f    %5.3e  %5.3e  %5.3e  %5.3e  %5.3e\n", 
             showpro_.x[i], showpro_.gammas[i], showpro_.electrons[i], showpro_.positrons[i], 
             showpro_.muonMinus[i], showpro_.muonPlus[i]);  
    }
    printf("%7s %5.3s %5.3s %5.3s %5.3s %5.3s\n", "x[g/cm^2]", "gammaDep", "emCut", "emIoniz", "neutrino", "total");
    for(i=0;i<showpro_.nDepSlices;i++){

      printf("%7.2f %5.3e %5.3e %5.3e %5.3e %5.3e\n", 
             showpro_.xDep[i], showpro_.gammaDep[i], showpro_.emIoniz[i], showpro_.emCut[i], 
             showpro_.neutrino[i], showpro_.total[i]);  
    }
  }
  return 0;
} 