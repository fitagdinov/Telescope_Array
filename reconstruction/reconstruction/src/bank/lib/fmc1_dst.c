/*
 * C functions for fmc1
 * MRM July 27
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fmc1_dst.h"  

fmc1_dst_common fmc1_;  /* allocate memory to fmc1_common */

static integer4 fmc1_blen = 0; 
static integer4 fmc1_maxlen = sizeof(integer4) * 2 + sizeof(fmc1_dst_common);
static integer1 *fmc1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fmc1_bank_buffer_ (integer4* fmc1_bank_buffer_size)
{
  (*fmc1_bank_buffer_size) = fmc1_blen;
  return fmc1_bank;
}



static void fmc1_bank_init()
{
  fmc1_.version = FMC1_BANKVERSION;
  fmc1_bank = (integer1 *)calloc(fmc1_maxlen, sizeof(integer1));
  if (fmc1_bank==NULL)
    {
      fprintf (stderr,"fmc1_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}    

integer4 fmc1_common_to_bank_()
{
  static integer4 id = FMC1_BANKID, ver = FMC1_BANKVERSION;
  integer4 i, rcode, nobj;

  if (fmc1_bank == NULL) fmc1_bank_init();
    
  rcode = dst_initbank_(&id, &ver, &fmc1_blen, &fmc1_maxlen, fmc1_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj=1;
  rcode += dst_packr4_(&fmc1_.the, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.phi, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.agam, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.energy, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.tau0, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.nmax, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.xmax, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.pfwd, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.a, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.rp_true, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.psi_true, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.t0_true, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.R_impact, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(&fmc1_.phi_impact, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packi4_(&fmc1_.nmir, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  if (fmc1_.version) {
    rcode += dst_packr4_(&fmc1_.R_shwmax, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.the_shwmax, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.R_viewmax, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.R_viewfirst, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.R_viewlast, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packi4_(&fmc1_.event_num, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  }
  if (fmc1_.version > 1) {
    rcode += dst_packr4_(&fmc1_.hal, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.vsh, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.mlh, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.atm0, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.atm1, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.atm2, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  }
  nobj=3;
  rcode += dst_packr4_(fmc1_.rpuv, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(fmc1_.shwn, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packr4_(fmc1_.u, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);

  nobj= fmc1_.nmir;
  rcode += dst_packi4_(fmc1_.imir, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packi4_(fmc1_.ntube, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_packi4_(fmc1_.npe_mir, &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);


  for (i=0;i<fmc1_.nmir;i++) {
    nobj=fmc1_.ntube[i];
    rcode += dst_packi4_(&fmc1_.itube[i][0], &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packr4_(&fmc1_.tav[i][0], &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_packi4_(&fmc1_.npe[i][0], &nobj, fmc1_bank, &fmc1_blen, &fmc1_maxlen);

  }
  
  return rcode ;
}

integer4 fmc1_bank_to_dst_ (integer4 *unit)
{
  return dst_write_bank_(unit, &fmc1_blen, fmc1_bank);
}

integer4 fmc1_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = fmc1_common_to_bank_()) )
    {
      fprintf(stderr, "fmc1_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
    if ( (rcode = fmc1_bank_to_dst_(unit) ))
    {
      fprintf(stderr, "fmc1_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  return 0;
}

integer4 fmc1_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0 ;
  integer4 i, nobj;
  fmc1_blen = 2 * sizeof(integer4);	/* skip id and version  */

  nobj=1;
  rcode += dst_unpackr4_(&fmc1_.the, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.phi, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.agam, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.energy, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.tau0, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.nmax, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.xmax, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.pfwd, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.a, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.rp_true, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.psi_true, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.t0_true, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.R_impact, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(&fmc1_.phi_impact, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpacki4_(&fmc1_.nmir, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  if (fmc1_.version) {
    rcode += dst_unpackr4_(&fmc1_.R_shwmax, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.the_shwmax, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.R_viewmax, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.R_viewfirst, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.R_viewlast, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpacki4_(&fmc1_.event_num, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  }
  if (fmc1_.version > 1) {
    rcode += dst_unpackr4_(&fmc1_.hal, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.vsh, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.mlh, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.atm0, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.atm1, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.atm2, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  }
  nobj=3;
  rcode += dst_unpackr4_(fmc1_.rpuv, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(fmc1_.shwn, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpackr4_(fmc1_.u, &nobj, bank, &fmc1_blen, &fmc1_maxlen);

  nobj= fmc1_.nmir;
  rcode += dst_unpacki4_(fmc1_.imir, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpacki4_(fmc1_.ntube, &nobj, bank, &fmc1_blen, &fmc1_maxlen);
  rcode += dst_unpacki4_(fmc1_.npe_mir, &nobj, bank, &fmc1_blen, &fmc1_maxlen);


  for (i=0;i<fmc1_.nmir;i++) {
    nobj=fmc1_.ntube[i];
    rcode += dst_unpacki4_(&fmc1_.itube[i][0], &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpackr4_(&fmc1_.tav[i][0], &nobj, bank, &fmc1_blen, &fmc1_maxlen);
    rcode += dst_unpacki4_(&fmc1_.npe[i][0], &nobj, bank, &fmc1_blen, &fmc1_maxlen);

  }

  return rcode ;
}

integer4 fmc1_common_to_dump_(integer4 *long_output)
{
  return fmc1_common_to_dumpf_(stdout,long_output);
}

integer4 fmc1_common_to_dumpf_(FILE* fp,integer4 *long_output)
{
  int j,k;

#define FMC1_RADIAN 57.29577951

  fprintf (fp, "FMC1 :");
  if (fmc1_.version) fprintf (fp, " event_num %d",fmc1_.event_num);
  fprintf (fp, "\n");
  fprintf (fp, " energy %9.2e    tau0 %12.1f      A %18.0f\n",
           fmc1_.energy,fmc1_.tau0,fmc1_.a);
  fprintf (fp, " Xmax %8.0f       Nmax %15.3e   Profile Width %6.0f\n",
           fmc1_.xmax,fmc1_.nmax*1e9,fmc1_.pfwd);
  fprintf (fp, " the %10.2f      phi %14.2f     agam %15.2f\n",
           fmc1_.the*FMC1_RADIAN,fmc1_.phi*FMC1_RADIAN,fmc1_.agam*FMC1_RADIAN);
  fprintf (fp, " rp_true %6.2f      psi_true %9.2f     t0_true %12.1f\n",
           fmc1_.rp_true,FMC1_RADIAN*fmc1_.psi_true,fmc1_.t0_true);
  fprintf (fp, " R_impact %6.2f      phi_impact %7.2f     nmir %15d\n",
           fmc1_.R_impact,fmc1_.phi_impact,fmc1_.nmir);
  if (fmc1_.version) {
    fprintf(fp, " R_shwmax %6.2f  theta_view_shwmax %6.2f\n",
            fmc1_.R_shwmax,fmc1_.the_shwmax);
    fprintf(fp, " R_signal_first %6.2f R_signal_last %6.2f R_signal_max %6.2f\n",
            fmc1_.R_viewfirst,fmc1_.R_viewlast,fmc1_.R_viewmax);
  }
  if (fmc1_.version > 1) {
    fprintf(fp, " Aerosol horizontal attenuation length %6.2f\n",fmc1_.hal);
    fprintf(fp, " Aerosol vertical scale height %6.2f\n",fmc1_.vsh);
    fprintf(fp, " Aerosol mixing layer height %6.2f\n",fmc1_.mlh);
  }
  fprintf(fp, "rpuv (%7.4f, %7.4f, %7.4f)\n",fmc1_.rpuv[0],fmc1_.rpuv[1],fmc1_.rpuv[2]);
  fprintf(fp, "shwn (%7.4f, %7.4f, %7.4f)\n",fmc1_.shwn[0],fmc1_.shwn[1],fmc1_.shwn[2]);
  fprintf(fp, "u    (%7.4f, %7.4f, %7.4f)\n",fmc1_.u[0],fmc1_.u[1],fmc1_.u[2]);
  for (j=0;j<fmc1_.nmir;j++) {
      fprintf(fp," imir %4d ntube %4d npe_mir %8d\n",
              fmc1_.imir[j],fmc1_.ntube[j],fmc1_.npe_mir[j]);
      if (*long_output==1) {
          for (k=0;k<fmc1_.ntube[j];k++) {
              fprintf(fp," tube %4d npe %6d tav %6.2f\n",
                      fmc1_.itube[j][k],fmc1_.npe[j][k],fmc1_.tav[j][k]);
          }
      }
  }
  
  return 0;
}










