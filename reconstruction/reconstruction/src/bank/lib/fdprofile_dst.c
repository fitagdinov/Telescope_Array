// Created 2008/09/23 DRB LMS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdprofile_dst.h"

fdprofile_dst_common fdprofile_;

integer4 fdprofile_blen = 0; /* not static because it needs to be accessed by the c files of the derived banks */
static integer4 fdprofile_maxlen = sizeof(integer4) * 2 + sizeof(fdprofile_dst_common);
static integer1 *fdprofile_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdprofile_bank_buffer_ (integer4* fdprofile_bank_buffer_size)
{
  (*fdprofile_bank_buffer_size) = fdprofile_blen;
  return fdprofile_bank;
}



static void fdprofile_abank_init(integer1* (*pbank) ) {
  *pbank = (integer1 *)calloc(fdprofile_maxlen, sizeof(integer1));
  if (*pbank==NULL) {
      fprintf (stderr,"fdprofile_abank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

static void fdprofile_bank_init() {fdprofile_abank_init(&fdprofile_bank);}

integer4 fdprofile_common_to_bank_() {
  if (fdprofile_bank == NULL) fdprofile_bank_init();
  return fdprofile_struct_to_abank_(&fdprofile_, &fdprofile_bank, FDPROFILE_BANKID, FDPROFILE_BANKVERSION);
}
integer4 fdprofile_bank_to_dst_ (integer4 *unit) {return fdprofile_abank_to_dst_(fdprofile_bank, unit);}
integer4 fdprofile_common_to_dst_(integer4 *unit) {
  if (fdprofile_bank == NULL) fdprofile_bank_init();
  return fdprofile_struct_to_dst_(&fdprofile_, fdprofile_bank, unit, FDPROFILE_BANKID, FDPROFILE_BANKVERSION);
}
integer4 fdprofile_bank_to_common_(integer1 *bank) {return fdprofile_abank_to_struct_(bank, &fdprofile_);}
integer4 fdprofile_common_to_dump_(integer4 *opt) {return fdprofile_struct_to_dumpf_(&fdprofile_, stdout, opt);}
integer4 fdprofile_common_to_dumpf_(FILE* fp, integer4 *opt) {return fdprofile_struct_to_dumpf_(&fdprofile_, fp, opt);}

integer4 fdprofile_struct_to_abank_(fdprofile_dst_common *fdprofile, integer1 *(*pbank), integer4 id, integer4 ver) {
  integer4 rcode, nobj;
  integer1 *bank;

  int i;

  if (*pbank == NULL) fdprofile_abank_init(pbank);

  bank = *pbank;
  rcode = dst_initbank_(&id, &ver, &fdprofile_blen, &fdprofile_maxlen, bank);

// Initialize fdprofile_blen and pack the id and version to bank

  nobj = 1;
  rcode += dst_packi4_(&fdprofile->siteid,      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packi4_(&fdprofile->ntslice,     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = 3;
  rcode += dst_packi4_(&fdprofile->ngtslice[0], &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packi4_(&fdprofile->status[0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = fdprofile->ntslice;
  rcode += dst_packi4_(&fdprofile->timebin[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = 3;
  rcode += dst_packr8_(&fdprofile->rp[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->psi[0], &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->t0[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  rcode += dst_packr8_(&fdprofile->Xmax[0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->eXmax[0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->Nmax[0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->eNmax[0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->Energy[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->eEnergy[0], &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->chi2[0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = fdprofile->ntslice;
  rcode += dst_packr8_(&fdprofile->npe[0],      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_packr8_(&fdprofile->enpe[0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  for (i=0; i<3; i++) {
    nobj = fdprofile->ntslice;

    rcode += dst_packr8_(&fdprofile->x[i][0],        &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_packr8_(&fdprofile->dtheta[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->darea[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_packr8_(&fdprofile->acpt[i][0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->eacpt[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_packr8_(&fdprofile->flux[i][0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->eflux[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_packr8_(&fdprofile->nfl[i][0],      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->ncvdir[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->ncvmie[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->ncvray[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->simflux[i][0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_packr8_(&fdprofile->tres[i][0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->tchi2[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_packr8_(&fdprofile->ne[i][0],       &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_packr8_(&fdprofile->ene[i][0],      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  }

  nobj = 1;
  rcode += dst_packi4_(&fdprofile->mc, &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  return rcode;
}

integer4 fdprofile_abank_to_dst_(integer1 *bank, integer4 *unit) {
  return dst_write_bank_(unit, &fdprofile_blen, bank);
}

integer4 fdprofile_struct_to_dst_(fdprofile_dst_common *fdprofile, integer1 *bank, integer4 *unit, integer4 id, integer4 ver) {
  integer4 rcode;
  if ( (rcode = fdprofile_struct_to_abank_(fdprofile, &bank, id, ver)) ) {
      fprintf(stderr, "fdprofile_struct_to_abank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  if ( (rcode = fdprofile_abank_to_dst_(bank, unit)) ) {
      fprintf(stderr, "fdprofile_abank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  return 0;
}

integer4 fdprofile_abank_to_struct_(integer1 *bank, fdprofile_dst_common *fdprofile) {
  integer4 rcode = 0 ;
  integer4 nobj;
  fdprofile_blen = 2 * sizeof(integer4);   /* skip id and version  */

  int i;

  nobj = 1;
  rcode += dst_unpacki4_(&fdprofile->siteid,      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpacki4_(&fdprofile->ntslice,     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = 3;
  rcode += dst_unpacki4_(&fdprofile->ngtslice[0], &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpacki4_(&fdprofile->status[0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = fdprofile->ntslice;
  rcode += dst_unpacki4_(&fdprofile->timebin[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = 3;
  rcode += dst_unpackr8_(&fdprofile->rp[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->psi[0], &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->t0[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  rcode += dst_unpackr8_(&fdprofile->Xmax[0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->eXmax[0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->Nmax[0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->eNmax[0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->Energy[0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->eEnergy[0], &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->chi2[0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  nobj = fdprofile->ntslice;
  rcode += dst_unpackr8_(&fdprofile->npe[0],      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  rcode += dst_unpackr8_(&fdprofile->enpe[0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  for (i=0; i<3; i++) {
    nobj = fdprofile->ntslice;

    rcode += dst_unpackr8_(&fdprofile->x[i][0],        &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_unpackr8_(&fdprofile->dtheta[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->darea[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_unpackr8_(&fdprofile->acpt[i][0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->eacpt[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_unpackr8_(&fdprofile->flux[i][0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->eflux[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_unpackr8_(&fdprofile->nfl[i][0],      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->ncvdir[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->ncvmie[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->ncvray[i][0],   &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->simflux[i][0],  &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_unpackr8_(&fdprofile->tres[i][0],     &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->tchi2[i][0],    &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

    rcode += dst_unpackr8_(&fdprofile->ne[i][0],       &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
    rcode += dst_unpackr8_(&fdprofile->ene[i][0],      &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);
  }

  nobj = 1;
  rcode += dst_unpacki4_(&fdprofile->mc, &nobj, bank, &fdprofile_blen, &fdprofile_maxlen);

  return rcode;
}

integer4 fdprofile_struct_to_dump_(fdprofile_dst_common *fdprofile, integer4 *long_output) {
  return fdprofile_struct_to_dumpf_(fdprofile, stdout, long_output);
}

integer4 fdprofile_struct_to_dumpf_(fdprofile_dst_common *fdprofile, FILE* fp, integer4 *long_output) {
  int i, j;

  if (fdprofile->siteid == BR)
    fprintf (fp, "\n\nBRPROFILE bank (TA FD shower profile information for Black Rock Mesa FD)\n");
  else if (fdprofile->siteid == LR)
    fprintf (fp, "\n\nLRPROFILE bank (TA FD shower profile information for Long Ridge FD)\n");
  if (fdprofile->mc == TRUE)
    fprintf (fp, "\nUses TRUMPMC geometry information\n\n");

  for (i=0; i<3; i++) {
    if      (fdprofile->status[i] == -2) {
      fprintf (fp, "\n%d : BANK NOT FILLED\n", i);
      continue;
    }
    else if (fdprofile->status[i] == -1)
      fprintf (fp, "\n%d : BAD GEOMETRY\n", i);
    else if (fdprofile->status[i] ==  0)
      fprintf (fp, "\n%d : FIT DID NOT CONVERGE\n", i);
    else if (fdprofile->status[i] ==  1)
      fprintf (fp, "\n%d : FIT CONVERGED\n", i);
    fprintf (fp, "  (Psi, Rp, t0) = (%8.3f, %8.1f, %8.1f)\n",
	     fdprofile->psi[i]*R2D, fdprofile->rp[i], fdprofile->t0[i]);
    fprintf (fp, "  %d good of %d time slices\n", fdprofile->ngtslice[i], fdprofile->ntslice);
    fprintf (fp, "  Xmax   : %6.1f +/- %6.1f\n", fdprofile->Xmax[i], fdprofile->eXmax[i]);
    fprintf (fp, "  Nmax   : %4.3e +/- %4.3e\n", fdprofile->Nmax[i], fdprofile->eNmax[i]);
    fprintf (fp, "  Energy : %4.3e +/- %4.3e\n", fdprofile->Energy[i], fdprofile->eEnergy[i]);
    fprintf (fp, "  Chi2   : %6.1f\n", fdprofile->chi2[i]);
  }

// Timeslice info
  if ( (*long_output) == 1) {
    for (i=0; i<3; i++) {
      if (fdprofile->status[i] >= 0) {
        fprintf (fp, "\nTime-slice information for %d : (Psi, Rp, t0) = (%8.3f, %8.1f, %8.1f)\n",
          i, fdprofile->psi[i]*R2D, fdprofile->rp[i], fdprofile->t0[i]);
        fprintf (fp, "idx  tbin      gram      npe     enpe      flux     eflux       nfl     cvdir     "
                     "cvmie     cvray   simflux      tres  tchi2        Ne       eNe\n");
        for (j=0; j<fdprofile->ntslice; j++)
          fprintf (fp, "%3d  %3d  %9.4f %8.2f %8.2f %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.2e %6.2f %9.3e %9.3e\n", 
                   j, fdprofile->timebin[j], fdprofile->x[i][j], fdprofile->npe[j], 
                   fdprofile->enpe[j], fdprofile->flux[i][j]/R2D, fdprofile->eflux[i][j]/R2D, 
                   fdprofile->nfl[i][j]/R2D, fdprofile->ncvdir[i][j]/R2D, fdprofile->ncvmie[i][j]/R2D, 
	           fdprofile->ncvray[i][j]/R2D, fdprofile->simflux[i][j]/R2D, 
	           fdprofile->tres[i][j], fdprofile->tchi2[i][j],
                   fdprofile->ne[i][j], fdprofile->ene[i][j]);
      }
    }
  }
  else
    fprintf (fp, "\nTime-slice information not displayed in short output\n");

  fprintf (fp, "\n\n");

  return 0;
}
