/*
 * C functions for trumpmc
 * DRB 2009/01/20
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "trumpmc_dst.h"  
#include "caldat.h"

trumpmc_dst_common trumpmc_;  /* allocate memory to trumpmc_common */

static integer4 trumpmc_blen = 0; 
static integer4 trumpmc_maxlen = sizeof(integer4) * 2 + sizeof(trumpmc_dst_common);
static integer1 *trumpmc_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* trumpmc_bank_buffer_ (integer4* trumpmc_bank_buffer_size)
{
  (*trumpmc_bank_buffer_size) = trumpmc_blen;
  return trumpmc_bank;
}



static void trumpmc_abank_init(integer1* (*pbank) ) {
  *pbank = (integer1 *)calloc(trumpmc_maxlen, sizeof(integer1));
  if (*pbank==NULL) {
    fprintf (stderr,"trumpmc_abank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  } 
}    
static void trumpmc_bank_init() {trumpmc_abank_init(&trumpmc_bank);}

integer4 trumpmc_common_to_bank_() {
  if (trumpmc_bank == NULL) trumpmc_bank_init();
  return trumpmc_struct_to_abank_(&trumpmc_, &trumpmc_bank, TRUMPMC_BANKID, TRUMPMC_BANKVERSION);
}
integer4 trumpmc_bank_to_dst_ (integer4 *unit) {return trumpmc_abank_to_dst_(trumpmc_bank, unit);}
integer4 trumpmc_common_to_dst_(integer4 *unit) {
  return trumpmc_struct_to_dst_(&trumpmc_, &trumpmc_bank, unit, TRUMPMC_BANKID, TRUMPMC_BANKVERSION);
}
integer4 trumpmc_bank_to_common_(integer1 *bank) {return trumpmc_abank_to_struct_(bank, &trumpmc_);}
integer4 trumpmc_common_to_dump_(integer4 *opt) {return trumpmc_struct_to_dumpf_(&trumpmc_, stdout, opt);}
integer4 trumpmc_common_to_dumpf_(FILE* fp, integer4 *opt) {return trumpmc_struct_to_dumpf_(&trumpmc_, fp, opt);}

integer4 trumpmc_struct_to_abank_(trumpmc_dst_common *trumpmc, integer1* (*pbank), integer4 id, integer4 ver) {
  integer4 rcode, nobj;
  integer1 *bank;

  int i, j;

  if ( *pbank == NULL ) trumpmc_abank_init(pbank);
    
  /* Initialize test_blen, and pack the id and version to bank */
  bank = *pbank;
  rcode = dst_initbank_(&id, &ver, &trumpmc_blen, &trumpmc_maxlen, bank);

  // Shower generic information
  nobj = 3;
  rcode += dst_packr4_(&trumpmc->impactPoint[0],    &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  rcode += dst_packr4_(&trumpmc->showerVector[0],   &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 1;
  rcode += dst_packr4_(&trumpmc->energy,         &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  rcode += dst_packi4_(&trumpmc->primary,        &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 4;
  rcode += dst_packr4_(&trumpmc->ghParm[0],        &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Site specific information: geometry
  nobj = 1;
  rcode += dst_packi4_(&trumpmc->nSites,         &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nSites;
  rcode += dst_packi4_(&trumpmc->siteid[0],     &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 3;
  for (i=0; i<trumpmc->nSites; i++)
    rcode += dst_packr4_(&trumpmc->siteLocation[i][0],   &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nSites;
  rcode += dst_packr4_(&trumpmc->psi[0],            &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 3;
  for (i=0; i<trumpmc->nSites; i++)
    rcode += dst_packr4_(&trumpmc->rp[i][0],             &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Depth information: Flux (photons/m2/rad)
  nobj = 1;
  rcode += dst_packi4_(&trumpmc->nDepths,        &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  rcode += dst_packr4_(&trumpmc->depth[0],          &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nSites;
  rcode += dst_packi4_(&trumpmc->nMirrors[0],       &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  for ( i=0; i<trumpmc->nSites; i++ ) {
    nobj = trumpmc->nMirrors[i];
    rcode += dst_packi4_(&trumpmc->mirror[i][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_packr4_(&trumpmc->fluoFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_packr4_(&trumpmc->raylFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_packr4_(&trumpmc->aeroFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_packr4_(&trumpmc->dirCFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Mirror/Tube information
  for (i=0; i<trumpmc->nSites; i++) {
    nobj = trumpmc->nMirrors[i];
    rcode += dst_packi4_(&trumpmc->totalNPEMirror[i][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }

  for (i=0; i<trumpmc->nSites; i++) {
    nobj = trumpmc->nMirrors[i];
    rcode += dst_packi4_(&trumpmc->nTubes[i][0],         &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }

  for (i=0; i<trumpmc->nSites; i++) {
    for (j=0; j<trumpmc->nMirrors[i]; j++) {
      nobj = trumpmc->nTubes[i][j];
      rcode += dst_packi4_(&trumpmc->tube[i][j][0],     &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    }
  }

  for (i=0; i<trumpmc->nSites; i++) {
    for (j=0; j<trumpmc->nMirrors[i]; j++) {
      nobj = trumpmc->nTubes[i][j];
      rcode += dst_packr4_(&trumpmc->aveTime[i][j][0],  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    }
  }

  for (i=0; i<trumpmc->nSites; i++) {
    for (j=0; j<trumpmc->nMirrors[i]; j++) {
      nobj = trumpmc->nTubes[i][j];
      rcode += dst_packi4_(&trumpmc->totalNPE[i][j][0],  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    }
  }

  nobj = 1;
  rcode += dst_packi4_(&trumpmc->julian,  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  rcode += dst_packi4_(&trumpmc->jsec,  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  rcode += dst_packi4_(&trumpmc->nano,  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  
  
  return rcode ;
}

integer4 trumpmc_abank_to_dst_(integer1 *bank, integer4 *unit) {return dst_write_bank_(unit, &trumpmc_blen, bank);}

integer4 trumpmc_struct_to_dst_(trumpmc_dst_common *trumpmc, integer1* (*pbank), integer4 *unit, integer4 id, integer4 ver) {
  integer4 rcode;
  if ( (rcode = trumpmc_struct_to_abank_(trumpmc, pbank, id, ver)) ) {
      fprintf(stderr, "trumpmc_struct_to_abank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  if ( (rcode = trumpmc_abank_to_dst_(*pbank, unit)) ) {
      fprintf(stderr, "trumpmc_abank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  return 0;
}

integer4 trumpmc_abank_to_struct_(integer1 *bank, trumpmc_dst_common *trumpmc) {
  integer4 rcode = 0 ;
  integer4 nobj, i, j;
 integer4 id, ver;

  // Don't skip ID and version
//   trumpmc_blen = 2 * sizeof(integer4);        /* skip id and version  */
 trumpmc_blen = 0;

  // We don't use this yet, but may eventually
  /* get id and version */
 nobj = 1;
 rcode += dst_unpacki4_(&id, &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
 rcode += dst_unpacki4_(&ver, &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Shower generic information
  nobj = 3;
  rcode += dst_unpackr4_(&trumpmc->impactPoint[0],    &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  rcode += dst_unpackr4_(&trumpmc->showerVector[0],   &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 1;
  rcode += dst_unpackr4_(&trumpmc->energy,         &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  rcode += dst_unpacki4_(&trumpmc->primary,        &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 4;
  rcode += dst_unpackr4_(&trumpmc->ghParm[0],        &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Site specific information: geometry
  nobj = 1;
  rcode += dst_unpacki4_(&trumpmc->nSites,         &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nSites;
  rcode += dst_unpacki4_(&trumpmc->siteid[0],     &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 3;
  for (i=0; i<trumpmc->nSites; i++)
    rcode += dst_unpackr4_(&trumpmc->siteLocation[i][0],   &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nSites;
  rcode += dst_unpackr4_(&trumpmc->psi[0],            &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = 3;
  for (i=0; i<trumpmc->nSites; i++)
    rcode += dst_unpackr4_(&trumpmc->rp[i][0],             &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Depth information: Flux (photons/m2/rad)
  nobj = 1;
  rcode += dst_unpacki4_(&trumpmc->nDepths,        &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  rcode += dst_unpackr4_(&trumpmc->depth[0],          &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nSites;
  rcode += dst_unpacki4_(&trumpmc->nMirrors[0],       &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  for ( i=0; i<trumpmc->nSites; i++ ) {
    nobj = trumpmc->nMirrors[i];
    rcode += dst_unpacki4_(&trumpmc->mirror[i][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_unpackr4_(&trumpmc->fluoFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_unpackr4_(&trumpmc->raylFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_unpackr4_(&trumpmc->aeroFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  nobj = trumpmc->nDepths;
  for (i=0; i<trumpmc->nSites; i++)
    for (j=0; j<trumpmc->nMirrors[i]; j++)
      rcode += dst_unpackr4_(&trumpmc->dirCFlux[i][j][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);

  // Mirror/Tube information
  for (i=0; i<trumpmc->nSites; i++) {
    nobj = trumpmc->nMirrors[i];
    rcode += dst_unpacki4_(&trumpmc->totalNPEMirror[i][0], &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }

  for (i=0; i<trumpmc->nSites; i++) {
    nobj = trumpmc->nMirrors[i];
    rcode += dst_unpacki4_(&trumpmc->nTubes[i][0],         &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }

  for (i=0; i<trumpmc->nSites; i++) {
    for (j=0; j<trumpmc->nMirrors[i]; j++) {
      nobj = trumpmc->nTubes[i][j];
      rcode += dst_unpacki4_(&trumpmc->tube[i][j][0],     &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    }
  }

  for (i=0; i<trumpmc->nSites; i++) {
    for (j=0; j<trumpmc->nMirrors[i]; j++) {
      nobj = trumpmc->nTubes[i][j];
      rcode += dst_unpackr4_(&trumpmc->aveTime[i][j][0],  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    }
  }

  for (i=0; i<trumpmc->nSites; i++) {
    for (j=0; j<trumpmc->nMirrors[i]; j++) {
      nobj = trumpmc->nTubes[i][j];
      rcode += dst_unpacki4_(&trumpmc->totalNPE[i][j][0],  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    }
  }

  if (ver >= 1) {
    nobj = 1;
    rcode += dst_unpacki4_(&trumpmc->julian,  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    rcode += dst_unpacki4_(&trumpmc->jsec,  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
    rcode += dst_unpacki4_(&trumpmc->nano,  &nobj, bank, &trumpmc_blen, &trumpmc_maxlen);
  }
  else {
    trumpmc->julian = -1;
    trumpmc->jsec = -1;
    trumpmc->nano = -1;
  }
  
  return rcode ;
}

integer4 trumpmc_struct_to_dump_(trumpmc_dst_common *trumpmc, integer4 *long_output) {
  return trumpmc_struct_to_dumpf_(trumpmc, stdout, long_output);
}

integer4 trumpmc_struct_to_dumpf_(trumpmc_dst_common *tmc, FILE* fp,integer4 *long_output) {
  int i,j,k;
  int ymd, hms, nano;
  int yr, mo, day, hr, min, sec;
  if (tmc->nano < 0) {
      ymd = 0;
      hms = 0;
      nano = 0;
    }
    else {
      hr = tmc->jsec / 3600 + 12;

      if (hr >= 24) {
        caldat((double)tmc->julian+1., &mo, &day, &yr);
        hr -= 24;
      }
      else
        caldat((double)tmc->julian, &mo, &day, &yr);

      min = ( tmc->jsec / 60 ) % 60;
      sec = tmc->jsec % 60;
      ymd = 10000*yr + 100*mo + day;
      hms = 10000*hr + 100*min + sec;
      nano = tmc->nano;
    }
  
  
  fprintf(fp, "TRUMPMC :\n");
  fprintf(fp, " Primary: %4d, Energy: %8.2e (eV)\n", tmc->primary, tmc->energy);
  fprintf(fp, " Impact Point (m): (%7.0f,%7.0f,%7.0f)\n",
	  tmc->impactPoint[0],tmc->impactPoint[1],tmc->impactPoint[2]);
  fprintf(fp, " Impact time: %08d %06d.%09d\n",
            ymd,hms,nano);
  fprintf(fp, " Shower Vector:    (%7.4f,%7.4f,%7.4f)\n",
	  tmc->showerVector[0],tmc->showerVector[1],tmc->showerVector[2]);
  fprintf(fp, " GH Fit, Xo %3.0f, Xmax %6.1f, Nmax %8.2e, Lambda %2.0f\n",
	  tmc->ghParm[0],tmc->ghParm[1],tmc->ghParm[2],tmc->ghParm[3]);

  fprintf(fp, "\n Sites: %d\n", tmc->nSites);
  for (i=0; i<tmc->nSites; i++) {
    fprintf(fp, "  Site %d: Location (m) ( %6.0f, %6.0f, %6.0f), Psi %6.1f\n", tmc->siteid[i],
	    tmc->siteLocation[i][0],tmc->siteLocation[i][1],tmc->siteLocation[i][2],
	    R2D*tmc->psi[i]); 
    double mRp = 0.;
    for (j=0;j<3;j++) mRp += tmc->rp[i][j]*tmc->rp[i][j];
    mRp = sqrt(mRp);
    fprintf(fp, "   Rp (m): ( %12.5f, %12.5f, %12.5f) |Rp|= %12.5f\n",
	    tmc->rp[i][0],tmc->rp[i][1],tmc->rp[i][2],mRp);
  }
  
  if (*long_output != 0) {
    if (tmc->primary)
      fprintf(fp, "\n Flux by Depth (ph/m2/rad): %d\n", tmc->nDepths);
    else
      fprintf(fp, "\n Flux by altitude (ph/m2/rad): %d\n", tmc->nDepths);
    for (i=0; i<tmc->nSites; i++)
      for (j=0; j<tmc->nMirrors[i]; j++) {
        if (tmc->primary)
          fprintf(fp, "  s%1dm%1X:   X    Fluor Rayleigh  Aerosol DirectCk\n",
            i,tmc->mirror[i][j]);
        else
          fprintf(fp, "  s%1dm%1X:   Alt  Fluor Rayleigh  Aerosol DirectCk\n",
            i,tmc->mirror[i][j]);
        for (k=0; k<tmc->nDepths; k++)
        // fprintf(fp, "  s%1dm%1X:x   X    Fluor Rayleigh  Aerosol DirectCk\n");
        // fprintf(fp, "   %1d %1X %4.0f ...%8.2e ...%8.2e ...%8.2e ...%8.2e\n"
          if (tmc->fluoFlux[i][j][k] > 0. || tmc->raylFlux[i][j][k] > 0. || 
              tmc->aeroFlux[i][j][k] > 0. || tmc->dirCFlux[i][j][k] > 0.)
            fprintf(fp, "   %1d %1X %4.0f %8.2e %8.2e %8.2e %8.2e\n",i,tmc->mirror[i][j],
              tmc->depth[k],tmc->fluoFlux[i][j][k]/R2D,tmc->raylFlux[i][j][k]/R2D,
              tmc->aeroFlux[i][j][k]/R2D,tmc->dirCFlux[i][j][k]/R2D);
      }

    fprintf(fp, "\n Mirrors/Tubes:\n");
    for (i=0; i<tmc->nSites; i++) {
      fprintf(fp, "  Site %1d: Mirrors %d\n",i,tmc->nMirrors[i]);
      for (j=0; j<tmc->nMirrors[i]; j++) {
        fprintf(fp, "   Mirror %1X: NPE %4d, Tubes %d\n",
          tmc->mirror[i][j],tmc->totalNPEMirror[i][j],tmc->nTubes[i][j]);
        if (tmc->nTubes[i][j]>0)
          fprintf(fp, "    ID   time  NPE\n");
        for (k=0; k<tmc->nTubes[i][j]; k++) 
          // fprintf(fp, "    xID ..time .NPE\n");
          // fprintf(fp, "    %2X .%6.0f .%4d\n")
          fprintf(fp, "    %2X %6.0f %4d\n",tmc->tube[i][j][k],
            tmc->aveTime[i][j][k],tmc->totalNPE[i][j][k]);
      }
    }
  }
  
  return 0;
} 
