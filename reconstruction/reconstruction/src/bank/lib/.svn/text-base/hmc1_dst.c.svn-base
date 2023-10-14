/*
 * hmc1_dst.c 
 *
 * $Source: /hires_soft/uvm2k/bank/hmc1_dst.c,v $
 * $Log: hmc1_dst.c,v $
 * Revision 1.2  1998/01/15 19:05:24  tareq
 * added laser and flasher support in the common_to_dumpf function
 *
 * Revision 1.1  1997/10/04  22:37:27  tareq
 * Initial revision
 *
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "hmc1_dst.h"  

hmc1_dst_common hmc1_;  /* allocate memory to hmc1_common */

static integer4 hmc1_blen = 0; 
static integer4 hmc1_maxlen = sizeof(integer4)*2 + sizeof(hmc1_dst_common);
static integer1 *hmc1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hmc1_bank_buffer_ (integer4* hmc1_bank_buffer_size)
{
  (*hmc1_bank_buffer_size) = hmc1_blen;
  return hmc1_bank;
}



static void
hmc1_bank_init(void)
{
  hmc1_bank = (integer1 *)calloc(hmc1_maxlen, sizeof(integer1));
  if (hmc1_bank==NULL)
    {
      fprintf(stderr, 
        "hmc1_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4
hmc1_common_to_bank_(void)
{  
  static integer4 id = HMC1_BANKID, ver = HMC1_BANKVERSION;
  integer4 rcode, nobj;

  if (hmc1_bank == NULL) hmc1_bank_init();

  /* Initialize hmc1_blen, and pack the id and version to bank */

  if ( (rcode = dst_initbank_(&id, &ver, &hmc1_blen, &hmc1_maxlen, hmc1_bank)) )
    return rcode;


  if ( (rcode = dst_packr8_( hmc1_.tr_dir, (nobj=3, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( hmc1_.tr_rpvec, (nobj=3, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( &hmc1_.tr_rp, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packr8_( hmc1_.site, (nobj=3, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( hmc1_.sh_rini, (nobj=3, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( hmc1_.sh_rfin, (nobj=3, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packr8_( &hmc1_.energy, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( &hmc1_.sh_csmax, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( &hmc1_.fl_totpho, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packr8_( &hmc1_.la_wavlen, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( &hmc1_.fl_twidth, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packr8_( &hmc1_.sh_x0, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( &hmc1_.sh_xmax, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packr8_( &hmc1_.sh_xfin, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packi4_( &hmc1_.sh_iprim, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packi4_( &hmc1_.setNr, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( &hmc1_.eventNr, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( &hmc1_.evttype, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( &hmc1_.iseed1, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( &hmc1_.iseed2, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packi4_( &hmc1_.nmir, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( &hmc1_.ntube, (nobj=1, &nobj), hmc1_bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_packi4_( hmc1_.tubemir, (nobj=hmc1_.ntube, &nobj),
                 hmc1_bank, &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( hmc1_.tube, (nobj=hmc1_.ntube, &nobj),
                 hmc1_bank, &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_packi4_( hmc1_.pe, (nobj=hmc1_.ntube, &nobj),
                 hmc1_bank, &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  return SUCCESS;
}


integer4
hmc1_bank_to_dst_(integer4 *unit)
{  
  return dst_write_bank_(unit, &hmc1_blen, hmc1_bank );
}

integer4
hmc1_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = hmc1_common_to_bank_()) )
    {
      fprintf (stderr,"hmc1_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);         
    }             
    if ( (rcode = hmc1_bank_to_dst_(unit) ))
    {
      fprintf (stderr,"hmc1_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);         
    }
  return SUCCESS;
}

integer4
hmc1_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  hmc1_blen = 2 * sizeof(integer4); /* skip id and version  */

  if ( (rcode = dst_unpackr8_( hmc1_.tr_dir, (nobj=3, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( hmc1_.tr_rpvec, (nobj=3, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( &hmc1_.tr_rp, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpackr8_( hmc1_.site, (nobj=3, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( hmc1_.sh_rini, (nobj=3, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( hmc1_.sh_rfin, (nobj=3, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpackr8_( &hmc1_.energy, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( &hmc1_.sh_csmax, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( &hmc1_.fl_totpho, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpackr8_( &hmc1_.la_wavlen, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( &hmc1_.fl_twidth, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpackr8_( &hmc1_.sh_x0, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( &hmc1_.sh_xmax, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpackr8_( &hmc1_.sh_xfin, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpacki4_( &hmc1_.sh_iprim, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpacki4_( &hmc1_.setNr, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( &hmc1_.eventNr, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( &hmc1_.evttype, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( &hmc1_.iseed1, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( &hmc1_.iseed2, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpacki4_( &hmc1_.nmir, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( &hmc1_.ntube, (nobj=1, &nobj), bank, 
          &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  if ( (rcode = dst_unpacki4_( hmc1_.tubemir, (nobj=hmc1_.ntube, &nobj),
                 bank, &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( hmc1_.tube, (nobj=hmc1_.ntube, &nobj),
                 bank, &hmc1_blen, &hmc1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_( hmc1_.pe, (nobj=hmc1_.ntube, &nobj),
                 bank, &hmc1_blen, &hmc1_maxlen)) ) return rcode;

  return SUCCESS;
}

integer4
hmc1_common_to_dump_(integer4 *long_output)
{
  return hmc1_common_to_dumpf_(stdout, long_output);
}

integer4
hmc1_common_to_dumpf_(FILE* fp, integer4 *long_output) {

  integer4 i;

  fprintf(fp, "\nHMC1\n");
  fprintf(fp, "setNr/eventNr:  %d/%d\t",   hmc1_.setNr, hmc1_.eventNr);
  fprintf(fp, "triggered nmir/ntube:  %d/%d\n", hmc1_.nmir, hmc1_.ntube);

  if( hmc1_.evttype == TYPE_SHOWER) {
    fprintf(fp, "event type:  SHOWER\n");
    fprintf(fp, "    energy:  %3.2le eV\t", hmc1_.energy);
    if(hmc1_.sh_iprim == 0) fprintf(fp, "   primary:  proton\n");
    if(hmc1_.sh_iprim == 1) fprintf(fp, "   primary:  iron\n");
    fprintf(fp, "        Rp:  %lg m\t", hmc1_.tr_rp);
    fprintf(fp, "      xmax:  %lg gm/cm^2\t", hmc1_.sh_xmax);
    fprintf(fp, "     csmax:  %lg\n\n", hmc1_.sh_csmax);
  }
  else if( hmc1_.evttype == TYPE_LASER) { 
    fprintf(fp, "event type:  LASER\n");
    fprintf(fp, "      site:  %8lg  %8lg  %8lg\n", hmc1_.site[0],
              hmc1_.site[1], hmc1_.site[2]); 
    fprintf(fp, " direction:  %8lg  %8lg  %8lg\n", hmc1_.tr_dir[0],
              hmc1_.tr_dir[1], hmc1_.tr_dir[2]); 
    fprintf(fp, "        Rp:  %lg m\n\n", hmc1_.tr_rp);
  }
  else if( hmc1_.evttype == TYPE_FLASHER) { 
    fprintf(fp, "event type:  FLASHER\n");
    fprintf(fp, "      site:  %8lg  %8lg  %8lg\n", hmc1_.site[0],
              hmc1_.site[1], hmc1_.site[2]); 
    fprintf(fp, " direction:  %8lg  %8lg  %8lg\n", hmc1_.tr_dir[0],
              hmc1_.tr_dir[1], hmc1_.tr_dir[2]); 
    fprintf(fp, "        Rp:  %lg m\n\n", hmc1_.tr_rp);
  }

  if ( *long_output == 1 ) {

    if( hmc1_.evttype == TYPE_SHOWER) {
      fprintf(fp, "     tr_dir:  %8lg  %8lg  %8lg\n", hmc1_.tr_dir[0],
              hmc1_.tr_dir[1], hmc1_.tr_dir[2]); 
      fprintf(fp, "  Rp vector:  %8lg  %8lg  %8lg  meters\n",hmc1_.tr_rpvec[0],
              hmc1_.tr_rpvec[1], hmc1_.tr_rpvec[2]); 
      fprintf(fp, "rini vector:  %8lg  %8lg  %8lg  meters\n", hmc1_.sh_rini[0],
              hmc1_.sh_rini[1], hmc1_.sh_rini[2]); 
      fprintf(fp, "rfin vector:  %8lg  %8lg  %8lg  meters\n", hmc1_.sh_rfin[0],
              hmc1_.sh_rfin[1], hmc1_.sh_rfin[2]); 
      fprintf(fp, "         x0:  %lg gm/cm^2\n", hmc1_.sh_x0);
      fprintf(fp, "       xfin:  %lg gm/cm^2\n", hmc1_.sh_xfin);

    }
    else if( hmc1_.evttype == TYPE_LASER) { 
      fprintf(fp, "    energy:  %lg mJ\n", hmc1_.energy);
      fprintf(fp, "wavelength:  %lg mJ\n", hmc1_.la_wavlen);
      fprintf(fp, "  Rp vector:  %8lg  %8lg  %8lg  meters\n",hmc1_.tr_rpvec[0],
              hmc1_.tr_rpvec[1], hmc1_.tr_rpvec[2]); 
    }
    else if( hmc1_.evttype == TYPE_FLASHER) { 
      fprintf(fp, "    totpho:  %lg \n", hmc1_.fl_totpho);
      fprintf(fp, "pulseWidth:  %lg ns\n", hmc1_.fl_twidth);
      fprintf(fp, "  Rp vector:  %8lg  %8lg  %8lg  meters\n",hmc1_.tr_rpvec[0],
              hmc1_.tr_rpvec[1], hmc1_.tr_rpvec[2]); 
    }

    fprintf(fp, "seed1/seed2:  %d/%d\n\n",   hmc1_.iseed1, hmc1_.iseed2);

    for(i=0;i<hmc1_.ntube;i++) {
      fprintf(fp, "m:%3d  t:%4d  pe:%8d\n", hmc1_.tubemir[i],
                   hmc1_.tube[i], hmc1_.pe[i]);
    }
    fprintf(fp, "\n");



  }
  return SUCCESS;
}




