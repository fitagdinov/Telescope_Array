/*
 * prfd_dst.c 
 *
 * bank PRFD Contains profile information from the group profile
 * program, pfl. This is Dai's original pfgh fitting program
 * after substantial additions by MJK.
 *
 * The PRFD bank is strict superset of the PRFA bank and is
 * intended as its replacement. (N.B.: PRFB was already claimed
 * by jTang)
 *
 * All hires eyes are handled in one bank.
 *
 * It is expected that alternative profiling banks will be named
 * prfd, prfe, ...
 *
 * $Source: /hires_soft/cvsroot/bank/prfd_dst.c,v $
 * $Log: prfd_dst.c,v $
 * Revision 1.3  1997/02/19 08:48:23  mjk
 * Fixed a bug in prfd_common_to_dumpf_() which caused whether or
 * not bin information was dumped out to be dependent on pflinfo[]
 * rather than the proper bininfo[].
 *
 * Revision 1.2  1996/07/02  22:30:33  mjk
 * Added PRFD_INSANE_TRAJECTORY failmode message.
 *
 * Revision 1.1  1996/05/03  22:27:34  mjk
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
#include "prfd_dst.h"  

integer4 eventNameFromId(integer4 bank_id, integer1 *name, integer4 len);


prfd_dst_common prfd_;  /* allocate memory to prfd_common */

static integer4 prfd_blen = 0; 
static integer4 prfd_maxlen = sizeof(integer4) * 2 + sizeof(prfd_dst_common);
static integer1 *prfd_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* prfd_bank_buffer_ (integer4* prfd_bank_buffer_size)
{
  (*prfd_bank_buffer_size) = prfd_blen;
  return prfd_bank;
}



static void prfd_bank_init(void)
{
  prfd_bank = (integer1 *)calloc(prfd_maxlen, sizeof(integer1));
  if (prfd_bank==NULL)
    {
      fprintf(stderr, 
	      "prfd_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 prfd_common_to_bank_(void)
{	
  static integer4 id = PRFD_BANKID, ver = PRFD_BANKVERSION;
  integer4 rcode;
  integer4 i, nobj;
  integer4 nbin, pflinfo=0, bininfo=0, mtxinfo=0;


  if (prfd_bank == NULL) prfd_bank_init();

  /* Initialize prfd_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &prfd_blen, &prfd_maxlen, prfd_bank)))
    return rcode;


  /* Pack pflinfo[], bininfo[], and mtxinfo[] tightly */

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    pflinfo = pflinfo << 1;
    bininfo = bininfo << 1;
    mtxinfo = mtxinfo << 1;
    if ( prfd_.pflinfo[i] == PRFD_PFLINFO_USED )  pflinfo++;
    if ( prfd_.bininfo[i] == PRFD_BININFO_USED )  bininfo++;
    if ( prfd_.mtxinfo[i] == PRFD_MTXINFO_USED )  mtxinfo++;
  }
  
  if ((rcode = dst_packi4asi2_( &pflinfo, (nobj=1, &nobj), prfd_bank, 
				&prfd_blen, &prfd_maxlen))) return rcode; 
  if ((rcode = dst_packi4asi2_( &bininfo, (nobj=1, &nobj), prfd_bank, 
				&prfd_blen, &prfd_maxlen))) return rcode; 
  if ((rcode = dst_packi4asi2_( &mtxinfo, (nobj=1, &nobj), prfd_bank, 
				&prfd_blen, &prfd_maxlen))) return rcode; 


  /* First pack the profile parameters and their errors */

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.pflinfo[i] == PRFD_PFLINFO_UNUSED ) continue;

    if ((rcode = dst_packi4_(&prfd_.failmode[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if (prfd_.failmode[i] != SUCCESS) continue;
    
    
    /* We only pack the profile parameters if the fit worked */
    
    if ((rcode = dst_packr8_(&prfd_.szmx [i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.dszmx[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.rszmx[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.lszmx[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.tszmx[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_packr8_(&prfd_.xm [i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.dxm[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.rxm[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.lxm[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.txm[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_packr8_(&prfd_.x0 [i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.dx0[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.rx0[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.lx0[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.tx0[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_packr8_(&prfd_.lambda [i], (nobj=1, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.dlambda[i], (nobj=1, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.rlambda[i], (nobj=1, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.llambda[i], (nobj=1, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.tlambda[i], (nobj=1, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_packr8_(&prfd_.eng [i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.deng[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.reng[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.leng[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.teng[i], (nobj=1, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    /* Pack remaining stuff */
    if ((rcode = dst_packi4_(&prfd_.traj_source[i], (nobj=1,&nobj), prfd_bank,
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packi4_(&prfd_.errstat[i], (nobj=1, &nobj), prfd_bank,
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packi4_(&prfd_.ndf[i], (nobj=1, &nobj), prfd_bank,
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(&prfd_.chi2[i], (nobj=1, &nobj), prfd_bank,
			     &prfd_blen, &prfd_maxlen))) return rcode; 
  }


  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.bininfo[i] == PRFD_BININFO_UNUSED ) continue;

    /* Pack number of bins - check that nbin[] are within range */
    
    if ( prfd_.nbin[i] < 0 )  
      nbin = 0;
    else if ( prfd_.nbin[i] > PRFD_MAXBIN ) 
      nbin = PRFD_MAXBIN;
    else 
      nbin = prfd_.nbin[i];
    
    if ( prfd_.nbin[i] < 0 || prfd_.nbin[i] > PRFD_MAXBIN ) {
      fprintf(stderr,
	      "%s Number of bins out of range (%d) for eye %d; only packing (%d)\n", 
	      "prfd_common_to_bank_:", prfd_.nbin[i], i+1, nbin);
    }
    if ((rcode = dst_packi4asi2_(&nbin, (nobj=1, &nobj), prfd_bank, 
				 &prfd_blen, &prfd_maxlen))) return rcode; 
    
    
    /* Pack information about bins along the show trajectory */
    
    if ((rcode = dst_packr8_(prfd_.dep[i], (nobj=nbin, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(prfd_.gm[i], (nobj=nbin, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    
    
    /* Pack information about light and MC match along show trajectory */
    
    if ((rcode = dst_packr8_(prfd_.scin[i], (nobj=nbin, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(prfd_.rayl[i], (nobj=nbin, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(prfd_.aero[i], (nobj=nbin, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(prfd_.crnk[i], (nobj=nbin, &nobj), prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(prfd_.sigmc[i], (nobj=nbin, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_(prfd_.sig[i], (nobj=nbin, &nobj),prfd_bank, 
			     &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packi4asi2_(prfd_.ig[i], (nobj=nbin, &nobj),prfd_bank,
				 &prfd_blen, &prfd_maxlen))) return rcode; 
  }
  
  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.mtxinfo[i] == PRFD_MTXINFO_UNUSED ) continue;
    
    if ((rcode = dst_packi4asi2_( &prfd_.nel[i], (nobj=1, &nobj), prfd_bank, 
				  &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packi4asi2_( &prfd_.mor[i], (nobj=1, &nobj), prfd_bank, 
				  &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_packr8_( prfd_.mxel[i], (nobj=prfd_.nel[i], &nobj),
			      prfd_bank, &prfd_blen, &prfd_maxlen))) return rcode;
  }

  return SUCCESS;
}


integer4 prfd_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &prfd_blen, prfd_bank );
}

integer4 prfd_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = prfd_common_to_bank_()))
    {
      fprintf (stderr,"prfd_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = prfd_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"prfd_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 prfd_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 i, nobj;
  integer4 nbin, pflinfo, bininfo, mtxinfo;

  
  prfd_blen = 2 * sizeof(integer4); /* skip id and version  */


  /* Unpack pflinfo[], bininfo[], and mtxinfo[] */

  if ((rcode = dst_unpacki2asi4_(&pflinfo, (nobj=1, &nobj), bank, 
				 &prfd_blen, &prfd_maxlen))) return rcode; 
  if ((rcode = dst_unpacki2asi4_(&bininfo, (nobj=1, &nobj), bank,
				 &prfd_blen, &prfd_maxlen))) return rcode; 
  if ((rcode = dst_unpacki2asi4_(&mtxinfo, (nobj=1, &nobj), bank,
				 &prfd_blen, &prfd_maxlen))) return rcode; 

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( pflinfo & 0x8000 )  prfd_.pflinfo[i] = PRFD_PFLINFO_USED;
    else                     prfd_.pflinfo[i] = PRFD_PFLINFO_UNUSED;
    if ( bininfo & 0x8000 )  prfd_.bininfo[i] = PRFD_BININFO_USED;
    else                     prfd_.bininfo[i] = PRFD_BININFO_UNUSED;
    if ( mtxinfo & 0x8000 )  prfd_.mtxinfo[i] = PRFD_MTXINFO_USED;
    else                     prfd_.mtxinfo[i] = PRFD_MTXINFO_UNUSED;
    pflinfo = pflinfo << 1;
    bininfo = bininfo << 1;
    mtxinfo = mtxinfo << 1;
  }


  /* First unpack the profile parameters and their errors */

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.pflinfo[i] == PRFD_PFLINFO_UNUSED ) continue;
      
    if ((rcode = dst_unpacki4_(&prfd_.failmode[i], (nobj=1,&nobj), bank,
			       &prfd_blen, &prfd_maxlen))) return rcode; 

    if (prfd_.failmode[i] != SUCCESS) continue;
    
    
    /* We only unpack the profile parameters if the fit worked */
    
    if ((rcode = dst_unpackr8_(&prfd_.szmx [i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.dszmx[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.rszmx[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.lszmx[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.tszmx[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_unpackr8_(&prfd_.xm [i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.dxm[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.rxm[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.lxm[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.txm[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_unpackr8_(&prfd_.x0 [i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.dx0[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.rx0[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.lx0[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.tx0[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_unpackr8_(&prfd_.lambda [i], (nobj=1, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.dlambda[i], (nobj=1, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.rlambda[i], (nobj=1, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.llambda[i], (nobj=1, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.tlambda[i], (nobj=1, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    
    if ((rcode = dst_unpackr8_(&prfd_.eng [i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.deng[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.reng[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.leng[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.teng[i], (nobj=1, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    
    /* Unpack remaining stuff */
    if ((rcode = dst_unpacki4_(&prfd_.traj_source[i], (nobj=1,&nobj), bank,
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpacki4_(&prfd_.errstat[i], (nobj=1, &nobj), bank,
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpacki4_(&prfd_.ndf[i], (nobj=1, &nobj), bank,
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(&prfd_.chi2[i], (nobj=1, &nobj), bank,
			       &prfd_blen, &prfd_maxlen))) return rcode; 
  }
  

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.bininfo[i] == PRFD_BININFO_UNUSED ) continue;

    /* Unpack number of bins */
    
    if ((rcode = dst_unpacki2asi4_(&nbin, (nobj=1, &nobj), bank, 
				   &prfd_blen, &prfd_maxlen))) return rcode; 
    prfd_.nbin[i] = nbin;
    

    /* Unpack information about bins along the shower trajectory */
    if ((rcode = dst_unpackr8_(prfd_.dep[i], (nobj=nbin, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(prfd_.gm[i], (nobj=nbin, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    
    
    /* Unpack information about light and MC match along shower trajectory */
    
    if ((rcode = dst_unpackr8_(prfd_.scin[i], (nobj=nbin, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(prfd_.rayl[i], (nobj=nbin, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(prfd_.aero[i], (nobj=nbin, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(prfd_.crnk[i], (nobj=nbin, &nobj), bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(prfd_.sigmc[i], (nobj=nbin, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpackr8_(prfd_.sig[i], (nobj=nbin, &nobj),bank, 
			       &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpacki2asi4_(prfd_.ig[i], (nobj=nbin, &nobj),bank, 
				   &prfd_blen, &prfd_maxlen))) return rcode; 
  }

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.mtxinfo[i] == PRFD_MTXINFO_UNUSED ) continue;
    
    if ((rcode = dst_unpacki2asi4_( &prfd_.nel[i], (nobj=1, &nobj), bank, 
				    &prfd_blen, &prfd_maxlen))) return rcode; 
    if ((rcode = dst_unpacki2asi4_( &prfd_.mor[i], (nobj=1, &nobj), bank, 
				    &prfd_blen, &prfd_maxlen))) return rcode; 
    if ( prfd_.nel[i] > PRFD_MAXMEL ) {
      fprintf(stderr, "%s: Error Matrix for fit %d has more elements (%d) than in the\n",
	      __FILE__, i+1, prfd_.nel[i] );
      fprintf(stderr, "  %d reserved in the structure. Only unpacking %d elements\n",
	      PRFD_MAXMEL, PRFD_MAXMEL );
      prfd_.nel[i] = PRFD_MAXMEL;
    }
    if ((rcode = dst_unpackr8_( prfd_.mxel[i], (nobj=prfd_.nel[i], &nobj),
				bank, &prfd_blen, &prfd_maxlen))) return rcode;
  }

  return SUCCESS;
}

integer4 prfd_common_to_dump_(integer4 *long_output)
{
  return prfd_common_to_dumpf_(stdout, long_output);
}

integer4 prfd_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 mark_header = 0;
  integer4 i, j, k, idx;
  integer1 trajName[16];

  static struct {
    integer4  code;
    integer1 *mess;
  } trans1[] = {
    { PRFD_IG_CHERENKOV_CUT, "cherenkov cut"         },
    { PRFD_IG_SICKPLNFIT,    "sick plane fit"        },
    { PRFD_IG_OVERCORRECTED, "too much corr"         },
    { PRFD_IG_GOODBIN,       ""                      }, /* good bin */
    { 999,                   "unknown ig code"       }  /* marks end of list */ 
  };

  static struct {
    integer4  code;
    integer1 *mess;
  } trans2[] = {
    { PRFD_FIT_NOT_REQUESTED,       "Fit not requested"                   },
    { PRFD_NOT_IMPLEMENTED,         "Fit not implemented"                 },
    { PRFD_REQUIRED_BANKS_MISSING,  
      "Bank(s) required for fit are missing or have failed"               },
    { PRFD_MISSING_TRAJECTORY_INFO,
      "Bank(s) required for desired trajectory source are missing/failed" },
    { PRFD_UPWARD_GOING_TRACK,      "Upward going track"                  },
    { PRFD_TOO_FEW_GOOD_BINS,       "Too few good bins"                   },
    { PRFD_FITTER_FAILURE,          "Fitter failed"                       },	
    { PRFD_INSANE_TRAJECTORY,
      "Trajectory (direction and/or core) unreasonable"                   },
    { 999,                          "Unknown failmode"                    } /* marks end of list */ 
  };


  fprintf(fp,"\nPRFD bank. bins: ");
  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    if ( prfd_.bininfo[i] == PRFD_BININFO_USED )
      fprintf(fp," %03d", prfd_.nbin[i]);
    else
      fprintf(fp," -- ");
  }
  fprintf(fp,"\n\n");

  
  /* Show geometry results */

  for ( i=0; i<PRFD_MAXFIT; i++ ) {
    
    if ( prfd_.pflinfo[i] == PRFD_PFLINFO_UNUSED ) continue;

    fprintf(fp,"    -> Profile Fit %1d\n", i+1);
    
    if ( prfd_.failmode[i] != SUCCESS ) {
      for ( k=0; trans2[k].code!=999 ; k++ )
	if ( prfd_.failmode[i] == trans2[k].code ) break;

      fprintf(fp,"    %s\n", trans2[k].mess );
      continue;    /* Nothing else to show for this fit */
    }
    
    if ( !mark_header ) {
      fprintf(fp,"\n            value   stat error        right       left     geom error\n");
      mark_header = 1;
    }

    fprintf(fp,"  Szmx: %9.3e +- %9.3e   (%9.3e, %9.3e)  +- %9.3e  particles\n",
	    prfd_.szmx[i], prfd_.dszmx[i], prfd_.rszmx[i], prfd_.lszmx[i],
	    prfd_.tszmx[i] );
    
    fprintf(fp,"  Xmax: %9.2f +- %9.2f   (%9.2f, %9.2f)  +- %9.2f  g/cm^2\n",
	    prfd_.xm[i],  prfd_.dxm[i],  prfd_.rxm[i],  prfd_.lxm[i],
	    prfd_.txm[i] );
    
    fprintf(fp,"    X0: %9.2f +- %9.2f   (%9.2f, %9.2f)  +- %9.2f  g/cm^2\n",
	    prfd_.x0[i],  prfd_.dx0[i],  prfd_.rx0[i],  prfd_.lx0[i],
	    prfd_.tx0[i] );
    
    fprintf(fp,"  Lamb: %9.2f +- %9.2f   (%9.2f, %9.2f)  +- %9.2f  g/cm^2\n",
	    prfd_.lambda[i],  prfd_.dlambda[i], prfd_.rlambda[i], 
	    prfd_.llambda[i], prfd_.tlambda[i] );
    
    fprintf(fp,"  Engy: %9.3f +- %9.3f   (%9.3f, %9.3f)  +- %9.3f  EeV\n",
	    prfd_.eng[i],  prfd_.deng[i],  prfd_.reng[i],  prfd_.leng[i],
	    prfd_.teng[i] );
    
    fprintf(fp,"\n");
    fprintf(fp," chi2/ndf: %7.3f / %3d\n", prfd_.chi2[i], prfd_.ndf[i] );
    if ( eventNameFromId( prfd_.traj_source[i], trajName, sizeof(trajName))
	 == 0 ) strcpy(trajName,"Unknown Bank");
    fprintf(fp," trajectory source: %5d (%s)    errstat: %d\n", 
	    prfd_.traj_source[i], trajName, prfd_.errstat[i] );
    if ( prfd_.errstat[i] != SUCCESS ) {
      if ( prfd_.errstat[i] & PRFD_STAT_ERROR_FAILURE )
	fprintf(fp,"   STATISTICAL errors failed\n");
      if ( prfd_.errstat[i] & PRFD_RIGHT_ERROR_FAILURE )
	fprintf(fp,"   RIGHT TRAJECTORY errors failed\n");
      if ( prfd_.errstat[i] & PRFD_LEFT_ERROR_FAILURE )
	fprintf(fp,"   LEFT TRAJECTORY errors failed\n");
      if ( prfd_.errstat[i] & PRFD_GEOM_ERROR_FAILURE )
	fprintf(fp,"   GEOMETRICAL errors failed\n");
      if ( prfd_.errstat[i] & PRFD_GEOM_ERROR_INCOMPLETE )
	fprintf(fp,"   GEOMETRICAL errors incomplete\n");
    }
    fprintf(fp,"\n");
  }


  /* If long output is selected show the light contributions from
     each light source for each profile */

  if ( (*long_output)==1 ) {

    for (i=0; i<PRFD_MAXFIT; i++) {
    
      if ( prfd_.bininfo[i] == PRFD_BININFO_UNUSED ) continue;

      fprintf(fp,"    -> Profile Bins %1d\n", i+1);
      fprintf(fp,"    slant     scin     rayl    aero     crnk   mc_tot  signal  ig\n");
      
      for (j=0; j<prfd_.nbin[i]; j++) {
        for ( k=0; trans1[k].code!=999 ; k++ )
	  if ( prfd_.ig[i][j] == trans1[k].code ) break;
	
	fprintf(fp,"  %8.2f  %7.3f %7.3f %7.3f %9.4f %7.2f %7.2f  %s\n", 
		prfd_.dep[i][j],  prfd_.scin[i][j], prfd_.rayl[i][j], 
		prfd_.aero[i][j], prfd_.crnk[i][j], prfd_.sigmc[i][j],
		prfd_.sig[i][j],  trans1[k].mess );
      }
      fprintf(fp,"\n");
    }
    
    for (i=0; i<PRFD_MAXFIT; i++) {

      if ( prfd_.mtxinfo[i] == PRFD_MTXINFO_UNUSED ) continue;

      fprintf(fp,"    -> Error matrix %1d\n", i+1);
      for ( j=0; j<prfd_.mor[i]; j++) {
	if ( j == 0 ) fprintf(fp,"  / ");
	else if ( j == prfd_.mor[i]-1 ) fprintf(fp,"  \\ ");
	else fprintf(fp,"  | ");
	for ( k=0; k<prfd_.mor[i]; k++) {
	  if ( k >= j )
	    idx = prfd_.nel[i] - ((prfd_.mor[i]-j) * (prfd_.mor[i]-j + 1))/2
	      + ( k - j);
	  else
	    idx = prfd_.nel[i] - ((prfd_.mor[i]-k) * (prfd_.mor[i]-k + 1))/2
	      + ( j - k);
	  
	  fprintf(fp,"%13.6lg", prfd_.mxel[i][idx]);
	}
	if ( j == 0 ) fprintf(fp,"  \\\n");
	else if ( j == prfd_.mor[i]-1 ) fprintf(fp,"  /\n");
	else fprintf(fp,"  |\n");
      }
      fprintf(fp,"\n");
    }
  }
  
  fprintf(fp,"\n");

  return SUCCESS;
}
