/*  hmc1_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/hmc1_dst.h,v $
 * $Log: hmc1_dst.h,v $
 * Revision 1.2  1998/04/21 22:13:17  ben
 * Included struct def. and block def. inside #if block to avoid redeclaration
 *
 * Revision 1.1  1997/10/04  22:37:03  tareq
 * Initial revision
 *
 *
 * output of mc97 event simulation
 *
 */


#ifndef _HMC1_
#define _HMC1_

#define  HMC1_BANKID 15020
#define  HMC1_BANKVERSION 0 

/* define event types */

#define TYPE_SHOWER 1
#define TYPE_LASER 2
#define TYPE_FLASHER 3

/***********************************************/

#ifdef __cplusplus
extern "C" {
#endif
integer4  hmc1_common_to_bank_(void);
integer4  hmc1_bank_to_dst_(integer4 *NumUnit);
integer4  hmc1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4  hmc1_bank_to_common_(integer1 *bank);
integer4  hmc1_common_to_dump_(integer4 *long_output);
integer4  hmc1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hmc1_bank_buffer_ (integer4* hmc1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/***********************************************/
/* Define common block structures    */

typedef struct {

  /* define geometry parameters */

  real8     tr_dir[3];    /* track direction unit vector */
  real8     tr_rpvec[3];  /* Rp vector to track (meters) */
  real8     tr_rp;        /* magnitude of tr_rpvec */

  real8     site[3];      /* vector to laser or flasher site (meters) */
  real8     sh_rini[3];   /* vector to first track segment (meters) */
  real8     sh_rfin[3];   /* vector to last track segment (*/

  /* define energy and other parameters */

  real8     energy;       /* shower energy (eV) or laser energy (mJ) */
  real8     sh_csmax;     /* shower size at shower max. */
  real8     fl_totpho;    /* total number of photons */

  real8     la_wavlen;    /* laser wave length (nm) */
  real8     fl_twidth;    /* flasher pulse width (ns) */

  real8     sh_x0;        /* depth of first interaction (gm/cm^2) */
  real8     sh_xmax;      /* depth of shower max. from x0 (gm/cm^2) */
  real8     sh_xfin;      /* depth of final shower segment (gm/cm^2) */

  integer4  sh_iprim;     /* primary particle: =0 protons, else iron */

  /* general description of event */

  integer4  setNr;        /* identifies the set of events this event
                             belongs to in the form YYMMDDPP */
  integer4  eventNr;      /* the event number in that set */
  integer4  evttype;
  integer4  iseed1;       /* iseed before event */
  integer4  iseed2;       /* iseed after event */

  integer4  nmir;         /* number of mirrors in event */
  integer4  ntube;        /* total number of tubes in event */

  integer4  tubemir[HR_UNIV_MAXTUBE];  /* mirror id */
  integer4  tube[HR_UNIV_MAXTUBE];     /* tube id */
  integer4  pe[HR_UNIV_MAXTUBE];       /* pe's received by tube */


}  hmc1_dst_common ;

extern  hmc1_dst_common  hmc1_ ; 

#endif

