#ifndef _FDMIRROR_REF_
#define _FDMIRROR_REF_

#define FDMIRROR_REF_BANKID 12400
#define FDMIRROR_REF_BANKVERSION 001

#define FDMIRROR_REF_MAXSITE 2
#define FDMIRROR_REF_MAXTELESCOPE 12
#define FDMIRROR_REF_MAXMIRROR 18
#define FDMIRROR_REF_MAXLAMBDA 1000

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdmirror_ref_bank_to_common_(integer1 *bank);
integer4 fdmirror_ref_common_to_dst_(integer4 *unit);
integer4 fdmirror_ref_common_to_bank_();
integer4 fdmirror_ref_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdmirror_ref_bank_buffer_ (integer4* fdmirror_ref_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdmirror_ref_data_t{
      /** 0:Good, 1 or more:Bad */
      integer4 badFlag;

      /** reflection rate */
      real4 reflection[FDMIRROR_REF_MAXLAMBDA];
      /** standard deviation of reflection distribution */
      real4 reflectionDev[FDMIRROR_REF_MAXLAMBDA];
} fdmirror_ref_data;

typedef struct _fdmirror_ref_t {

   /** uniq ID */
   integer4 uniqID;

   /** available date from */
   integer4 dateFrom; //sec from 1970/1/1
   /** available date to */
   integer4 dateTo; //sec from 1970/1/1

   /** Number of site */
   integer2 nSite; // 0 is BRM, 1 is LR
   /** Number of telescope in one site */
   integer2 nTelescope; // from 0 to 11 (@BRM,LR)
   /** Number of mirror in one telescope */
   integer2 nMirror; // from 1 to 18 (@BRM,LR)

   /** minimum value of prepared lambda */
   real4 minLambda;
   /** delta value of prepared lambda */
   real4 deltaLambda;
   /** number of prepared lambda */
   integer4 nLambda;

   fdmirror_ref_data mirrorData[FDMIRROR_REF_MAXSITE][FDMIRROR_REF_MAXTELESCOPE][FDMIRROR_REF_MAXMIRROR]; // e.g.) fMirrorData[site][tele][mirror]

} fdmirror_ref_dst_common;

extern fdmirror_ref_dst_common fdmirror_ref_;


#endif
