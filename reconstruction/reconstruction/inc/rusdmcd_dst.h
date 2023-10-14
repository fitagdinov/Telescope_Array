/*
 *     Temporary Bank for debugging sd mc
 *     Dmitri Ivanov (ivanov@physics.rutgers.edu)
 *     Apr 23, 2009
 *     Last modified: Apr 23, 2009
*/
#ifndef _RUSDMCD_
#define _RUSDMCD_

#define RUSDMCD_BANKID  13108
#define RUSDMCD_BANKVERSION   000


#define RUSDMCD_MSDS 0x100 /* max. number of sds/event */

#ifdef __cplusplus
extern "C" {
#endif
integer4 rusdmcd_common_to_bank_ ();
integer4 rusdmcd_bank_to_dst_ (integer4 * NumUnit);
integer4 rusdmcd_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 rusdmcd_bank_to_common_ (integer1 * bank);
integer4 rusdmcd_common_to_dump_ (integer4 * opt1);
integer4 rusdmcd_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* rusdmcd_bank_buffer_ (integer4* rusdmcd_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{  
  integer4 nsds;                  /* Number of SDs in the event */
  integer4 xxyy[RUSDMCD_MSDS];    /* Position ID */
  integer4 igsd[RUSDMCD_MSDS];    /* Is part of the shower ? 1 - yes, 0 - no */
  real8    edep[RUSDMCD_MSDS][2]; /* Energy deposited, MeV, [0] - lower, [1] - upper */
} rusdmcd_dst_common;

extern rusdmcd_dst_common rusdmcd_;

#endif
