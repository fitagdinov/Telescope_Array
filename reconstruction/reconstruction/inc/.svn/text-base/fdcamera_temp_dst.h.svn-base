#ifndef _FDCAMERA_TEMP_
#define _FDCAMERA_TEMP_

#define FDCAMERA_TEMP_BANKID 12406
#define FDCAMERA_TEMP_BANKVERSION 001

#define FDCAMERA_TEMP_MAXSITE 2
#define FDCAMERA_TEMP_MAXTELESCOPE 12

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdcamera_temp_bank_to_common_(integer1 *bank);
integer4 fdcamera_temp_common_to_dst_(integer4 *unit);
integer4 fdcamera_temp_common_to_bank_();
integer4 fdcamera_temp_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdcamera_temp_bank_buffer_ (integer4* fdcamera_temp_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct _fdcamera_temp_t {

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

   /** bad flag */
  integer4 badFlag[FDCAMERA_TEMP_MAXSITE][FDCAMERA_TEMP_MAXTELESCOPE];
   /** temperature */
   real4 temp[FDCAMERA_TEMP_MAXSITE][FDCAMERA_TEMP_MAXTELESCOPE];

} fdcamera_temp_dst_common;

extern fdcamera_temp_dst_common fdcamera_temp_;


#endif
