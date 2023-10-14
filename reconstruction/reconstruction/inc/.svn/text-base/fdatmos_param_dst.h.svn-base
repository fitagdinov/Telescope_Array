#ifndef _FDATMOS_PARAM_
#define _FDATMOS_PARAM_

#define FDATMOS_PARAM_BANKID 12409
#define FDATMOS_PARAM_BANKVERSION 001

#define FDATMOS_PARAM_MAXITEM 500

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdatmos_param_bank_to_common_(integer1 *bank);
integer4 fdatmos_param_common_to_dst_(integer4 *unit);
integer4 fdatmos_param_common_to_bank_();
integer4 fdatmos_param_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdatmos_param_bank_buffer_ (integer4* fdatmos_param_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdatmos_param_t {

   /** uniq ID */
   integer4 uniqID;

   /** available date from */
   integer4 dateFrom; //sec from 1970/1/1
   /** available date to */
   integer4 dateTo; //sec from 1970/1/1

   /** number of data line */
   integer4 nItem;

   /** height [km] */
   real4 height[FDATMOS_PARAM_MAXITEM];
   /** pressure [hPa] */
   real4 pressure[FDATMOS_PARAM_MAXITEM];
   /** pressure error [hPa] */
   real4 pressureError[FDATMOS_PARAM_MAXITEM];
   /** temperature [degree] */
   real4 temperature[FDATMOS_PARAM_MAXITEM];
   /** temperature error [degree] */
   real4 temperatureError[FDATMOS_PARAM_MAXITEM];
   /** dew point [degree] */
   real4 dewPoint[FDATMOS_PARAM_MAXITEM];
   /** dew point error [degree] */
   real4 dewPointError[FDATMOS_PARAM_MAXITEM];


} fdatmos_param_dst_common;

extern fdatmos_param_dst_common fdatmos_param_;


#endif
