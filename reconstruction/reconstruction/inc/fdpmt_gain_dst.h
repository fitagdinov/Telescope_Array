#ifndef _FDPMT_GAIN_
#define _FDPMT_GAIN_

#define FDPMT_GAIN_BANKID 12405
#define FDPMT_GAIN_BANKVERSION 001

#define FDPMT_GAIN_MAXSITE 2
#define FDPMT_GAIN_MAXTELESCOPE 12
#define FDPMT_GAIN_MAXPMT 256

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdpmt_gain_bank_to_common_(integer1 *bank);
integer4 fdpmt_gain_common_to_dst_(integer4 *unit);
integer4 fdpmt_gain_common_to_bank_();
integer4 fdpmt_gain_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdpmt_gain_bank_buffer_ (integer4* fdpmt_gain_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdpmt_gain_data_t{
   /** bad flag(bit string of G0~G5)*/
   integer4 badFlag;
   /** gain*/
   real4 gain;
   /** standard error of gain */
   real4 gainError;
} fdpmt_gain_data;

typedef struct _fdpmt_gain_t {

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
   /** Number of pmt in one camera */
   integer2 nPmt; // from 1 to 18 (@BRM,LR)

   fdpmt_gain_data gainData[FDPMT_GAIN_MAXSITE][FDPMT_GAIN_MAXTELESCOPE][FDPMT_GAIN_MAXPMT]; // e.g.) gainData[site][tele][pmt]

} fdpmt_gain_dst_common;

extern fdpmt_gain_dst_common fdpmt_gain_;


#endif
