#ifndef _FDPARAGLAS_TRANS_
#define _FDPARAGLAS_TRANS_

#define FDPARAGLAS_TRANS_BANKID 12402
#define FDPARAGLAS_TRANS_BANKVERSION 001

#define FDPARAGLAS_TRANS_MAXLAMBDA 1000

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdparaglas_trans_bank_to_common_(integer1 *bank);
integer4 fdparaglas_trans_common_to_dst_(integer4 *unit);
integer4 fdparaglas_trans_common_to_bank_();
integer4 fdparaglas_trans_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdparaglas_trans_bank_buffer_ (integer4* fdparaglas_trans_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdparaglas_trans_t {

   /** uniq ID */
   integer4 uniqID;

   /** numbr of measured BG3 */
   integer4 nMeasuredData;

   /** measured number of lambda */
   integer4 nLambda;
   /** minimum lambda */
   real4 minLambda;
   /** delta lambda */
   real4 deltaLambda;

   /** median of measured transmittance */
   real4 transmittance[FDPARAGLAS_TRANS_MAXLAMBDA];
   /** standard deviation*/
   real4 transmittanceDev[FDPARAGLAS_TRANS_MAXLAMBDA];

} fdparaglas_trans_dst_common;

extern fdparaglas_trans_dst_common fdparaglas_trans_;


#endif
