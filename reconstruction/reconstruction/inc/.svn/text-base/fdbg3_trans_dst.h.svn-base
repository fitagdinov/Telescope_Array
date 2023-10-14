#ifndef _FDBG3_TRANS_
#define _FDBG3_TRANS_

#define FDBG3_TRANS_BANKID 12401
#define FDBG3_TRANS_BANKVERSION 001

#define FDBG3_TRANS_MAXLAMBDA 1000

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdbg3_trans_bank_to_common_(integer1 *bank);
integer4 fdbg3_trans_common_to_dst_(integer4 *unit);
integer4 fdbg3_trans_common_to_bank_();
integer4 fdbg3_trans_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdbg3_trans_bank_buffer_ (integer4* fdbg3_trans_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdbg3_trans_t {

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

   /** median of measured BG3 transmittance */
   real4 transmittance[FDBG3_TRANS_MAXLAMBDA];
   /** lower deviation (median - 34%) */
   real4 transmittanceDevLower[FDBG3_TRANS_MAXLAMBDA];
   /** upper deviation (median + 34%) */
   real4 transmittanceDevUpper[FDBG3_TRANS_MAXLAMBDA];

} fdbg3_trans_dst_common;

extern fdbg3_trans_dst_common fdbg3_trans_;


#endif
