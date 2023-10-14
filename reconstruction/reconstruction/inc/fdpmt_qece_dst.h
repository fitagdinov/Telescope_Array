#ifndef _FDPMT_QECE_
#define _FDPMT_QECE_

#define FDPMT_QECE_BANKID 12403
#define FDPMT_QECE_BANKVERSION 001

#define FDPMT_QECE_MAXLAMBDA 1000

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdpmt_qece_bank_to_common_(integer1 *bank);
integer4 fdpmt_qece_common_to_dst_(integer4 *unit);
integer4 fdpmt_qece_common_to_bank_();
integer4 fdpmt_qece_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdpmt_qece_bank_buffer_ (integer4* fdpmt_qece_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdpmt_qece_t {

   /** uniq ID */
   integer4 uniqID;

   /** number of measured PMT for CE*/
   integer4 ceNMeasuredData;
   /** collection eficiency */
   real4 ce;
   /** CE lower deviation */
   real4 ceDevLower;
   /** CE upper deviation */
   real4 ceDevUpper;

   /** number of measured PMT for QE */
   integer4 qeNMeasuredData;
   /** number of lambda */
   integer4 qeNLambda;
   /** minimum lambda */
   real4 qeMinLambda;
   /** delta lambda */
   real4 qeDeltaLambda;
   
   /** median of measured PMT quantum efficiency */
   real4 qe[FDPMT_QECE_MAXLAMBDA];
   /** lower deviation (median - 34%) */
   real4 qeDevLower[FDPMT_QECE_MAXLAMBDA];
   /** upper deviation (median + 34%) */
   real4 qeDevUpper[FDPMT_QECE_MAXLAMBDA];

} fdpmt_qece_dst_common;

extern fdpmt_qece_dst_common fdpmt_qece_;


#endif
