#ifndef _FDPMT_UNIFORMITY_
#define _FDPMT_UNIFORMITY_

#define FDPMT_UNIFORMITY_BANKID 12404
#define FDPMT_UNIFORMITY_BANKVERSION 001

#define FDPMT_UNIFORMITY_MAXDIVISION 200

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdpmt_uniformity_bank_to_common_(integer1 *bank);
integer4 fdpmt_uniformity_common_to_dst_(integer4 *unit);
integer4 fdpmt_uniformity_common_to_bank_();
integer4 fdpmt_uniformity_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdpmt_uniformity_bank_buffer_ (integer4* fdpmt_uniformity_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdpmt_uniformity_t {

   /** uniq ID */
   integer4 uniqID;

   /** number of division of X axis */
   integer4 xNDivision;
   /** minimum value of X axis */
   real4 xMinimum;
   /** maximum value of X axis */
   real4 xMaximum;

   /** number of division of Y axis */
   integer4 yNDivision;
   /** minimum value of Y axis */
   real4 yMinimum;
   /** maximum value of Y axis */
   real4 yMaximum;

   /** uniformity */
   real4 uniformity[FDPMT_UNIFORMITY_MAXDIVISION][FDPMT_UNIFORMITY_MAXDIVISION];
   /** standard deviation of uniformity */
   real4 uniformityDev[FDPMT_UNIFORMITY_MAXDIVISION][FDPMT_UNIFORMITY_MAXDIVISION];
   /** number of data in each bins*/
   integer4 nData[FDPMT_UNIFORMITY_MAXDIVISION][FDPMT_UNIFORMITY_MAXDIVISION];

} fdpmt_uniformity_dst_common;

extern fdpmt_uniformity_dst_common fdpmt_uniformity_;


#endif
