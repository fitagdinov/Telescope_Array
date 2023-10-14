/*
 * hnpe_dst.h
 *
 * $Source: /hires_soft/cvsroot/bank/hnpe_dst.h,v $
 * $Log: hnpe_dst.h,v $
 * Revision 1.3  2000/03/21 21:15:53  ben
 * Minor change in dstdump format.
 *
 * Revision 1.1  1999/06/01  20:46:51  ben
 * Initial revision
 *
 *
 * This is a calibration bank for Big-H. It is mainly intended to store the 
 * results of analysing Roving Xenon Flasher and YAG Laser calibration data.
 * It is generalizable to any other type of calibration light source however,
 * and it is also is extendable to include electronics calibration as well.
 *
 */
#ifndef HNPE_BANKID

#define HNPE_BANKID      15019
#define HNPE_BANKVERSION     0 

#define HNPE_MAX_SRC        20     /* Max. number of sources.. */
#define HNPE_FNC           256     /* Max. number of filename characters.. */
#define HNPE_DC            256     /* Max. number of descript. characters.. */
#define HNPE_NULL_TIME     0.0
#define HNPE_BAD_VALUE      -100000
#define HNPE_BAD_VALUE_TEST -99999

#define HNPE_QE_337      0.265     /* Ave. QE at 337 nm. for philips PMTs */

#define HNPE_BIT(x)      ((unsigned char)1 << (x))

#ifdef __cplusplus
extern "C" {
#endif
integer4 hnpe_common_to_bank_(void);
integer4 hnpe_bank_to_dst_(integer4 *NumUnit);
integer4 hnpe_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hnpe_bank_to_common_(integer1 *bank);
integer4 hnpe_common_to_dump_(integer4 *long_output);
integer4 hnpe_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hnpe_bank_buffer_ (integer4* hnpe_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct 
{
  
  /* Which mirror the data in this bank applies to.. */

  integer4 mirror;

  /* 
     The Julian dates through which this calibration is valid.. 
     Generally, start_date should be the date on which the data used in 
     producing this bank was taken, and the end_date can be set to some 
     predetermined future date or just left at the HNPE_NULL_TIME value, 0.0.
  */

  real8 start_date;
  real8 end_date;

  /* 
     These values modify the given HiRes UV filter curve for every tube.
     They are applied as an exponent to the curve being used.
  */
  
  real8 uv_exp[HR_UNIV_MIRTUBE];

  /* 
     This is the data file name of the UV filter curve that applies to this
     mirror and date. The file specified should exist in the standard directory
     for HSPEC sources/filters/quantum efficiency curves:
     /hires_soft/dst95/const/hspec
  */

  integer1 uv_file_name[HNPE_FNC];

  /* 
     These values are the acctual quantum efficiencies for each tube at 337 nm.
     The values are to be used in scaling a standard quantum curve.
  */

  real8 mean_qe_337[HR_UNIV_MIRTUBE];
  real8 sigma_qe_337[HR_UNIV_MIRTUBE];
  real8 qe_337[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];

  /* 
     This is the name of the DST file that contains a quantum efficiency
     spectrum --HSPEC bank-- that applies to this mirror and date. This file
     also should exist in the standard directory for sources/filters/quantum 
     efficiency curves..
  */

  integer1 qe_file_name[HNPE_FNC];

  /* This is the number of sources of data used to make this calibration 
     bank.. Should be between 1 and HNPE_MAX_SRC */

  integer4 number_src;

  /* These are the one-line, HNPE_DC number of characters, descriptions for 
     each source */
  
  integer1 source_desc[HNPE_MAX_SRC][HNPE_DC];

  /* 
     These are the DST file names for the HSPEC banks that represent each 
     source. These files should all exist in the standard area for HSPEC bank
     sources: /hires_soft/dst95/const/hspec
  */

  integer1 source_file_name[HNPE_MAX_SRC][HNPE_FNC];

  /* Mean and sigma QDCB values for each source and each tube in the mirror. */
  
  real8 mean_qdcb[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];
  real8 sigma_qdcb[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];

  /* 
     Flag for possible tube saturation based on QDCB values,
     or Chi2 failing for a giving source and tube.

     Saturation:
     3,700 QDCB counts and above for REV3 mirrors and
     30,000 QDCB counts and above for REV4 mirrors.

     Data from tubes that are registering in the non-linear range should NOT
     be trusted for estimating photo-electrons. This is the case since the 
     standard deviation of the QDCB distribution in the high, non-linearl, 
     QDCB range is smaller than normal. Since photo-electron estimation
     depends upon dividing by the standard deviation of the QDCB distribution,
     non-linearly responding tubes will show an abnormally high number of 
     photo-electrons.

     Chi2/n: For gaussian fit to a tube's QDC distribution.
     cut is from CHI2_LOW_CUT to CHI2_HIGH_CUT.

  */
			
  integer4 valid_src_flag[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];

  /* 
     Mean and sigma N.P.E and pulse area for each source and each tube 
     These values should NOT be trusted if the saturate_flag for the given
     source and tube is 1.
  */

  real8 mean_area[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];
  real8 sigma_area[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];
  
  real8 mean_npe[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];
  real8 sigma_npe[HNPE_MAX_SRC][HR_UNIV_MIRTUBE];

  /* 
     first order tube gains are those caluclated by doing fits of QDCB counts
     verses number of photo-electrons. This calculation does not include
     the electronic calibration that the second order tube gains include.
     The tube gains relate QDCB counts to photo-electrons. 
 
     QDCB[i][t] * first_order_gain[i][t] = NPE[i][t]

     first order tube flag is an indication if the given tube had problems in 
     the fitting process. 

     BITS set in the gain flag mean the following:

     0: Tube has shown pedestal triggers.
     1: Tube is saturating for some source.
     2: Tube failed gaussian fit X^2/n cut.
     3: Tube has low gain compared with rest of cluster (3 sigma cut).
     4: Tube has high "      "      "    "   "     "          "
     5: Tube had a bad gain fit (Arbitrary gain sdev cut).
     6: 
     7:

     first_order_fit_goodness[] contains a "goodness of fit" estimate for the
     QDCB vs. NPE fit. This is currently a standard deviation.
  */

  real8 first_order_gain[HR_UNIV_MIRTUBE];
  unsigned char first_order_gain_flag[HR_UNIV_MIRTUBE];
  real8 first_order_fit_goodness[HR_UNIV_MIRTUBE];

  /* 
     Second order tube gains include Jeremy Smith's magic electronic 
     calibration. These gains relate a nanovolt-second pulse area (calulated by
     feeding a QDCB count and an estimated pulse width to Jeremy's magic 
     function and getting out a number of photo-electrons).

     nVs(QDCB[i][t], width) * second_order_gain[i][t] = NPE[i][t]

     The second order gain flag is also an indication of the gain's validity, 
     but based on the fit of nVs area versus QDCB counts. The second order fit
     goodness is also the same as the first order, but based on the 
     nVs(QDCB, w) vs. NPE fit. Bits 6 and 7 are used in the 2nd order gain
     flag:

     6: Area was not calculated on some shot/source combo.
     7: Area was returned "suspect" on some shot/source combo.
  */

  real8 second_order_gain[HR_UNIV_MIRTUBE];
  unsigned char second_order_gain_flag[HR_UNIV_MIRTUBE];
  integer4 ecalib_flag[HR_UNIV_MIRTUBE];
  real8 second_order_fit_goodness[HR_UNIV_MIRTUBE];


} hnpe_dst_common ;

extern hnpe_dst_common hnpe_ ; 

#endif








