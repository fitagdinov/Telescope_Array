/*
 * hspec_dst.h
 *
 * $Source: /hires_soft/cvsroot/bank/hspec_dst.h,v $
 * $Log: hspec_dst.h,v $
 * Revision 1.2  2002/04/10 22:28:07  ben
 * Added a new constant HSPEC_CREATION_NORMALIZE to describe creation of
 * HSPEC banks from a normalized spectrum and a given (measured) integrated
 * area.
 *
 * Revision 1.1  1999/05/25 15:46:50  ben
 * Initial revision
 *
 *
 * This bank contains emission/transmission curves for the components involved
 * with Roving Xenon Flasher and YAG calibration for Big-H.
 *
 */

#ifndef HSPEC_BANKID

#define HSPEC_BANKID 15018
#define HSPEC_BANKVERSION 0 
 
/* The number of lines and characters per line for the text description.. */

#define HSPEC_BINS 601     /* 601 bins from 200 to 800 nm inclusive.. */
#define HSPEC_DL    20
#define HSPEC_DC    80

/* Definitions for the different spectrum types that the bank handles.. */

#define HSPEC_CREATION_ARB              0
#define HSPEC_CREATION_ANALYSIS         1
#define HSPEC_CREATION_MEASURE          2
#define HSPEC_CREATION_APPROX           3
#define HSPEC_CREATION_NORMALIZE        4

#define HSPEC_SPECTYPE_ARB              0
#define HSPEC_SPECTYPE_RXF              1
#define HSPEC_SPECTYPE_YAG              2
#define HSPEC_SPECTYPE_RXFNARROW        3
#define HSPEC_SPECTYPE_RXFNARROWAPPROX  4
#define HSPEC_SPECTYPE_LED              5
#define HSPEC_SPECTYPE_FILTER           6
#define HSPEC_SPECTYPE_QE               7 

#ifdef __cplusplus
extern "C" {
#endif
integer4 hspec_common_to_bank_(void);
integer4 hspec_bank_to_dst_(integer4 *NumUnit);
integer4 hspec_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hspec_bank_to_common_(integer1 *bank);
integer4 hspec_common_to_dump_(integer4 *long_output);
integer4 hspec_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hspec_bank_buffer_ (integer4* hspec_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct 
{

  /* HiRes modified Julian day. This is the date that the bank was created.. */

  real8 jdate;

  /* Gregorian calendar date.. (year should be four digits) */

  integer4 year, month, day;

  /* 
     20, 80 character, lines for a textual description of the data.
     This MUST include the type of source/filter/qe curve being stored, as well
     as the assumptions made in creating the transmission/emission data and
     the method of creation (i.e. from processed and analyzed data, or from
     measurements with a photo-spectrometer, etc.. or from theoretical 
     approximations..)
  */

  integer1 description[HSPEC_DL][HSPEC_DC];

  /* 
     This is the acctual transmission/emission data in 1 nm. bins from 200 nm
     to 800 nm, inclusive.
  */

  real8 spectrum[HSPEC_BINS];

  /* Flag that describes the method of creation..

      0: Arbitrary creation
      1: Created from analyzing data and making assumptions
      2: Created from measurement data (photo-spectrometer, etc..)
      3: Created using some theoretical approximation
           .
           .
  */

  integer1 creation_flag;

  /* Spectrum type flag..

     0: Arbitrary type
     1: Roving Xenon Flasher with ordinary internal filter
     2: YAG laser
     3: RXF with a narrow band internal filter
     4: RXF with a narrow band internal filter approximated by a gaussian
     5: LED, blue, UV, etc..
     6: Filter (UV or whatever..)
     7: Quantum Efficiency
          .
	  .
  */

  integer1 spectrum_flag;

} hspec_dst_common ;

extern hspec_dst_common hspec_ ; 

#endif


