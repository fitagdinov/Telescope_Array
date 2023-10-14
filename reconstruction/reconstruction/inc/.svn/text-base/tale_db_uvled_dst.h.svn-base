#ifndef _TALE_DB_UVLED_
#define _TALE_DB_UVLED_

#define TALE_DB_UVLED_BANKID 12021
#define TALE_DB_UVLED_BANKVERSION 0 

#define TALE_DB_UVLED_NMIR 10
#define TALE_DB_UVLED_NCHN 256

#ifdef __cplusplus
extern "C" {
#endif
integer4 tale_db_uvled_common_to_bank_();
integer4 tale_db_uvled_bank_to_dst_(integer4 *NumUnit);
integer4 tale_db_uvled_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 tale_db_uvled_bank_to_common_(integer1 *bank);
integer4 tale_db_uvled_common_to_dump_(integer4 *long_output);
integer4 tale_db_uvled_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* tale_db_uvled_bank_buffer_ (integer4* tale_db_uvled_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{
  /* 2440000 subtracted from the julian day to give room for millisecond precision */
  /* checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5; jday=0 gives 1968 May 23.5 UT */
  integer4 jday;        /* mean julian day - 2.44e6 */
  integer4 idate;
  integer4 ipart;
  integer4 mirid;

  integer4 nshots    [TALE_DB_UVLED_NCHN]; // number of UVLED shots
  integer4 nsaturated[TALE_DB_UVLED_NCHN]; // number of UVLED shots which saturated tube
  integer4 nvalid    [TALE_DB_UVLED_NCHN]; // number of valid points
  real4    qdc_mean  [TALE_DB_UVLED_NCHN];
  real4    qdc_sdev  [TALE_DB_UVLED_NCHN];
  real4    npe_mean  [TALE_DB_UVLED_NCHN];
  real4    npe_sdev  [TALE_DB_UVLED_NCHN];
  real4    tube_gain [TALE_DB_UVLED_NCHN];
  real4    tube_cfqe [TALE_DB_UVLED_NCHN]; // QE correction factor

} tale_db_uvled_dst_common ;

extern tale_db_uvled_dst_common tale_db_uvled_ ; 

#endif
