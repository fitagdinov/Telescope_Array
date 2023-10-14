/*
 * declaration of return codes for DST95 routines
 * $Source: /hires_soft/uvm2k/dst/dst_err_codes.h,v $
 * $Log: dst_err_codes.h,v $
 * Revision 1.4  2001/10/09 15:12:26  hires
 * Add some error codes (boyer)
 *
 * Revision 1.3  1995/04/16 21:28:38  jui
 * added definition for MALLOC_ERR (-1111) which indicates that
 * malloc had failed to reserve necessary buffer space
 *
 * Revision 1.2  1995/03/18  00:35:13  jeremy
 * *** empty log message ***
 *
 * created:  CCJ  17-JAN-1995
 *           ultrix version only...will modify as needed to accomdate
 *           other platforms
 */

#define SUCCESS 0
#define END_OF_FILE -1

#define MAXLEN_EXCEEDED -2

#define TO_DST_FAIL -101
#define R4_NTOI_FAIL -102
#define R8_NTOI_FAIL -103

#define FROM_DST_FAIL -201
#define R4_ITON_FAIL -202
#define R8_ITON_FAIL -203

#define PACKI4_FAIL -300
#define UNPACKI4_FAIL -400

#define UNIT_OUT_OF_RANGE -1000
#define UNIT_USED_FOR_IN -1001
#define UNIT_USED_FOR_OUT -1002
#define UNIT_NOT_OPEN -1003

#define UNDEFINED_MODE -1010
#define MALLOC_ERR -1111

#define FOPEN_ERR -1200
#define SETVBUF_ERR -1201
#define FCLOSE_ERR -1210
#define FWRITE_ERR -1220

#define END_BLOCK_ERR -2100
#define OUT_BLOCK_ERR -2200
#define NEW_BLOCK_ERR -2300
#define GET_BLOCK_ERR -2400

#define GET_BLOCK_BYTE_CNT    -2500
#define GET_BLOCK_CRC_ERROR   -2501
#define GET_BLOCK_CORRUPT_HDR -2502
#define GET_BLOCK_OUT_OF_SEQ  -2503
#define GET_BLOCK_READ_ERROR  -2504

#define GET_BANK_UNEXP_CONT    -2600
#define GET_BANK_UNEXP_END_BNK -2601
#define GET_BANK_UNEXP_TBCNT   -2602
#define GET_BANK_UNKWN_OPCODE  -2603
#define GET_BANK_LENGTH_ERROR  -2604
#define GET_BANK_UNEXP_END_BLK -2605
#define GET_BANK_OPCODE_ERROR  -2606
#define GET_BANK_UNEXP_START   -2607
#define GET_BANK_CRC_ERROR     -2608
#define GET_BANK_UNKWN_BANK    -2609
#define GET_BANK_VERSION_ERROR -2610
