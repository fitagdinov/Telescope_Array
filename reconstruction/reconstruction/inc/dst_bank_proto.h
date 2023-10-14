/*
 * function prototypes 
 * $Source: /hires_soft/uvm2k/dst/dst_bank_proto.h,v $
 * $Log: dst_bank_proto.h,v $
 * Last modified 2019/12/14 DI
 * Revision 1.5  2001/05/14 15:02:46  reil
 * Added CPP extern C definitions
 *
 * Revision 1.4  1996/02/13 21:33:57  mjk
 * Put in #ifndef control
 *
 * Revision 1.3  1995/07/24  21:00:32  mjk
 * Supply prototype for dst_initbank_
 *
 * Revision 1.2  1995/03/18  00:35:10  jeremy
 * *** empty log message ***
 *
 */

#ifndef _DST_BANK_PROTO_
#define _DST_BANK_PROTO_


#ifdef __cplusplus
extern "C" {
#endif
  integer4 dst_open_unit_(integer4 *NumUnit, integer1 NameUnit[], integer4 *mode);
  integer4 dst_close_unit_(integer4 *NumUnit);
  integer4 dst_write_bank_(integer4 *NumUnit, integer4 *LenBank, integer1 Bank[]);
  integer4 dst_read_bank_(integer4 *NumUnit, integer4 *DiagLevel,
			  integer1 Bank[], integer4 *LenBank,
			  integer4 *BankTyp, integer4 *BankVer);
  integer4  dst_initbank_(integer4 *bankid, integer4 *bankversion, integer4 *blen, 
			  integer4 *maxlen, integer1 *bank);
#ifdef __cplusplus
} /* end extern "C" */
#endif
  
#endif
