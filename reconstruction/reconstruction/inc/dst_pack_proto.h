/*
 * function prototypes for external use: 
 *
 * $Source: /hires_soft/uvm2k/dst/dst_pack_proto.h,v $
 * $Log: dst_pack_proto.h,v $
 * Last modified: 2019/12/14 DI
 * Revision 1.4  1996/02/13 21:36:57  mjk
 * Put in #ifndef conditional
 *
 * Revision 1.3  1995/03/21  21:18:36  jui
 * added dst_packi4asi2_ and dst_unpacki2asi4_
 *
 * Revision 1.2  1995/03/18  00:35:14  jeremy
 * *** empty log message ***
 *
 */

#ifndef _DST_PACK_PROTO_
#define _DST_PACK_PROTO_

#include "dst_std_types.h"

#ifdef __cplusplus
extern "C" {
#endif
  /* packing */
  integer4 dst_packi1_(integer1 I1obj[], integer4 *Nobj,
		       integer1 Bank[], integer4 *LenBank, integer4 *MaxLen);
  integer4 dst_packi2_(integer2 I2obj[], integer4 *Nobj,
		       integer1 Bank[], integer4 *LenBank, integer4 *MaxLen);
  integer4 dst_packi4asi2_(integer4 I4obj[], integer4 *Nobj,
			   integer1 Bank[], integer4 *LenBank, integer4 *MaxLen);
  integer4 dst_packi4_(integer4 I4obj[], integer4 *Nobj,
		       integer1 Bank[], integer4 *LenBank, integer4 *MaxLen);
  integer4 dst_packr4_(real4 R4obj[], integer4 *Nobj,
		       integer1 Bank[], integer4 *LenBank, integer4 *MaxLen);
  integer4 dst_packr8_(real8 R8obj[], integer4 *Nobj,
		       integer1 Bank[], integer4 *LenBank, integer4 *MaxLen);
  /* unpacking */
  integer4 dst_unpacki1_(integer1 I1obj[], integer4 *Nobj,
			 integer1 Bank[], integer4 *PosBank, integer4 *MaxLen);
  integer4 dst_unpacki2_(integer2 I2obj[], integer4 *Nobj,
			 integer1 Bank[], integer4 *PosBank, integer4 *MaxLen);
  integer4 dst_unpacki2asi4_(integer4 I4obj[], integer4 *Nobj,
			     integer1 Bank[], integer4 *PosBank, integer4 *MaxLen);
  integer4 dst_unpacki4_(integer4 I4obj[], integer4 *Nobj,
			 integer1 Bank[], integer4 *PosBank, integer4 *MaxLen);
  integer4 dst_unpackr4_(real4 R4obj[], integer4 *Nobj,
			 integer1 Bank[], integer4 *PosBank, integer4 *MaxLen);
  integer4 dst_unpackr8_(real8 R8obj[], integer4 *Nobj,
			 integer1 Bank[], integer4 *PosBank, integer4 *MaxLen);
#ifdef __cplusplus
} /* end extern "C" */
#endif


#endif
