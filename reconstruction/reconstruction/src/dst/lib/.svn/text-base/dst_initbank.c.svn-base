/*
 * $Source: /hires_soft/uvm2k/dst/dst_initbank.c,v $
 * $Log: dst_initbank.c,v $
 * Revision 1.4  1995/10/05 15:56:07  boyer
 * Get rid of dst_initbank
 *
 * Revision 1.3  1995/07/24  21:02:26  mjk
 * Create an underscore appended version of dst_initbank.
 * Old dst_initbank will be phases out very soon
 * Also changed an int --> integer4
 *
 * Revision 1.2  1995/03/18  00:35:13  jeremy
 * *** empty log message ***
 *
 * modified by jT at 10:55 AM on 3/10/95
*/

#include "dst_std_types.h"
#include "dst_pack_proto.h"


/* Obsolete and will be phased out soon */
/* Now phased Out
 *int dst_initbank (integer4 *bankid, 
 *                 integer4 *bankversion, 
 *                 integer4 *blen, 
 *                 integer4 *maxlen,
 *                 integer1 *bank)        
 * it initialized blen, and pack the id and version to the bank returns rcode
 *{ 
 *  integer4 nobj=1 ;
 *  integer4 rcode;
 *    *blen=0;
 *    rcode=dst_packi4_(bankid, &nobj, bank,blen,maxlen);
 *    rcode=dst_packi4_(bankversion, &nobj,bank,blen,maxlen);
 *    return rcode;
 *}
*/     


/* New code. Notice the underscore */

integer4 dst_initbank_ (integer4 *bankid, 
                        integer4 *bankversion, 
                        integer4 *blen, 
                        integer4 *maxlen,
                        integer1 *bank) 

	/* it initialized blen, and pack the id and version to the 
	   bank returns rcode */
{ 
   integer4 nobj=1 ;
   integer4 rcode;

   *blen=0;
   rcode=dst_packi4_(bankid, &nobj, bank,blen,maxlen);
   rcode=dst_packi4_(bankversion, &nobj,bank,blen,maxlen);
   return rcode;
}
