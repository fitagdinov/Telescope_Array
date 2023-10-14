/* dst_packi4asui2_.c
 * $Source: /hires_soft/uvm2k/dst/dst_packi4asui2_.c,v $
 * $Log: dst_packi4asui2_.c,v $
 * Revision 1.1  1995/04/01 17:52:05  jtang
 * Initial revision
 *
 * Revision 1.2  1995/03/18  00:35:15  jeremy
 * *** empty log message ***
 *
 * modified by jT at 8:37 PM on 3/12/95
*/

#include <stdlib.h>
#include <stdio.h>

#include "dst_std_types.h"		/* standard derived data types */
#include "dst_pack_proto.h"

integer4
dst_packi4asui2_(integer4 *I4obj, integer4 *Nobj,
	    integer1 *Bank, integer4 *LenBank, integer4 *MaxLen)
{
   register int j;
   integer4 rcode;
   unsigned short *i;
   i=(unsigned short *) calloc (*Nobj, sizeof(integer2));
   for (j=0;j<*Nobj;j++) {
   	   i[j]=I4obj[j];
   }
   // rcode=dst_packi2_((integer1*)i, Nobj, Bank, LenBank, MaxLen) ;
   rcode=dst_packi2_((integer2 *)i, Nobj, Bank, LenBank, MaxLen) ;
   free(i);
   return rcode;
}


integer4
dst_unpackui2asi4_(integer4 *I4obj, integer4 *Nobj,
              integer1 *Bank, integer4 *PosBank, integer4 *MaxLen){
   register int j;
   integer4 rcode;
   unsigned short  *i;
   i=(unsigned short *) calloc (*Nobj, sizeof(integer2));
   // rcode=dst_unpacki2_((integer1*)i, Nobj, Bank, PosBank, MaxLen) ;
   rcode=dst_unpacki2_((integer2 *)i, Nobj, Bank, PosBank, MaxLen) ;
   for (j=0;j<*Nobj;j++) {
   	   I4obj[j]=i[j];
   }
   free(i);
   return rcode;
}       
