/*
 * base-level packing and unpacking routines for the DST95 package 
 * $Source: /hires_soft/uvm2k/dst/dst_pack.c,v $
 * $Log: dst_pack.c,v $
 * Revision 1.3  1995/03/21 21:15:32  jui
 * slight change in internal logic, added dst_packi4asi2_ and
 * dst_unpacki2asi4_ This makes dst_packi4asi2_.c obsolete
 *
 * Revision 1.2  1995/03/18  00:35:14  jeremy
 * *** empty log message ***
 *
 * created:  CCJ  17-JAN-1995 ultrix version only...will modify as needed to
 * accomdate other platforms 
 */

#include <stdio.h>

#include "dst_std_types.h"	/* standard derived data types */
#include "dst_err_codes.h"	/* return codes */
#include "dst_data_proto.h"	/* prototypes for data maninpulation */

#include "dst_pack_proto.h"     /* prototypes for pack/unpack functions*/

/* ============== */

integer4
dst_packi1_(integer1 I1obj[], integer4 *Nobj,
	    integer1 Bank[], integer4 *LenBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        proj_end;
   
   proj_end = (*LenBank) + (*Nobj);
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   for (i = 0; i < *Nobj; i = i + 1) {
      Bank[*LenBank] = I1obj[i];
      *LenBank = *LenBank + 1;
   }
   return SUCCESS;
}

/* ============== */

integer4
dst_packi2_(integer2 I2obj[], integer4 *Nobj,
	    integer1 Bank[], integer4 *LenBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      integer2        tempi2;
      integer1        obj[2];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*LenBank) + (*Nobj)*2;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.tempi2 = I2obj[i];
      rctemp = dst_byte_order_(2, u.obj);

      if (rctemp != SUCCESS) {
	 failcode = TO_DST_FAIL;
	 break;
      }
      Bank[*LenBank] = u.obj[0];
      Bank[*LenBank + 1] = u.obj[1];
      *LenBank = *LenBank + 2;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_packi4asi2_(integer4 I4obj[], integer4 *Nobj,
	    integer1 Bank[], integer4 *LenBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      integer2        tempi2;
      integer1        obj[2];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*LenBank) + (*Nobj)*2;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.tempi2 = (integer2) I4obj[i];
      rctemp = dst_byte_order_(2, u.obj);

      if (rctemp != SUCCESS) {
	 failcode = TO_DST_FAIL;
	 break;
      }
      Bank[*LenBank] = u.obj[0];
      Bank[*LenBank + 1] = u.obj[1];
      *LenBank = *LenBank + 2;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_packi4_(integer4 I4obj[], integer4 *Nobj,
	    integer1 Bank[], integer4 *LenBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      integer4     tempi4;
      integer1     obj[4];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*LenBank) + (*Nobj)*4;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.tempi4 = I4obj[i];
      rctemp = dst_byte_order_(4, u.obj);

      if (rctemp != SUCCESS) {
	 failcode = TO_DST_FAIL;
	 break;
      }
      Bank[*LenBank] = u.obj[0];
      Bank[*LenBank + 1] = u.obj[1];
      Bank[*LenBank + 2] = u.obj[2];
      Bank[*LenBank + 3] = u.obj[3];
      *LenBank = *LenBank + 4;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_packr4_(real4 R4obj[], integer4 *Nobj,
	    integer1 Bank[], integer4 *LenBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      real4        tempr4;
      integer1     obj[4];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*LenBank) + (*Nobj)*4;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.tempr4 = R4obj[i];
      rctemp = dst_r4_ntoi_(u.obj);

      if (rctemp != SUCCESS) {
	 failcode = R4_NTOI_FAIL;
	 break;
      }
      rctemp = dst_byte_order_(4, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = TO_DST_FAIL;
	 break;
      }
      Bank[*LenBank] = u.obj[0];
      Bank[*LenBank + 1] = u.obj[1];
      Bank[*LenBank + 2] = u.obj[2];
      Bank[*LenBank + 3] = u.obj[3];
      *LenBank = *LenBank + 4;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_packr8_(real8 R8obj[], integer4 *Nobj,
	    integer1 Bank[], integer4 *LenBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      real8        tempr8;
      integer1     obj[8];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*LenBank) + (*Nobj)*8;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.tempr8 = R8obj[i];
      rctemp = dst_r8_ntoi_(u.obj);

      if (rctemp != SUCCESS) {
	 failcode = R8_NTOI_FAIL;
	 break;
      }
      rctemp = dst_byte_order_(8, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = TO_DST_FAIL;
	 break;
      }
      Bank[*LenBank] = u.obj[0];
      Bank[*LenBank + 1] = u.obj[1];
      Bank[*LenBank + 2] = u.obj[2];
      Bank[*LenBank + 3] = u.obj[3];
      Bank[*LenBank + 4] = u.obj[4];
      Bank[*LenBank + 5] = u.obj[5];
      Bank[*LenBank + 6] = u.obj[6];
      Bank[*LenBank + 7] = u.obj[7];
      *LenBank = *LenBank + 8;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_unpacki1_(integer1 I1obj[], integer4 *Nobj,
              integer1 Bank[], integer4 *PosBank, integer4 *MaxLen)
{
   integer4        i;

   integer4        proj_end;
   
   proj_end = (*PosBank) + (*Nobj);
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   for (i = 0; i < *Nobj; i = i + 1) {
      I1obj[i] = Bank[*PosBank];
      *PosBank = *PosBank + 1;
   }
   return SUCCESS;
}

/* ============== */

integer4
dst_unpacki2_(integer2 I2obj[], integer4 *Nobj,
              integer1 Bank[], integer4 *PosBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      integer2     tempi2;
      integer1     obj[2];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*PosBank) + (*Nobj)*2;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.obj[0] = Bank[*PosBank];
      u.obj[1] = Bank[*PosBank + 1];

      rctemp = dst_byte_order_(2, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = FROM_DST_FAIL;
	 break;
      }
      I2obj[i] = u.tempi2;
      *PosBank = *PosBank + 2;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_unpacki2asi4_(integer4 I4obj[], integer4 *Nobj,
              integer1 Bank[], integer4 *PosBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      integer2     tempi2;
      integer1     obj[2];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*PosBank) + (*Nobj)*2;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.obj[0] = Bank[*PosBank];
      u.obj[1] = Bank[*PosBank + 1];

      rctemp = dst_byte_order_(2, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = FROM_DST_FAIL;
	 break;
      }
      I4obj[i] = (integer4) u.tempi2;
      *PosBank = *PosBank + 2;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_unpacki4_(integer4 I4obj[], integer4 *Nobj,
              integer1 Bank[], integer4 *PosBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      integer4     tempi4;
      integer1     obj[4];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*PosBank) + (*Nobj)*4;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.obj[0] = Bank[*PosBank];
      u.obj[1] = Bank[*PosBank + 1];
      u.obj[2] = Bank[*PosBank + 2];
      u.obj[3] = Bank[*PosBank + 3];

      rctemp = dst_byte_order_(4, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = FROM_DST_FAIL;
	 break;
      }
      I4obj[i] = u.tempi4;
      *PosBank = *PosBank + 4;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_unpackr4_(real4 R4obj[], integer4 *Nobj,
              integer1 Bank[], integer4 *PosBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      real4        tempr4;
      integer1     obj[4];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*PosBank) + (*Nobj)*4;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.obj[0] = Bank[*PosBank];
      u.obj[1] = Bank[*PosBank + 1];
      u.obj[2] = Bank[*PosBank + 2];
      u.obj[3] = Bank[*PosBank + 3];

      rctemp = dst_byte_order_(4, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = FROM_DST_FAIL;
	 break;
      }
      rctemp = dst_r4_iton_(u.obj);
      if (rctemp != SUCCESS) {
	 failcode = R8_ITON_FAIL;
	 break;
      }
      R4obj[i] = u.tempr4;
      *PosBank = *PosBank + 4;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}

/* ============== */

integer4
dst_unpackr8_(real8 R8obj[], integer4 *Nobj,
              integer1 Bank[], integer4 *PosBank, integer4 *MaxLen)
{
   integer4        i;
   integer4        failcode;
   union {
      real8        tempr8;
      integer1     obj[8];
   } u;
   integer4        rctemp;

   integer4        proj_end;
   
   proj_end = (*PosBank) + (*Nobj)*8;
   if (proj_end > *MaxLen) {
      return MAXLEN_EXCEEDED;
   }

   failcode = 0;

   for (i = 0; i < *Nobj; i = i + 1) {
      u.obj[0] = Bank[*PosBank];
      u.obj[1] = Bank[*PosBank + 1];
      u.obj[2] = Bank[*PosBank + 2];
      u.obj[3] = Bank[*PosBank + 3];
      u.obj[4] = Bank[*PosBank + 4];
      u.obj[5] = Bank[*PosBank + 5];
      u.obj[6] = Bank[*PosBank + 6];
      u.obj[7] = Bank[*PosBank + 7];

      rctemp = dst_byte_order_(8, u.obj);
      if (rctemp != SUCCESS) {
	 failcode = FROM_DST_FAIL;
	 break;
      }
      rctemp = dst_r8_iton_(u.obj);
      if (rctemp != SUCCESS) {
	 failcode = R8_ITON_FAIL;
	 break;
      }
      R8obj[i] = u.tempr8;
      *PosBank = *PosBank + 8;
   }
   if (failcode != 0) {
      return failcode;
   } else {
      return SUCCESS;
   }
}
