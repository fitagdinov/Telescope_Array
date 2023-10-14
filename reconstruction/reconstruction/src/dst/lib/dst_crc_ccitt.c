/*
 * CRC-CCITT conforming "CRC" checksum routine for DST95 package
 * $Source: /hires_soft/uvm2k/dst/dst_crc_ccitt.c,v $
 * $Log: dst_crc_ccitt.c,v $
 * Revision 1.2  1995/03/18 00:35:10  jeremy
 * *** empty log message ***
 *
 * created:  CCJ  14-FEB-1995 ultrix/ALPHA version only...will modify
 * as needed to accomdate other platforms 
 */

#include "dst_std_types.h"		/* standard derived types */

/*
 * some external definitions 
 */

typedef unsigned char u_char;
#define LOBYTE(x) ((u_char)((x) & 0xFF))
#define HIBYTE(x) ((u_char)((x) >> 8))

/*
 * function prototypes 
 */

integer4
dst_crc_ccitt_(integer4 *lenbuf, u_char bufptr[]);

unsigned short
dst_icrc1(unsigned short crc, unsigned char onech);


/*
 * functions 
 */

integer4
dst_crc_ccitt_(integer4 *lenbuf, u_char bufptr[])
{
   unsigned short  crc = 0;

   static unsigned short icrctab[256], init = 0;
   static u_char   rchr[256];

   integer4        j;
   unsigned short  cword = crc;
   static u_char   it[16] = {0, 8, 4, 12, 2, 10, 6, 14,
			     1, 9, 5, 13, 3, 11, 7, 15};

   /*
    * definitions for CRC_CCITT convention 
    */

   static short    jinit = 0;
   static int      jrev = -1;

   if (!init) {
      init = 1;
      for (j = 0; j < 256; j++) {
	 icrctab[j] = dst_icrc1(((unsigned short) (j << 8)), (u_char) 0);
	 rchr[j] = (u_char) (it[j & 0xF] << 4 | it[j >> 4]);
      }
   }
   if (jinit >= 0)
      cword = ((u_char) jinit) | (((u_char) jinit) << 8);

   for (j = 0; j < *lenbuf; j++)
      cword = icrctab[(jrev < 0 ? rchr[bufptr[j]] :
		       bufptr[j]) ^ HIBYTE(cword)] ^ LOBYTE(cword) << 8;
   return ((integer4) (jrev >= 0 ? cword : rchr[HIBYTE(cword)] |
		       rchr[LOBYTE(cword)] << 8));
}

/* =============== */


unsigned short
dst_icrc1(unsigned short crc, u_char onech)
{
   int             i;
   unsigned short  ans = (crc ^ onech << 8);
   
   for (i = 0; i < 8; i++) {
     if (ans & 0x8000) {
       // ans = (ans <<= 1) ^ 4129;
       ans <<= 1;
       ans = ans^4129;
     }
     else
       ans <<= 1;
   }
   return ans;
}
