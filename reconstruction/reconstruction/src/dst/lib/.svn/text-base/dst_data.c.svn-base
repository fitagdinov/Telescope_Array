/*
 * $Source: /hires_soft/uvm2k/dst/dst_data.c,v $
 * $Log: dst_data.c,v $
 * Revision 1.5  1997/04/13 00:38:35  jui
 * removed initialization message (concerning byte-ordering and
 * floating point storage formats) to "stderr" as mandated by
 * popular request
 *
 * Revision 1.4  1996/05/22  03:52:18  mjk
 * Corrected small typo in a fprintf message.
 *
 * Revision 1.3  1995/03/20  17:28:14  jui
 * changed "printf(" to "fprintf(stderr,", and added some info to err msgs.
 *
 * Revision 1.2  1995/03/18  00:35:12  jeremy
 * *** empty log message ***
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_data_proto.h"

/*
 * Some external variables 
 */

integer4        dst_data_init = 0;
/* has the data-handling section been initialized? yes=1, no=0 */

integer4        dst_big_endian = 0;
/*
 * is "native format "big-endian": 0  NO (little-endian == DST format) 1  YES
 * (swap required) 
 */

integer4        dst_r4_format = 0;
/*
 * "native" REAL*4 storage format: 0    IEEE 1  VAX/ALPHA VMS F-FLOATING -1
 * unknown 
 */

integer4        dst_r8_format = 0;
/*
 * "native" REAL*8 storage format: 0    IEEE 1  VAX/ALPHA VMS G-FLOATING 2
 * VAX/ALPHA VMS D-FLOATING -1 unknown 
 */

integer4        dst_mslw;	/* array index for most significant long word
				 * in R*8 */
integer4        dst_lslw;	/* array index for least significant long
				 * word in R*8 */

/* ============== */

/*
 * warning: dst_init_data_ is strictly speaking NON-PORTABLE code. 
 */

void
dst_init_data_()
{
   union {
      unsigned char   i1[8];
      unsigned short  i2[4];
      unsigned int    i4[2];
      float           r4[2];
      double          r8;
   }               u;

   /*
    * first we test for byte-ordering 
    */

   u.i4[0] = 1012345678;

   if (u.i2[0] == 0x2b4e && u.i2[1] == 0x3c57) {

      dst_big_endian = 0;
/*      fprintf(stderr,
         " $$$ Your machine uses LITTLE-ENDIAN byte ordering\n"); */
      dst_mslw = 1;		/* most significant long word is the second
				 * (index=1) */
      dst_lslw = 0;		/* least significant long word is the first
				 * (index=0) */

   } else if (u.i2[0] == 0x3c57 && u.i2[1] == 0x2b4e) {

      dst_big_endian = 1;
/*      fprintf(stderr,
         " $$$ Your machine uses BIG-ENDIAN byte ordering\n"); */
      dst_mslw = 0;		/* most significant long word is the first
				 * (index=0) */
      dst_lslw = 1;		/* least significant long word is the second
				 * (index=1) */

   } else {

      fprintf(stderr,
         " ^^^ dst_init_data_ failed:\n");
      fprintf(stderr,
         "  your machine uses a VERY non-standard integer storage \n");
      fprintf(stderr,
         "  format and is therefore NOT SUPPORTED by DST95 \n");
      fprintf(stderr,
         "  please contact jui@phoebe.physics.utah.edu \n");
      exit(-1);

   }

   /*
    * now we try to find out about REAL format 
    */

   u.r4[0] = 1.23456789e+12;

   if ((u.i4[0] & 0xffffff00) == 0x538fb800) {

      dst_r4_format = 0;
/*      fprintf(stderr,
         " $$$ Your machine uses IEEE format for float/REAL*4\n"); */

   } else if ((u.i4[0] & 0xff00ffff) == 0xb800548f) {

      dst_r4_format = 1;
/*      fprintf(stderr,
         " $$$ Your machine uses F format for float/REAL*4\n"); */

   } else {

/*      fprintf(stderr,
         " $$$ Your machine uses an unknown format for float/REAL*4\n"); */
      dst_r4_format = (-1);

   }

   /*
    * now we try to find out about DOUBLE PRECISION format 
    */

   u.r8 = 1.234567890123450000e+12;
   if ((u.i4[dst_lslw] & 0xffffff00) == 0xb04cb700 &&
       u.i4[dst_mslw] == 0x4271f71f) {

      dst_r8_format = 0;
/*      fprintf(stderr,
         " $$$ Your machine uses IEEE format for double/REAL*8\n"); */

   } else if (u.i4[dst_lslw] == 0xf71f4291 &&
	      (u.i4[dst_mslw] & 0xff00ffff) == 0xb700b04c) {

      dst_r8_format = 1;
/*      fprintf(stderr,
         " $$$ Your machine uses G format for double/REAL*8\n"); */

   } else if (u.i4[dst_lslw] == 0xb8fd548f &&
	      (u.i4[dst_mslw] & 0xf000ffff) == 0xb0008265) {

      dst_r8_format = 2;
/*      fprintf(stderr,
         " $$$ Your machine uses D format for double/REAL*8\n"); */

   } else {

      dst_r8_format = (-1);
/*      fprintf(stderr,
         " $$$ Your machine uses an unknown format for double/REAL*8\n"); */

   }

   dst_data_init = 1;

   return;
}

/* ============== */

integer4
dst_byte_order_(integer4 nbyte, integer1 obj[])
{
   integer4        i;
   integer1        temp[8];

   if (!dst_data_init) {
      dst_init_data_();
   }
   if (dst_big_endian) {
      for (i = 0; i < nbyte; i = i + 1) {
	 temp[i] = obj[i];
      }
      for (i = 0; i < nbyte; i = i + 1) {
	 obj[i] = temp[nbyte - i - 1];
      }
   }
   return SUCCESS;
}

/* ============== */

integer4
dst_r4_ntoi_(integer1 obj[])
{
   unsigned int    it, l16, u7;
   unsigned int    sign;
   int             kexpon;

   integer4        i, n, k;
   float           r;
   /* double          f, f1; */
   double          f1;

   union {
      integer1        i1[4];
      unsigned int    i4;
      real4           r4;
   }               u;

   if (!dst_data_init) {
      dst_init_data_();
   }
   for (i = 0; i < 4; i = i + 1) {
      u.i1[i] = obj[i];
   }

   /*
    * Now we look at the 3 different cases: IEEE, VAX-F, or unknown: 
    */

   if (dst_r4_format == 0) {	/* IEEE */

   } else if (dst_r4_format == 1) {	/* VAX-F */

      r = u.r4;
      if (r == 0.0) {

	 u.i4 = 0x00000000;

      } else {

	 it = u.i4;
	 l16 = ((it & 0xffff0000) >> 16);
	 u7 = ((it & 0x0000007f) << 16);
	 sign = ((it & 0x00008000) << 16);
	 kexpon = ((it & 0x00007f80) >> 7);
	 if (kexpon < 2) {	/* underflow: set to zero */
	    u.i4 = 0x00000000;
	    fprintf(stderr,
	       " ^^^ dst_r4_ntoi_ warning: VAX-F R*4 value: %14.7g \n"
	       ,u.r4);
	    fprintf(stderr,
	       "  is too small to be represented in IEEE format\n");
	    fprintf(stderr,
	       "  and therefore value is set to zero on input\n");
	 } else {
	    u.i4 = (l16 | u7 | ((kexpon - 2) << 23) | sign);
	 }
      }

   } else {			/* unknown */

      r = u.r4;
      if (r == 0.0) {

	 u.i4 = 0x00000000;

      } else {

	 it = 0x00000000;
	 if (r < 0.0) {
	    it = (it | 0x80000000);
	 }
	 f1 = fabs((double) (r));
	 /* f = frexp(f1, &n); */
	 frexp(f1, &n);

	 it = (it | (((int) (unsigned char) (n - 1 + 127)) << 23));
	 f1 = f1 / pow(((double) (2.0)), ((double) (n - 1)))
	    - ((double) (1.0));

	 n = 0;
	 while ((n > (-24)) && (f1 != ((double) (0.0)))) {
	   /* f = frexp(f1, &k); */
	    frexp(f1, &k);
	    f1 = f1 / pow(((double) (2.0)), ((double) (k - 1)))
	       - ((double) (1.0));
	    n = n + (k - 1);
	    it = (it | ((int) (((int) (1)) << (23 + n))));
	 }
	 u.i4 = it;

      }

   }

   for (i = 0; i < 4; i = i + 1) {
      obj[i] = u.i1[i];
   }

   return SUCCESS;
}

/* ============== */

integer4
dst_r8_ntoi_(integer1 obj[])
{
   unsigned int    imost, ileast;
   unsigned int    ll16, lu16, ml16, mu4;	/* for VAX G-format */
   unsigned int    ll13, lm16, lu3, ml13, mu7;	/* for VAX D-format */
   unsigned int    sign;
   int             kexpon;

   integer4        i, n, k;
   double          r;
   /* double          f, f1; */
   double          f1;

   union {
      integer1        i1[8];
      unsigned int    i4[2];
      real8           r8;
   }               u;

   if (!dst_data_init) {
      dst_init_data_();
   }
   for (i = 0; i < 8; i = i + 1) {
      u.i1[i] = obj[i];
   }

   /*
    * Now we look at the 3 different cases: IEEE, VAX-F, or unknown: 
    */

   if (dst_r8_format == 0) {	/* IEEE */

   } else if (dst_r8_format == 1) {	/* VAX-G */

      imost = u.i4[dst_mslw];
      ileast = u.i4[dst_lslw];

      ll16 = ((imost & 0xffff0000) >> 16);
      lu16 = ((imost & 0x0000ffff) << 16);

      ml16 = ((ileast & 0xffff0000) >> 16);
      mu4 = ((ileast & 0x0000000f) << 16);
      sign = ((ileast & 0x00008000) << 16);

      kexpon = ((ileast & 0x00007ff0) >> 4);
      if (kexpon < 2) {		/* underflow: set to zero */
	 u.i4[dst_lslw] = 0x00000000;
	 u.i4[dst_mslw] = 0x00000000;
	 fprintf(stderr,
	    " ^^^ dst_r8_ntoi_ warning: VAX-G R*8 value: %21.14g \n",
	    u.r8);
	 fprintf(stderr,
	    "  is too small to be represented in IEEE format\n");
	 fprintf(stderr,
	    "  and therefore value is set to zero on input\n");
      } else {
	 u.i4[dst_lslw] = (ll16 | lu16);
	 u.i4[dst_mslw] = (ml16 | mu4 | ((kexpon - 2) << 20) | sign);
      }

   } else if (dst_r8_format == 2) {	/* VAX-D */

      imost = u.i4[dst_mslw];
      ileast = u.i4[dst_lslw];

      ll13 = ((imost & 0xfff80000) >> 19);
      lm16 = ((imost & 0x0000ffff) << 13);
      lu3 = ((ileast & 0x00070000) << 13);

      ml13 = ((ileast & 0xfff80000) >> 19);
      mu7 = ((ileast & 0x0000007f) << 13);
      kexpon = ((ileast & 0x00007f80) >> 7) - 129;
      sign = ((ileast & 0x00008000) << 16);

      u.i4[dst_lslw] = (ll13 | lm16 | lu3);
      u.i4[dst_mslw] = (ml13 | mu7 | ((kexpon + 1023) << 20) | sign);

   } else {			/* unknown */

      r = u.r8;
      if (r == ((double) (0.0))) {

	 u.i4[0] = 0x00000000;
	 u.i4[1] = 0x00000000;

      } else {

	 imost = 0x00000000;
	 ileast = 0x00000000;

	 if (r < ((double) (0.0))) {
	    imost = (imost | 0x80000000);
	 }
	 f1 = fabs(r);
	 /* f = frexp(f1, &n); */
	 frexp(f1, &n);

	 imost = (imost | ((int)
		       (((unsigned short) (n - 1 + 1023) & 0x07ff) << 20)));

	 f1 = f1 / pow(((double) (2.0)), ((double) (n - 1)))
	    - ((double) (1.0));

	 n = 0;
	 while ((n > (-53)) && (f1 != ((double) (0.0)))) {
	   /* f = frexp(f1, &k); */
	   frexp(f1, &k);
	    f1 = f1 / pow(((double) (2.0)), ((double) (k - 1)))
	       - ((double) (1.0));
	    n = n + (k - 1);
	    if (n > -21) {
	       imost = (imost | ((int) (((int) (1)) << (20 + n))));
	    } else {
	       ileast = (ileast | ((int) (((int) (1)) << (52 + n))));
	    }
	 }

	 u.i4[dst_lslw] = ileast;
	 u.i4[dst_mslw] = imost;

      }

   }

   for (i = 0; i < 8; i = i + 1) {
      obj[i] = u.i1[i];
   }

   return SUCCESS;
}

/* ============== */

integer4
dst_r4_iton_(integer1 obj[])
{
   unsigned int    it, l16, u7;
   unsigned int    sign;
   int             kexpon;

   integer4        i, iexpon;
   double          f, frac, expon;

   union {
      integer1        i1[4];
      unsigned int    i4;
      real4           r4;
   }               u;

   if (!dst_data_init) {
      dst_init_data_();
   }
   for (i = 0; i < 4; i = i + 1) {
      u.i1[i] = obj[i];
   }

   /*
    * Now we look at the 3 different cases: IEEE, VAX-F, or unknown: 
    */

   if (dst_r4_format == 0) {	/* IEEE */

   } else if (dst_r4_format == 1) {	/* VAX-F */

      it = u.i4;
      if (it == 0x00000000) {
	 u.r4 = 0.0;
      } else {
	 l16 = ((it & 0x0000ffff) << 16);
	 u7 = ((it & 0x007f0000) >> 16);
	 kexpon = ((it & 0x7f800000) >> 23);
	 sign = ((it & 0x80000000) >> 16);
	 if (kexpon > 253) {	/* overflow: set to: */
	    u.i4 = (0xffff7fff | sign);
	    fprintf(stderr,
	       " ^^^ dst_r4_iton_ warning: IEEE R*4 value,\n");
            fprintf(stderr,
	       "  shown in hex. by: %8x , is\n", it);
	    fprintf(stderr,
	       "  is too large to be represented in VAX F-format and\n");
	    fprintf(stderr,
	       "  thus the absoulte value is set to 1.701412e+38\n");
	 } else {
	    u.i4 = (l16 | u7 | ((kexpon + 2) << 7) | sign);
	 }
      }

   } else {			/* unknown */

      it = u.i4;
      if (it == 0x00000000) {

	 u.r4 = 0.0;

      } else {

	 frac = ((double) (1.0));
	 for (i = 23; i > 0; i = i - 1) {
	    if (it & (0x00000001 << (i - 1))) {
	       frac = frac + pow(((double) (2.0)), ((double) (i - 24)));
	    }
	 }

	 iexpon = ((it & (0x7f800000)) >> 23) - 127;
	 expon = pow(((double) (2.0)), ((double) (iexpon)));
	 f = frac * expon;

	 if (it & 0x80000000) {
	    f = (-f);
	 }
	 u.r4 = ((float) (f));

      }

   }

   for (i = 0; i < 4; i = i + 1) {
      obj[i] = u.i1[i];
   }

   return SUCCESS;
}

/* ============== */

integer4
dst_r8_iton_(integer1 obj[])
{
   unsigned int    imost, ileast;
   unsigned int    ll16, lu16, ml16, mu4;	/* for VAX G-format */
   unsigned int    ll13, lm16, lu3, ml13, mu7;	/* for VAX D-format */
   unsigned int    sign;
   int             kexpon;

   double          f, frac, expon;
   int             iexpon, i;

   union {
      integer1        i1[8];
      unsigned int    i4[2];
      real8           r8;
   }               u;

   if (!dst_data_init) {
      dst_init_data_();
   }
   for (i = 0; i < 8; i = i + 1) {
      u.i1[i] = obj[i];
   }

   /*
    * Now we look at the 3 different cases: IEEE, VAX-F, or unknown: 
    */

   if (dst_r8_format == 0) {	/* IEEE */

   } else if (dst_r8_format == 1) {	/* VAX-G */

      imost = u.i4[dst_mslw];
      ileast = u.i4[dst_lslw];

      ll16 = ((ileast & 0x0000ffff) << 16);
      lu16 = ((ileast & 0xffff0000) >> 16);

      ml16 = ((imost & 0x0000ffff) << 16);
      mu4 = ((imost & 0x000f0000) >> 16);
      sign = ((imost & 0x80000000) >> 16);
      kexpon = ((imost & 0x7ff00000) >> 20);

      if (kexpon > 2046) {	/* underflow: set to zero */
	 u.i4[dst_mslw] = 0xffffffff;
	 u.i4[dst_lslw] = (0xffff7fff & sign);
	 fprintf(stderr,
	    " ^^^ dst_r4_iton_ warning: IEEE R*8 value,\n");
	 fprintf(stderr,
	    "  shown in hex. by: %8x %8x , is\n", imost, ileast);
	 fprintf(stderr,
	    "  too large to be represented in VAX G-format and the\n");
	 fprintf(stderr,
	    "  absoulte value is set to 8.9884656743115785e+307\n");
      } else {
	 u.i4[dst_mslw] = (ll16 | lu16);
	 u.i4[dst_lslw] = (ml16 | mu4 | ((kexpon + 2) << 4) | sign);
      }

   } else if (dst_r8_format == 2) {	/* VAX-D */

      imost = u.i4[dst_mslw];
      ileast = u.i4[dst_lslw];

      ll13 = ((ileast & 0x00001fff) << 19);
      lm16 = ((ileast & 0x1fffe000) >> 13);

      lu3 = ((ileast & 0xe0000000) >> 13);
      ml13 = ((imost & 0x00001fff) << 19);
      mu7 = ((imost & 0x000fe000) >> 13);
      sign = ((imost & 0x80000000) >> 16);

      kexpon = ((imost & 0x7ff00000) >> 20) - 1023 + 129;

      if (kexpon > 255) {
	 u.i4[dst_mslw] = 0xffffffff;
	 u.i4[dst_lslw] = (0xffff7fff | sign);
      } else if (kexpon < 0) {
	 u.r8 = ((double) (0.0));
      } else {
	 u.i4[dst_mslw] = (ll13 | lm16);
	 u.i4[dst_lslw] = (lu3 | ml13 | mu7 | (kexpon << 7) | sign);
      }

   } else {			/* unknown */

      imost = u.i4[dst_mslw];
      ileast = u.i4[dst_lslw];

      if (imost == 0x00000000 && ileast == 0x00000000) {

	 u.r8 = 0.0;

      } else {

	 frac = ((double) (1.0));

	 for (i = 20; i > 0; i = i - 1) {
	    if (imost & (0x00000001 << (i - 1))) {
	       frac = frac + pow(((double) (2.0)), ((double) (i - 21)));
	    }
	 }

	 for (i = 32; i > 0; i = i - 1) {
	    if (ileast & (0x00000001 << (i - 1))) {
	       frac = frac + pow(((double) (2.0)), ((double) (i - 53)));
	    }
	 }

	 iexpon = ((imost & (0x7ff00000)) >> 20) - 1023;
	 expon = pow(((double) (2.0)), ((double) (iexpon)));
	 f = frac * expon;

	 if (imost & 0x80000000) {
	    f = (-f);
	 }
	 u.r8 = f;
      }
   }

   for (i = 0; i < 8; i = i + 1) {
      obj[i] = u.i1[i];
   }

   return SUCCESS;
}
