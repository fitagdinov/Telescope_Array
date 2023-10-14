/*
 * declaration of derived types to be used for interfacing to DST95 standard
 * types
 *
 * $Source: /hires_soft/uvm2k/dst/dst_std_types.h,v $
 * $Log: dst_std_types.h,v $
 * Revision 1.6  1996/05/02 08:30:38  mjk
 * Added DST standard type uinteger4 to be an unsigned integer4. While
 * one can say 'static integer4' or 'extern integer4', one can not say
 * 'unsigned integer4'. This is because the former are 'storage class
 * specifiers', while the words 'signed' and 'unsigned' constitute part
 * of the 'type specifer'. Thus 'unsigned int' must have its own typedef.
 * The practical reason for adding this type is for use with the DST
 * modified versions of various Numerical Recipes routines.
 *
 * Revision 1.5  1995/10/11  15:57:46  mjk
 * Forgot to put in #define during last fix
 *
 * Revision 1.4  1995/10/11  15:55:29  mjk
 * Provide for conditional inclusion via #ifndef
 *
 * Revision 1.3  1995/03/18  00:35:15  jeremy
 * *** empty log message ***
 *
 * created:  CCJ  14-JAN-1995
 *           ultrix version only...will modify as needed to accomdate
 *           other platforms
 */

/* the following is the table of number of bytes each architecture assume
   for the 8 different basic types (n/a == not available) :
             
CPU/compiler 	pointer char	short	int	long	float	double	LD*
------------------------------------------------------------------------------
decstation/cc	4	1     	2	4	4	4	8	n/a
decstation/gcc	4	1	2	4	4	4	8	8
alpha-osf1/cc	8	1	2	4	8	4	8	n/a
alpha-osf1/gcc	8	1	2	4	8	4	8	8
R4xxx-IRIX/cc	4	1	2	4	4	4	8	n/a
R4xxx-IRIX/gcc
IBM-AIX/cc	4	1	2	4	4	4	8	8**
IBM-AIX/gcc	4	1	2	4	4	4	8	8
VAX-VMS/cc      4	1	2	4	4	4	8	n/a
alpha-vms/cc    4       1       2       4       4       4       8       16

*/


#ifndef _DST_STD_TYPES_
#define _DST_STD_TYPES_

typedef  char           integer1  ;
typedef  short int      integer2  ;
typedef  unsigned short uinteger2 ;
typedef  int            integer4  ;
typedef  unsigned int   uinteger4 ;

/* reals */

typedef  float       real4     ;
typedef  double      real8     ;

#endif

