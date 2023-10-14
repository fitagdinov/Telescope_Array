/* Universal global variables use by most xxx_dst.c
 * $Source: /hires_soft/uvm2k/bank/univ_dst.h,v $
 * $Log: univ_dst.h,v $
 * Revision 1.12  2000/04/08 01:09:17  reil
 * Brought down most of library with my last change.  Users writing code for hires2 s
 * NOT use HR_UNIV_* since those numbers reflect a 24 mirror maximum.  Essentially our UNV
 * Number are only for HR1.  Perhaps we should define HR2_UNIV_MAXMIR etc
 *
 * Revision 1.11  2000/04/07  03:41:01  reil
 * Upgraded HR_UNIV_MAXMIR to 42 to allow hires 2 programs to function properly.
 * This has been done before but was lost.
 *
 * Revision 1.10  1997/09/08  20:58:15  acme
 * added definitions of true/false to assist usi compilation
 *
 * Revision 1.9  1997/04/28  18:49:10  jui
 * changed HR_UNIV_MAXMIR to 24 from 16
 * in anticipation of BigH operations
 *
 * Revision 1.8  1995/10/25  17:37:53  wiencke
 * added HR_UNIV_PRO2MIR parameter
 *
 * Revision 1.7  1995/08/22  23:39:35  vtodd
 * provided for conditional inclusion:  #ifndef _UNIV_
 *
 * Revision 1.6  1995/06/15  21:42:12  jeremy
 * changed HR_UNIV_MAXMIR to 16
 *
 * Revision 1.5  1995/04/01  00:23:04  jeremy
 * added HR_UNIV_MIRTUBE definition.
 *
 * Revision 1.4  1995/03/23  17:37:25  jeremy
 * fix up log comments from previous checkin.
 *
 * Revision 1.3  1995/03/23  17:33:29  jeremy
 * Added mode, diag & default bank size constants.
 *
 * Revision 1.2  1995/03/18  00:35:23  jeremy
 * added Source and Log RCS key words.
 *
 * modified by jT at 4:26 PM on 3/13/95
 * modified by jds at 9:55 AM on 3/16/95
*/

#ifndef _UNIV_
#define _UNIV_

#define HR_UNIV_MAXMIR  26
#define HR_UNIV_PRO2MIR 4
#define HR_UNIV_MIRTUBE 256
#define HR_UNIV_MAXTUBE (HR_UNIV_MIRTUBE * HR_UNIV_MAXMIR)

#define TA_UNIV_MAXMIR  14
#define TA_UNIV_PRO2MIR 4
#define TA_UNIV_MIRTUBE 256
#define TA_UNIV_MAXTUBE (TA_UNIV_MIRTUBE * TA_UNIV_MAXMIR)

/* mode constants for dst_open_unit_() */
#define MODE_READ_DST   1
#define MODE_WRITE_DST  2
#define MODE_APPEND_DST 3

/* diagnostic level constants for dst_read_bank_() */
#define DIAG_NONE_DST 0    /* no checking */
#define DIAG_WARN_DST 1    /* check, issue warning, take no action */
#define DIAG_FULL_DST 2    /* check, issue warning, take appropriate action */

/* default bank size */
#define DFLT_DST_BANK_SIZE 100000   /* arbitrary large bank */

/* constants */
#define TRUE  1
#define FALSE 0
#define PI   3.1415926535897931159979634685441851615906
#define D2R  0.0174532925199432954743716805978692718782
#define R2D 57.2957795130823228646477218717336654663086
//#define CSPEED  29.9792458  // (m / 100 ns)
//#define CSPEED  0.299792458 // (m / ns)

/* TA FD site IDs */
#define BR      0
#define LR      1
#define MD      2
#define TL      3

/* Central Laser Facility GPS */
// #define CLF_LATITUDE     0.68586081  // (  39.29693 degrees)
// #define CLF_LONGITUDE  -1.97062944 // (-112.90875 degrees)
// #define CLF_ALTITUDE     1382.0    // meters
// #define CLF_LATITUDE     0.68586060 // (  39.296918 degrees)
// #define CLF_LONGITUDE   -1.97062914 // (-112.908733 degrees)
#define CLF_LATITUDE     0.68586060387114 // (  39.296917936112 degrees)
#define CLF_LONGITUDE   -1.9706291367663 // (-112.90873252222 degrees)
#define CLF_ALTITUDE     1370.046           // meters

// #define BR_LATITUDE       0.683964863
// #define BR_LONGITUDE     -1.967190271
// #define BR_ALTITUDE       1404.0

// #define LR_LATITUDE       0.684307296
// #define LR_LONGITUDE     -1.974342106
// #define LR_ALTITUDE       1554.0

// #define MD_LATITUDE       0.688930674
// #define MD_LONGITUDE     -1.972111401
// #define MD_ALTITUDE       1600.0

// #define TL_LATITUDE       MD_LATITUDE
// #define TL_LONGITUDE      MD_LONGITUDE
// #define TL_ALTITUDE       MD_ALTITUDE

#endif
