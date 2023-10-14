/* geotl_dst.h

 * Created 2011/08/30 TAS
 
 */

#ifndef _GEOTL_DST_
#define _GEOTL_DST_

#include "geofd_dst.h"

#define GEOTL_BANKID		12501
#define GEOTL_BANKVERSION	002

// #define GEOTL_MAXMIR	14
#define GEOTL_MAXMIR   10
#define GEOTL_MIRTUBE	256

#define GEOTL_ROW	16	// Number of rows of PMTs
#define GEOTL_COL	16	// Number of columns of PMTs
#define GEOTL_SEGMENT    4    // Number of mirror segments

#define GEOTL_PMT_GAP   0.002

// #define TL_LATITUDE       0.688930679 // old camera 0
#define TL_LATITUDE       0.688931877 // new camera 0 = old camera 2
// #define TL_LONGITUDE     -1.972117259
#define TL_LONGITUDE     -1.972116993
// #define TL_ALTITUDE       1589.144
#define TL_ALTITUDE       1589.166

#define GEOTL_LATITUDE  TL_LATITUDE
#define GEOTL_LONGITUDE TL_LONGITUDE
#define GEOTL_ALTITUDE  TL_ALTITUDE




#ifdef __cplusplus
extern "C" {
#endif
integer4 geotl_common_to_bank_();
integer4 geotl_bank_to_dst_(integer4 *NumUnit);
integer4 geotl_common_to_dst_(integer4 *NumUnit); // combines above 2
integer4 geotl_bank_to_common_(integer1 *bank);
integer4 geotl_common_to_dump_(integer4 *opt1) ;
integer4 geotl_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* geotl_bank_buffer_ (integer4* geotl_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef geofd_dst_common geotl_dst_common;
extern geotl_dst_common geotl_;

#endif
