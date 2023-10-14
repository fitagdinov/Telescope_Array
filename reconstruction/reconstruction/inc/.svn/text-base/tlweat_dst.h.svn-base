/* Created 2015/12/21 DI */
/* Last Updated 2020/03/09 DI */

#ifndef _TLWEAT_DST_
#define _TLWEAT_DST_

#define TLWEAT_BANKID		12508
#define TLWEAT_BANKVERSION	0

#ifdef __cplusplus
extern "C" {
#endif
integer4 tlweat_common_to_bank_ ();
integer4 tlweat_bank_to_dst_ (integer4 * NumUnit);
integer4 tlweat_common_to_dst_ (integer4 * NumUnit);   /* combines above 2 */
integer4 tlweat_bank_to_common_ (integer1 * bank);
integer4 tlweat_common_to_dump_ (integer4 * opt1);
integer4 tlweat_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* tlweat_bank_buffer_ (integer4* tlweat_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  
  integer4 part_num;  // Part number
  integer4 yymmdd;    // UTC date that corresponds to part start time 
  integer4 hhmmss;    // UTC hour minute second that corresponds to part start time  
  integer4 dt;        // Duration of the part, [s]
  
  // n e s w o t h
  // n = 1,  0 Clouds North?
  // e = 1,  0 Clouds East?
  // s = 1,  0 Clouds South?
  // w = 1,  0 Clouds West?
  // o = 0 - 4 Overhead cloud thickness? 5 - weat code invalid
  // t = 1,  0 Stars visible? 
  // h = 1,  0 Was it hazy? 2 - can't tell 
  integer4 code;      // 7-digit weather code recorded by runners
  
} tlweat_dst_common;

extern tlweat_dst_common tlweat_;


#endif
