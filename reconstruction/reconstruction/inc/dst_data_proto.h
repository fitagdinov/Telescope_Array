/*
 * $Source: /hires_soft/uvm2k/dst/dst_data_proto.h,v $
 * $Log: dst_data_proto.h,v $
 * Last modified 2019/12/14 DI
 * Revision 1.2  1995/03/18 00:35:12  jeremy
 * *** empty log message ***
 *
*/
/* requires prior inclusion of htypes.h */

#ifdef __cplusplus
extern "C" {
#endif
void     dst_init_data();
integer4 dst_byte_order_(integer4 nbyte, integer1 obj[]);
integer4 dst_r4_ntoi_(integer1 obj[]);
integer4 dst_r8_ntoi_(integer1 obj[]);
integer4 dst_r4_iton_(integer1 obj[]);
integer4 dst_r8_iton_(integer1 obj[]);
#ifdef __cplusplus
} /* end extern "C" */
#endif
