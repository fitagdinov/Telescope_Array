/*
 * filestack.h
 *
 * $Source: /hires_soft/uvm2k/uti/filestack.h,v $
 * $Log: filestack.h,v $
 * Revision 1.3  2001/03/14 20:45:06  reil
 * Added #ifdef __cplusplus to allow function use in cpp
 *
 * Revision 1.2  1996/10/16 21:11:51  jeremy
 * fixed pushFile prototype.
 * added countFiles function.
 *
 * Revision 1.1  1996/01/17  05:16:31  mjk
 * Initial revision
 *
 *
 * Prototypes for filestack routines
 *
*/
#ifdef __cplusplus
extern "C" integer4 pushFile(char *name);
extern "C" char *pullFile(void);
extern "C" integer4 countFiles(void);
#else
integer4 pushFile(char *name);
char *pullFile(void);
integer4 countFiles(void);
#endif
