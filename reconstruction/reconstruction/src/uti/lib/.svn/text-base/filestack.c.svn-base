/*
 * filestack.c
 *
 * $Source: /sources/c/dst2k-ta
 *
 * $Source: /hires_soft/uvm2k/uti/filestack.c,v $
 * $Log: filestack.c,v $
 *
 * Revision 1.3  2019/08/15 12:05:00  DI
 *
 * Revision 1.2  1996/10/16 21:11:31  jeremy
 * addded countFiles function
 *
 * Revision 1.1  1996/01/17  05:16:21  mjk
 * Initial revision
 *
 *
 * Two simple routines to help parse command lines. Originally
 * written by Jeremy. Turned into a DST library object and made
 * more robust by MJK.
 * 
*/

#include <string.h>
#include <stdlib.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"

/* this pointer points to the file that's being pulled
   so that string sizes as long as computer's memory can 
   hold can be used and clean up is automatically performed 
   by pullFile also. */
static char* filestack_pulled_file = 0;

typedef struct fileQ {
  struct fileQ *next;
  char file[1];
} fileQ;

/* Use static here because we want this to be local to pushFile
   and pullFile only. */

static struct {
  fileQ *first, *last;
} files = {NULL, NULL};


/* Push a filename onto a linked list of filenames */

integer4 pushFile(char *name)
{
  fileQ *fq = (fileQ *)malloc(sizeof(fileQ) + strlen(name));

  if ( fq == NULL ) return MALLOC_ERR;
  else {
    strcpy(fq->file, name);
    fq->next = NULL;
    if (files.last) files.last->next = fq;
    else files.first = fq;
    files.last = fq;
  }
  return SUCCESS;
}


/* Pulls a filename from linked list of filenames. The first file
   pulled is the first file pushed. */

char *pullFile(void)
{
  fileQ *fq;
  if(filestack_pulled_file)
    {
      free(filestack_pulled_file);
      filestack_pulled_file = NULL;
    }
  if ( (fq = files.first) ) {
    if ((files.first = fq->next) == NULL)
      files.last = NULL;
    filestack_pulled_file = (char *)calloc(strlen(fq->file)+1,sizeof(char));
    if (filestack_pulled_file == NULL)
      return NULL;
    strncpy(filestack_pulled_file, fq->file, strlen(fq->file)+1);
    free(fq);
    return filestack_pulled_file;
  }
  return NULL;
}

/* return number of files in queue */

integer4 countFiles(void)
{
  fileQ *fq;
  integer4 cnt = 0;
  for (fq = files.first; fq; fq = fq->next)
    cnt += 1;
  return cnt;
}
