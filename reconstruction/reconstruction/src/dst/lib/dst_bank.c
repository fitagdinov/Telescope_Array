/*
 * base-level bank reading and writing routines for the DST95 package 
 *
 * $Source: /hires_soft/uvm2k/dst/dst_bank.c,v $
 * $Log: dst_bank.c,v $
 *
 * Revision 2.00  2008/02/29           seans
 * repaired deprecated routines, modified handling of zipped files
 * major house-keeping
 *
 * Revision 1.1  1995/03/17  22:41:32  jeremy
 * Initial revision
 *
 * created:  CCJ  07-FEB-1995 ultrix/ALPHA version only...will modify as
 * needed to accommodate other platforms 
 */

#ifdef unix
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>
#include <bzlib.h>

#include "dst_std_types.h"      /* standard derived data types */
#include "dst_err_codes.h"      /* return codes */
#include "dst_pack_proto.h"     /* prototypes for packing/unpacking */
#include "dst_crc_proto.h"	/* CRC checksum routine */

#ifdef INTERCEPT_GZIO
#  include "iomonitor.h"
#endif

/*
 * definitions for INTERNAL USE ONLY 
 */

#include "dst_size_limits.h"

#define BlockLen 32000		/* lenth of tape/file blocks in bytes */

#define StartBl_Res 6
#define EndBl_Phy_Res 6
#define EndBl_Log_Res 2
#define StartBk_Res 6
#define EndBk_Res 6
#define MinWrite 16

#define CLOSED 0		/* unit is closed */
#define READ 1			/* read mode for OPEN */
#define WRITE 2			/* write mode for OPEN */
#define APPEND 3                /* write-append mode for OPEN */

#define OPCODE 96		/* OPCODE prefix */

#define START_BLOCK 97
#define END_BLOCK_LOGICAL  98	/* logical end of block */
#define END_BLOCK_PHYSICAL  99	/* physical end of block */

#define FILLER 100		/* meaningless filler */
#define START_BANK 7		/* start of a new bank */
#define CONTINUE  8		/* continuing a previous unfinished bank */
#define END_BANK  14		/* end of a bank */
#define TO_BE_CONTD 15		/* break in bank--this bank will be
				   continued */

#define STD_IOMODE  0
#define GZ_IOMODE   1
#define BZ2_IOMODE  2

/*
 * declare some external variables to be used ONLY by the bank I/O
 * routines 
 */

integer4        dst_ustat[MAX_DST_FILE_UNITS] = {MAX_DST_FILE_UNITS * CLOSED};
/* status: =0,1,2 for closed, open for read, write */

#ifdef unix /* on UNIX machines, use "open", "read" and "write" */
/* note use of "int" here for the system call!!!! */

long int        dst_fp[MAX_DST_FILE_UNITS];
gzFile          dst_gz_fp[MAX_DST_FILE_UNITS];
int		dst_piped[MAX_DST_FILE_UNITS];
int             dst_iomode[MAX_DST_FILE_UNITS];
pid_t		dst_pid[MAX_DST_FILE_UNITS];

#else /* on non-UNIX machines, use "fopen", "fread" and "fwrite" */
FILE           *dst_fp[MAX_DST_FILE_UNITS];
/* file pointers for the 10 units */
#endif

integer4        dst_nblk[MAX_DST_FILE_UNITS];
/* total number of blocks processed via this unit thus far */

integer4        dst_nbyt[MAX_DST_FILE_UNITS];
/* number of bytes processed in current buffer */

integer1        *dst_buf[MAX_DST_FILE_UNITS];
/* allocate enough buffer space for all the units */

integer1        *dst_iobuf[MAX_DST_FILE_UNITS];

/*
 * function prototypes 
 */

integer4 dst_open_unit_(integer4 *NumUnit, integer1 NameUnit[], integer4 *mode);
integer4 dst_rewind_unit_(integer4 *NumUnit);
integer4 dst_close_unit_(integer4 *NumUnit);
integer4 dst_write_bank_(integer4 *NumUnit, integer4 *LenBank, integer1 Bank[]);
integer4 dst_end_block_(integer4 *NumUnit);
integer4 dst_out_block_(integer4 *NumUnit);
integer4 dst_new_block_(integer4 *NumUnit);
integer4 dst_read_bank_(integer4 *NumUnit, integer4 *DiagLevel,
			integer1 Bank[], integer4 *LenBank,
			integer4 *BankTyp, integer4 *BankVer);
integer4 dst_get_block_(integer4 *NumUnit, integer4 *DiagLevel);
#ifdef unix
size_t dst_Zread(int filedes, integer1 *buffer, size_t nbytes);
#endif

/* ============== */

integer4 dst_rewind_unit_(integer4 *NumUnit)
{

  dst_nblk[*NumUnit] = 0;
  dst_nbyt[*NumUnit] = 0;
  return lseek((long int)dst_fp[*NumUnit],0,SEEK_SET);
}
/* ============== */

integer4 dst_open_unit_(integer4 *NumUnit, integer1 NameUnit[], integer4 *mode)
{

#ifdef unix
#ifdef ultrix
   int             cmode;
#else
   mode_t          cmode;
#endif
   int             oflag=0;
   int             fptemp=0;
   int             piped_temp=0;

   integer1	   suffix[2];
   integer1	   suffix2[4];
   integer1	   suffix3[3];
   int             pd[2];
   pid_t           Zpid=0;
   static integer1 Zcmd[] = "uncompress";
   static integer1 Zcmd4[] = "mycompress.sh";
   static char *Zparam[] = { Zcmd, "-c", NULL, NULL };
   static char *Zparam4[] = { Zcmd4, NULL, NULL };

#else
   size_t          iobuflen;
   integer1        cmode[4];
   FILE           *fptemp;
   integer4        rcode;
#endif
   integer4        i;

   
   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
      return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == READ) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for input\n", *NumUnit);
      return UNIT_USED_FOR_IN;
   } else if (dst_ustat[*NumUnit] == WRITE) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for output\n", *NumUnit);
      return UNIT_USED_FOR_OUT;
   }
   if (*mode != READ && *mode != WRITE && *mode !=APPEND ) {
      fprintf(stderr,
         " ^^^ mode %d is undefined: read=%d, write=%d\n",
	 *NumUnit, READ, WRITE);
      return UNDEFINED_MODE;
   }
   /*
    * now "regularize" the file name by replacing first "whitespace" by
    * "NULL" 
    */

   for (i = 0; NameUnit[i] != ' '
	&& NameUnit[i] != '\n'
	&& NameUnit[i] != '\t'
	&& NameUnit[i] != '\0'; i = i + 1);
   NameUnit[i] = '\0';

   errno=0;
   dst_buf[*NumUnit]=(integer1 *)malloc(BlockLen);
   if (errno != 0) {
      perror("malloc");
      return MALLOC_ERR;
   }
   
#ifdef unix

   strncpy(suffix, &NameUnit[(strlen(NameUnit) > (size_t) 2 ?
                      strlen(NameUnit)-2 : 0) ], 2);
   strncpy(suffix2, &NameUnit[(strlen(NameUnit) > (size_t) 4 ?
                      strlen(NameUnit)-4 : 0) ], 4);
   strncpy(suffix3, &NameUnit[(strlen(NameUnit) > (size_t) 3 ?
                      strlen(NameUnit)-3 : 0) ], 3);

   if ( (*mode == READ) && (strncmp( suffix, ".Z", 2 ) == 0) ) {
     sigset_t set, oset;
     fprintf(stderr, "why are you using .Z compression you silly bastard!\n");

     dst_iomode[*NumUnit] = 0;

      errno = 0;
      /*rcode = pipe(pd); */ 
      if(pipe(pd))
	fprintf(stderr,"WARNING: %s(%d): broken pipe\n",__FILE__,__LINE__);
      Zpid = fork();
      switch (Zpid) {

      default:      /* this part executed by the parent process */
         close(pd[1]);
         fptemp = pd[0];
         piped_temp = 1;
         sleep(1);
         break;
            
      case 0:       /* this part executed by the child process */
         close(pd[0]);
         dup2(pd[1],1);	/* attach stdout */
         // sigblock( 1 << (SIGQUIT - 1));
	 sigaddset(&set,(1<<(SIGQUIT-1)));
	 sigprocmask(SIG_BLOCK,&set,&oset);
         Zparam[2] = NameUnit;
         if ( execvp ( Zcmd, Zparam) ) {
            perror("\nCould not execute: uncompress\n");
            fprintf(stderr, "Check that it is in your path\n");
            fprintf(stderr, "and is executable.\n");
            exit (-1);
         }
         break;
         
      case (-1):
         Zpid = 0;
         perror("fork");
         return FOPEN_ERR;
         break;
      }
   }
   else if ( (*mode == WRITE) && (strncmp( suffix, ".Z", 2 ) == 0) ) {
     sigset_t set, oset;
     fprintf(stderr, "why are you using .Z compression you silly bastard!\n");
     dst_iomode[*NumUnit] = 0;

      errno = 0;
      /* rcode = pipe(pd); */
      if(pipe(pd))
	fprintf(stderr,"WARNING: %s(%d): broken pipe\n",__FILE__,__LINE__);
      Zpid = fork();
      switch (Zpid) {

      default:      /* this part executed by the parent process */
         close(pd[0]);
         fptemp = pd[1];
         piped_temp = 1;
         sleep(1);
         break;
            
      case 0:       /* this part executed by the child process */
         close(pd[1]);
         dup2(pd[0],0);/*	 attach stdin */
         // sigblock( 1 << (SIGQUIT - 1));
	 sigaddset(&set,(1<<(SIGQUIT-1)));
	 sigprocmask(SIG_BLOCK,&set,&oset);
         Zparam4[1] = NameUnit;
         if ( execvp ( Zcmd4, Zparam4) ) {
            perror("\nCould not execute: mycompress.sh\n");
            fprintf(stderr, "Check that it is in your path\n");
            fprintf(stderr, "and is executable.\n");
            exit (-1);
         }
         break;
         
      case (-1):
         Zpid = 0;
         perror("fork");
         return FOPEN_ERR;
         break;
      }
   } 
   else if ( (*mode == WRITE) && (strncmp( suffix3, ".gz", 3 ) == 0) ) {
      errno = 0;
      dst_gz_fp[*NumUnit] = gzopen(NameUnit, "w");
      fptemp = (long int) dst_gz_fp[*NumUnit];
      piped_temp = 0;
      Zpid = getpid();
      dst_iomode[*NumUnit] = GZ_IOMODE;
   } 
   else if ( (*mode == WRITE) && (strncmp( suffix2, ".bz2", 4 ) == 0) ) {
      errno = 0;
      fptemp = (long int) BZ2_bzopen(NameUnit, "w");
      piped_temp = 0;
      Zpid = getpid();
      dst_iomode[*NumUnit] = BZ2_IOMODE;
   } 
   else if ( (*mode == READ) && (strncmp( suffix2, ".bz2", 4 ) == 0) ) {
      errno = 0;
      fptemp = (long int) BZ2_bzopen(NameUnit, "r");
      piped_temp = 0;
      Zpid = getpid();
      dst_iomode[*NumUnit] = BZ2_IOMODE;
   }
   else if ( (*mode == READ) && (strncmp( suffix3, ".gz", 3 ) == 0) ) {
      errno = 0;
      dst_gz_fp[*NumUnit] = gzopen(NameUnit, "r");
      fptemp = (long int) dst_gz_fp[*NumUnit];
      piped_temp = 0;
      Zpid = getpid();
      dst_iomode[*NumUnit] = GZ_IOMODE;
   }
   else {
         
      cmode = 0664; /* -rw-rw-r-- file mode */
      if (*mode == READ) {
         oflag = O_RDONLY;
      } else if (*mode == WRITE) {
         oflag = (O_RDWR | O_CREAT | O_TRUNC);
      } else if (*mode == APPEND) {
         oflag = (O_RDWR | O_APPEND);
      }
      errno=0;
      piped_temp = 0;
      fptemp = open(NameUnit, oflag, cmode);
      if (errno != 0) {
         perror("open");
         return FOPEN_ERR;
      }
      dst_iomode[*NumUnit] = STD_IOMODE;
   }

#else
   if(strncmp(&NameUnit[strlen(NameUnit)>(size_t)4 ? strlen(NameUnit)-4 : 0],".dst",4))
     {
       fprintf(stderr,"ERROR: dst_open_unit_: ONLY STANDARD DST IO SUPPORTED ON NON UNIX FAMILY PLATFORMS!\n");
       fprintf(stderr,"Hint: if you have a .gz or .bz2 file then unpack it first, with gunzip (for .gz\n");
       fprintf(stderr,"files or bunzip2 (for .bz2 files) so that your input file has .dst extension\n");
       return GET_BLOCK_READ_ERROR;
     }

   if (*mode == READ) {
      cmode[0] = 'r';
      cmode[1]='\0';   
   } else if (*mode == WRITE) {
      cmode[0] = 'w';
      cmode[1]='\0';   
   } else if (*mode == APPEND) {
      cmode[0] = 'a';
      cmode[1]='\0';   
   }
   
   errno=0;
   fptemp = fopen(NameUnit, cmode);
   if (errno != 0) {
      perror("fopen");
      return FOPEN_ERR;
   }

/*
 * next block of code does not work on VAX...it is specific to tape
 * operations under UNIX...will exclude for VAX compilation.
 */

#ifndef VAX
   errno=0;
   if(*mode == WRITE) {
     iobuflen=BlockLen;;
   } else {
     iobuflen=BlockLen*2;
   }
   dst_iobuf[*NumUnit]=(integer1 *)malloc(iobuflen);
   if (errno != 0) {
      perror("malloc");
      return MALLOC_ERR;
   }
   
   errno=0;
   rcode=setvbuf(fptemp,dst_iobuf[*NumUnit],_IOFBF,iobuflen);
   if (rcode !=0 ) {
      fprintf(stderr,
         " ^^^ n_unit_ failed: setvbuf returns code %d\n",
         rcode);
      return SETVBUF_ERR;
   }
   if (errno != 0) {
      perror("setvbuf");
      return SETVBUF_ERR;
   }
#endif
/* end of special VAX exclusion */

#endif
   
   if (*mode == APPEND) {
      *mode = WRITE;
   }

#ifdef unix
   dst_piped[*NumUnit] = piped_temp;
   dst_pid[*NumUnit] = Zpid;
#endif
   dst_fp[*NumUnit] = fptemp;
   dst_ustat[*NumUnit] = (*mode);

   dst_nblk[*NumUnit] = 0;
   dst_nbyt[*NumUnit] = 0;
   return SUCCESS;
}

/* ============== */

integer4 dst_close_unit_(integer4 *NumUnit)
{
   integer4        rcode;

   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit,MAX_DST_FILE_UNITS - 1);
      return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is already closed\n", *NumUnit);
      return UNIT_NOT_OPEN;
   }

   if (dst_ustat[*NumUnit] == WRITE) {
      rcode = dst_end_block_(NumUnit);
      if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_close_unit_ failed: EndBlock error: %d\n",
            rcode);
	 return END_BLOCK_ERR;
      }
      rcode = dst_out_block_(NumUnit);
      if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_close_unit_ failed: OutBlock error: %d\n",
	    rcode);
	 return OUT_BLOCK_ERR;
      }
   }

#ifdef unix
   switch ( dst_iomode[*NumUnit] ) {
     int pidstat;
   case GZ_IOMODE:
     rcode = gzclose(dst_gz_fp[*NumUnit]);
     // not important, this is an indicative of the fact
     // that deflate was inefficient at the time of closing the gz file
     // this happens when one reads through a bunch of large gzipped DST files
     // and one should not call this one a failure of DST readout system
     if(rcode == Z_BUF_ERROR)
       rcode = 0;
     break;

   case BZ2_IOMODE:
     BZ2_bzclose((BZFILE *)dst_fp[*NumUnit]);
     rcode = 0;
     break;

   case STD_IOMODE:
   default:
     rcode = close(dst_fp[*NumUnit]);
     if(dst_piped[*NumUnit]==1)waitpid(dst_pid[*NumUnit], &pidstat, 0);
     ;;
   }

   if (rcode != 0) {
     perror("close");
     return FCLOSE_ERR;
   }

#else
   rcode = fclose(dst_fp[*NumUnit]);
   if (rcode != 0) {
      perror("fclose");
      return FCLOSE_ERR;
   }
#endif

   dst_ustat[*NumUnit] = CLOSED;

   dst_nblk[*NumUnit] = 0;
   dst_nbyt[*NumUnit] = 0;
   free(dst_buf[*NumUnit]);
   free(dst_iobuf[*NumUnit]);
   return SUCCESS;
}

/* ============== */

integer4 dst_write_bank_(integer4 *NumUnit, integer4 *LenBank, integer1 Bank[])
{
   integer4        i;
   integer4        rcode;

   integer4        nobj = 1;

   integer4        out_bytes = 0;
   integer4        start_b_out = 0;
   integer4        end_b_out = 0;
   integer4        num_b_out = 0;

   integer4        reserve;
   integer4        chksum;
   integer4        MaxLen;

   /*
    * first check if unit is legal and opened for write 
    */
   
   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
     return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == READ) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for input\n", *NumUnit);
      return UNIT_USED_FOR_IN;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is not yet opened \n", *NumUnit);
      return UNIT_NOT_OPEN;
   }
   reserve = StartBk_Res + EndBk_Res + EndBl_Phy_Res + EndBl_Log_Res;
   chksum = dst_crc_ccitt_(LenBank, Bank);

   /*
    * If this is the first time this unit is being used, start a new block
    * before trying anything else. 
    */

   if (dst_nblk[*NumUnit] == 0 && dst_nbyt[*NumUnit] == 0) {
      rcode = dst_new_block_(NumUnit);
      if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_write_bank_ failed: NewBlock error: %d\n",
	    rcode);
	 return NEW_BLOCK_ERR;
      }
   }
   while (out_bytes < *LenBank) {

      /*
       * If there is less than 16 bytes of available space for writing,
       * then it is simply not worth trying to do anything with it... fill
       * the block with "FILLER", terminate it, write it out and start a
       * new one. 
       */

      if (BlockLen - dst_nbyt[*NumUnit] - reserve < MinWrite) {

	 rcode = dst_end_block_(NumUnit);
	 if (rcode != SUCCESS) {
	    fprintf(stderr,
	       " ^^^ dst_write_bank_ failed: EndBlock error: %d\n",
	       rcode);
	    return END_BLOCK_ERR;
	 }
	 rcode = dst_out_block_(NumUnit);
	 if (rcode != SUCCESS) {
	    fprintf(stderr,
	        " ^^^ dst_write_bank_ failed: OutBlock error: %d\n",
		rcode);
	    return OUT_BLOCK_ERR;
	 }
	 rcode = dst_new_block_(NumUnit);
	 if (rcode != SUCCESS) {
	    fprintf(stderr,
	       " ^^^ dst_write_bank_ failed: NewBlock error: %d\n",
	       rcode);
	    return NEW_BLOCK_ERR;

	 }
      }
      if (BlockLen - dst_nbyt[*NumUnit] - reserve >= *LenBank - out_bytes) {
	 num_b_out = (*LenBank - out_bytes);
      } else {
	 num_b_out = BlockLen - dst_nbyt[*NumUnit] - reserve;
      }

      end_b_out = start_b_out + num_b_out;

      *(dst_buf[*NumUnit] + dst_nbyt[*NumUnit]) = OPCODE;
      dst_nbyt[*NumUnit]++;

      if (start_b_out == 0) {
	 *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = START_BANK;
      } else {
	 *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = CONTINUE;
      }

      dst_nbyt[*NumUnit]++;

      MaxLen=BlockLen;
      rcode = dst_packi4_(&num_b_out, &nobj,
	      dst_buf[*NumUnit], &dst_nbyt[*NumUnit], &MaxLen);

      if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_write_bank_ failed: packi4 error: %d\n", rcode);
	 return PACKI4_FAIL;
      }
      for (i = start_b_out; i < end_b_out; i = i + 1) {
	 *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = Bank[i];
	 dst_nbyt[*NumUnit]++;
      }

      out_bytes = out_bytes + num_b_out;
      start_b_out = start_b_out + num_b_out;

      *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = OPCODE;
      dst_nbyt[*NumUnit]++;

      if (out_bytes == *LenBank) {
	 *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = END_BANK;
      } else {
	 *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = TO_BE_CONTD;
      }

      dst_nbyt[*NumUnit]++;

      MaxLen=BlockLen;
      rcode = dst_packi4_(&chksum, &nobj,
	    dst_buf[*NumUnit], &dst_nbyt[*NumUnit],&MaxLen);

      if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_write_bank_ failed: packi4 error: %d\n", rcode);
	 return PACKI4_FAIL;

      }
   }

   return SUCCESS;

}

/* ============== */

integer4
dst_end_block_(integer4 *NumUnit)
{
   integer4        i;
   integer4        len;
   integer4        chksum;
   integer4        nobj = 1;
   integer4        rcode;
   integer4        MaxLen;

   /*
    * first check if unit is legal and opened for write 
    */
   
   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
     return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == READ) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for input\n", *NumUnit);
      return UNIT_USED_FOR_IN;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is not yet opened \n", *NumUnit);
      return UNIT_NOT_OPEN;
   }
   *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = OPCODE;
   dst_nbyt[*NumUnit]++;
   *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = END_BLOCK_LOGICAL;
   dst_nbyt[*NumUnit]++;

   for (i = dst_nbyt[*NumUnit]; i < BlockLen - EndBl_Phy_Res; i = i + 1) {
      *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = FILLER;
      dst_nbyt[*NumUnit]++;
   }

   *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = OPCODE;
   dst_nbyt[*NumUnit]++;
   *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = END_BLOCK_PHYSICAL;
   dst_nbyt[*NumUnit]++;

   len = BlockLen - 4;
   chksum = dst_crc_ccitt_(&len, dst_buf[*NumUnit]);

   MaxLen=BlockLen;
   rcode = dst_packi4_(&chksum, &nobj,
		       dst_buf[*NumUnit], &dst_nbyt[*NumUnit], &MaxLen);

   if (rcode != SUCCESS) {
      fprintf(stderr,
         " ^^^ dst_end_block_ failed: packi4_ error: %d\n", rcode);
      return PACKI4_FAIL;
   }
   dst_nblk[*NumUnit]++;
   return SUCCESS;

}

/* ============== */

integer4 dst_out_block_(integer4 *NumUnit)
{
   size_t          num_items = BlockLen;
   size_t          num_written;

   /*
    * first check if unit is legal and opened for write 
    */

   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
      return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == READ) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for input\n", *NumUnit);
      return UNIT_USED_FOR_IN;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is not yet opened \n", *NumUnit);
      return UNIT_NOT_OPEN;
   }
   errno=0;
#ifdef unix
   switch ( dst_iomode[*NumUnit] ) {
   case GZ_IOMODE:
     num_written = gzwrite(dst_gz_fp[*NumUnit],(void *)dst_buf[*NumUnit], num_items);
     break;

   case BZ2_IOMODE:
     num_written = BZ2_bzwrite((BZFILE *)dst_fp[*NumUnit],(void *)dst_buf[*NumUnit], num_items);
     break;

   case STD_IOMODE:
   default:
     num_written = write(dst_fp[*NumUnit],(void *)dst_buf[*NumUnit],num_items);
     ;;
   }

#else
   num_written = fwrite(dst_buf[*NumUnit],1,num_items,dst_fp[*NumUnit]);
#endif

   if (errno != 0) {
#ifdef unix
      perror("write");
#else
      perror("fwrite");
#endif
      fprintf(stderr,
         " ^^^ dst_out_block_ failed: only %ld bytes written\n"
	      ,(long int)num_written);
      return FWRITE_ERR;
   }
   return SUCCESS;
}

/* ============== */

integer4
dst_new_block_(integer4 *NumUnit)
{
   integer4        nobj = 1;
   integer4        rcode;
   integer4        MaxLen;

   /*
    * first check if unit is legal and opened for write 
    */
   
   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
     return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == READ) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for input\n", *NumUnit);
      return UNIT_USED_FOR_IN;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is not yet opened \n", *NumUnit);
      return UNIT_NOT_OPEN;
   }
   dst_nbyt[*NumUnit] = 0;

   *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = OPCODE;
   dst_nbyt[*NumUnit]++;
   *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) = START_BLOCK;
   dst_nbyt[*NumUnit]++;

   MaxLen=BlockLen;
   rcode = dst_packi4_(&dst_nblk[*NumUnit], &nobj,
	    dst_buf[*NumUnit], &dst_nbyt[*NumUnit], &MaxLen);
   if (rcode != SUCCESS) {
      fprintf(stderr,
         " ^^^ dst_new_block_ failed: packi4_ error: %d\n", rcode);
      return PACKI4_FAIL;
   }
   return SUCCESS;
}

/* ============== */

integer4 dst_read_bank_(integer4 *NumUnit, integer4 *DiagLevel,
			integer1 Bank[], integer4 *LenBank,
			integer4 *BankTyp, integer4 *BankVer)
{
   integer4        i;
   integer4        rcode;
   integer4        num_b_read;
   integer4        skipped;
   integer4        crc;
   integer4        chksum;

   integer4        nobj = 1;
   integer4        started = 0;
   integer4        finished = 0;

   integer4        dummy;
   integer4        MaxLen;
   integer4        status;

   *LenBank = 0;

   /*
    * first check if unit is legal and opened for read 
    */

   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
     return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == WRITE) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for output\n", *NumUnit);
      return UNIT_USED_FOR_OUT;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is not yet opened \n", *NumUnit);
      return UNIT_NOT_OPEN;
   }
   if (dst_nblk[*NumUnit] == 0 && dst_nbyt[*NumUnit] == 0) {
      rcode = dst_get_block_(NumUnit, DiagLevel);
      if (rcode == END_OF_FILE) {
         return END_OF_FILE;
      } else if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_read_bank_ failed: GetBlock error: %d\n", rcode);
	 return rcode;
      }
   }
   status = SUCCESS;
   while (finished == 0) {

      skipped=0;
      if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) != OPCODE) {
	 dst_nbyt[*NumUnit]++;
	 skipped++;
      }
      if( skipped > 0 ) {
         fprintf(stderr,
            " ^^^ dst_read_bank_ warning: skipped %d bytes\n",skipped);
      }
      dst_nbyt[*NumUnit]++;

      if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == END_BLOCK_LOGICAL ||
	  *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == END_BLOCK_PHYSICAL) {

	 rcode = dst_get_block_(NumUnit, DiagLevel);
         if (rcode == END_OF_FILE) {
            return END_OF_FILE;
	 } else if (rcode != SUCCESS) {
	    fprintf(stderr,
	       " ^^^ dst_read_bank_ failed: GetBlock error: %d\n",
	       rcode);
	    return GET_BLOCK_ERR;
	 }
	 continue;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == START_BANK) {

	 if (started != 0) {
	    fprintf(stderr,
	       " ^^^ dst_read_bank_ warning: unexpected start_of_bank\n");
	    fprintf(stderr,
	       "   current bank data released and new bank started\n");
	    *LenBank = 0;
	 }
	 started = 1;
	 dst_nbyt[*NumUnit]++;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == CONTINUE) {

	 if (started == 0) {
	    fprintf(stderr,
	       " ^^^ dst_read_bank_ warning: unexpected continue\n");
	    fprintf(stderr,
	       "   remainder of continued bank will be skipped\n");
	    *LenBank = 0;
	    dst_nbyt[*NumUnit]++;
	    status = GET_BANK_UNEXP_CONT;
	    continue;
	 } else {
	    started = 1;
	    dst_nbyt[*NumUnit]++;
	 }

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == END_BANK) {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unexpected end_bank\n");
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_UNEXP_END_BNK;
	 continue;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == TO_BE_CONTD) {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unexpected to_be_contd\n");
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_UNEXP_TBCNT;
	 continue;

      } else {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unknown opcode: %d\n",
	    *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) );
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_UNKWN_OPCODE;
	 continue;

      }

      /*
       * we have what appears to be a well-defined bank segment: 
       */

      MaxLen=BlockLen;
      rcode = dst_unpacki4_(&num_b_read, &nobj, 
              dst_buf[*NumUnit], &dst_nbyt[*NumUnit], &MaxLen);
      if (rcode != SUCCESS) {
	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unpacki4 error: %d\n", rcode);
	 fprintf(stderr,
	    "     cannot unpack bank segment length\n");
	 fprintf(stderr,
	    "     current bank data released\n");
	 *LenBank = 0;
	 started = 0;
	 status = GET_BANK_LENGTH_ERROR;
	 continue;
      }
      for (i = 0; i < num_b_read; i = i + 1) {
	 Bank[*LenBank] = *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]);
	 dst_nbyt[*NumUnit]++;
	 (*LenBank) = (*LenBank) + 1;
      }

      if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) != OPCODE) {
	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: expected OPCODE not found\n");
	 fprintf(stderr,
	    "   current bank data released\n");
	 *LenBank = 0;
	 started = 0;
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_OPCODE_ERROR;
	 continue;
      }
      dst_nbyt[*NumUnit]++;

      if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == END_BLOCK_LOGICAL ||
	  *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == END_BLOCK_PHYSICAL) {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unexpected end_of_block\n");
	 fprintf(stderr,
	    "   current bank data released\n");
	 *LenBank = 0;
	 started = 0;
	 dst_nbyt[*NumUnit]--;
	 status = GET_BANK_UNEXP_END_BLK;
	 continue;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == START_BANK) {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unexpected start_bank\n");
	 fprintf(stderr,
	    "   current bank data released\n");
	 *LenBank = 0;
	 started = 0;
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_UNEXP_START;
	 continue;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == CONTINUE) {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unexpected continue\n");
	 fprintf(stderr,
	    "   current bank data released\n");
	 *LenBank = 0;
	 started = 0;
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_UNEXP_CONT;
	 continue;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == END_BANK) {

	 dst_nbyt[*NumUnit]++;

	 if (*DiagLevel > 0) {

            MaxLen=BlockLen;
            rcode = dst_unpacki4_(&crc, &nobj, 
                  dst_buf[*NumUnit], &dst_nbyt[*NumUnit], &MaxLen);
	    if (rcode != SUCCESS) {
	       fprintf(stderr,
	          " ^^^ dst_read_bank_ warning: unpacki4 error: %d\n",
		  rcode);
	       fprintf(stderr,
	          "     cannot unpack CRC for bank\n");
	       if (*DiagLevel > 1) {
		  fprintf(stderr,
		     "     current bank data released\n");
		  *LenBank = 0;
		  started = 0;
		  status = GET_BANK_CRC_ERROR;
		  continue;
	       }
	    }
	    chksum = dst_crc_ccitt_(LenBank, Bank);
	    if (crc != chksum) {
	       fprintf(stderr,
	          " ^^^ dst_read_bank_ warning: CRC checksum error\n");
	       if (*DiagLevel > 1) {
		  fprintf(stderr,
		     "     current bank data released\n");
		  *LenBank = 0;
		  started = 0;
		  status = GET_BANK_CRC_ERROR;
		  continue;
	       }
	    }
	 } else {
	    dst_nbyt[*NumUnit]=dst_nbyt[*NumUnit]+4;
	 }
	 
	 dummy = 0;

         MaxLen=BlockLen;
         rcode = dst_unpacki4_(BankTyp, &nobj, Bank, &dummy, &MaxLen);
	 if (rcode != SUCCESS) {
	    fprintf(stderr,
	       " ^^^ dst_read_bank_ warning: unpacki4 error: %d\n",
	       rcode);
	    fprintf(stderr,
	       "     cannot unpack bank type\n");
	    fprintf(stderr,
	       "     current bank data released\n");
	    *LenBank = 0;
	    started = 0;
	    status = GET_BANK_UNKWN_BANK;
	    continue;
	 }

         MaxLen=BlockLen;
         rcode = dst_unpacki4_(BankVer, &nobj, Bank, &dummy, &MaxLen);
	 if (rcode != SUCCESS) {
	    fprintf(stderr,
	       " ^^^ dst_read_bank_ warning: unpacki4 error: %d\n",
	       rcode);
	    fprintf(stderr,
	       "     cannot unpack bank version\n");
	    fprintf(stderr,
	       "     current bank data released\n");
	    *LenBank = 0;
	    started = 0;
	    status = GET_BANK_VERSION_ERROR;
	    continue;
	 }
	 finished = 1;

      } else if (*(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) == TO_BE_CONTD) {

         dst_nbyt[*NumUnit]=dst_nbyt[*NumUnit]+5;
	 finished = 0;

      } else {

	 fprintf(stderr,
	    " ^^^ dst_read_bank_ warning: unknown opcode: %d\n",
            *(dst_buf[*NumUnit]+dst_nbyt[*NumUnit]) );
	 fprintf(stderr,
	    "   current bank data released\n");
	 *LenBank = 0;
	 started = 0;
	 dst_nbyt[*NumUnit]++;
	 status = GET_BANK_UNKWN_OPCODE;
	 continue;

      }
   }

   return (status);
}

/* ============== */

integer4 dst_get_block_(integer4 *NumUnit, integer4 *DiagLevel)
{

   size_t          num_items = BlockLen;
   size_t          num_read;

   integer4        got_block = 0;
   integer4        nobj = 1;

   integer4        rcode;
   integer4        len;
   integer4        crc;
   integer4        chksum;
   integer4        block_no;
   
   integer4        MaxLen;
   integer4        status;

   /*
    * first check if unit is legal and opened for read 
    */
   
   if (*NumUnit < 0 || *NumUnit >= MAX_DST_FILE_UNITS) {
     fprintf(stderr,
	     " ^^^ unit %d is out of allowed range [0-%d]\n", *NumUnit, MAX_DST_FILE_UNITS - 1);
     return UNIT_OUT_OF_RANGE;
   } else if (dst_ustat[*NumUnit] == WRITE) {
      fprintf(stderr,
         " ^^^ unit %d is already allocated for output\n", *NumUnit);
      return UNIT_USED_FOR_OUT;
   } else if (dst_ustat[*NumUnit] == CLOSED) {
      fprintf(stderr,
         " ^^^ unit %d is not yet opened \n", *NumUnit);
      return UNIT_NOT_OPEN;
   }
   status = SUCCESS;
   while (got_block == 0) {

#ifdef unix
      if (dst_piped[*NumUnit] == 1) {
         num_read = dst_Zread(dst_fp[*NumUnit], dst_buf[*NumUnit],
                       num_items);
         if (num_read < BlockLen) {
            fprintf(stderr,
                " $$$ dst_get_block_ : End of input file reached\n");
            return END_OF_FILE;
         }          

      } else {
         errno=0;
	 switch ( dst_iomode[*NumUnit] ) {
	 case GZ_IOMODE:
	   num_read = gzread(dst_gz_fp[*NumUnit], 
			     (void *)dst_buf[*NumUnit], num_items);
	   break;

	 case BZ2_IOMODE:
	   num_read = BZ2_bzread((BZFILE *)dst_fp[*NumUnit], 
			     (void *)dst_buf[*NumUnit], num_items);
	   break;

	 case STD_IOMODE:
	 default:
	   num_read = read(dst_fp[*NumUnit], (void *)dst_buf[*NumUnit],
			   num_items);
	   ;;
	 }

         if (errno == 0 && num_read < BlockLen) {
            fprintf(stderr,
                " $$$ dst_get_block_ : End of input file reached\n");
            return END_OF_FILE;
         }          

         if (errno != 0) {
            perror("read");
	    fprintf(stderr,
	       " ^^^ dst_get_block_ warning: %ld bytes read\n",
		    (long int)num_read);
	    if (num_read != BlockLen) {
	       fprintf(stderr,
	          "dropping block\n");
	       dst_nblk[*NumUnit]++;
	       status = GET_BLOCK_BYTE_CNT;
	       continue;
	    }
         }
      }
#else 
      errno=0;
      num_read = fread(dst_buf[*NumUnit],1,num_items, dst_fp[*NumUnit]);
      if (errno == 0 && num_read < BlockLen) {
          fprintf(stderr,
             " $$$ dst_get_block_ : End of input file reached\n");
          return END_OF_FILE;
      }          

      if (errno != 0) {
         perror("fread");
	 fprintf(stderr,
	    " ^^^ dst_get_block_ warning: %ld bytes read\n",
	    num_read);
	 if (num_read != BlockLen) {
	    fprintf(stderr,
	       "dropping block\n");
	    status = GET_BLOCK_BYTE_CNT;
	    dst_nblk[*NumUnit]++;
	    continue;
	 }
      }

#endif

      if (*DiagLevel > 0) {

	 len = BlockLen - 4;
         MaxLen=BlockLen;
         rcode = dst_unpacki4_(&crc, &nobj, 
                 dst_buf[*NumUnit], &len, &MaxLen);
	 if (rcode != SUCCESS) {

	    fprintf(stderr,
	       " ^^^ dst_get_block_ warning: unpacki4 error: %d\n",
	       rcode);
	    fprintf(stderr,
	       "     cannot unpack CRC for block\n");
	    status = GET_BLOCK_CRC_ERROR;
	    if (*DiagLevel > 1) {
	       fprintf(stderr,
	          "dropping block\n");
	       dst_nblk[*NumUnit]++;
	       continue;
	    }
	 }

	 len = BlockLen - 4;
	 chksum = dst_crc_ccitt_(&len, dst_buf[*NumUnit]);
	 if (crc != chksum) {
	    fprintf(stderr,
	       " ^^^ dst_get_block_ warning: CRC checksum error\n");
	    status = GET_BLOCK_CRC_ERROR;
	    if (*DiagLevel > 1) {
	       fprintf(stderr,
	          "dropping block\n");
	       dst_nblk[*NumUnit]++;
	       continue;
	    }
	 }

	 if (*(dst_buf[*NumUnit]) != OPCODE ||
	     *(dst_buf[*NumUnit]+1) != START_BLOCK) {
	    fprintf(stderr,
	       " ^^^ dst_get_block_ warning: corrupted block header\n");
	    status = GET_BLOCK_CORRUPT_HDR;
	    if (*DiagLevel > 1) {
	       fprintf(stderr,
	          "dropping block\n");
	       dst_nblk[*NumUnit]++;
	       continue;
	    }
	 }

         len=2;
         MaxLen=BlockLen;
         rcode = dst_unpacki4_(&block_no, &nobj, 
              dst_buf[*NumUnit], &len, &MaxLen);
	 if (rcode != SUCCESS) {
	    fprintf(stderr,
	       " ^^^ dst_get_block_ warning: unpacki4 error: %d\n",
	       rcode);
	    fprintf(stderr,
	       "     cannot unpack block no.\n");
	    status = GET_BLOCK_READ_ERROR;
	    if (*DiagLevel > 1) {
	       fprintf(stderr,
	          "dropping block\n");
	       dst_nblk[*NumUnit]++;
	       continue;
	    }
	 }

	 if (block_no != dst_nblk[*NumUnit]) {
	    fprintf(stderr,
	       " ^^^ dst_get_block_ warning: block %d out of sequence\n",
	       block_no);
	    dst_nblk[*NumUnit] = block_no;
	    status = GET_BLOCK_OUT_OF_SEQ;
	 }
      }
      dst_nblk[*NumUnit]++;
      got_block = 1;
   }

   dst_nbyt[*NumUnit] = 6;
   return (status);
}

#ifdef unix
size_t dst_Zread(int filedes, integer1 *buffer, size_t nbytes)
{
  integer4 ntemp, nwant;
  integer4 errcnt;
  size_t nread; 
   nread=0;
   errcnt=0;
   while (nread < nbytes) {
      nwant = nbytes - nread;
      errno=0;
      ntemp = read(filedes, (void *)&buffer[nread], nwant);
/*       printf(" %d",ntemp); */

      if (errno !=0) {
         fprintf(stderr,"READ error from pipe");
         nread = 0;
         break;
      }
      if (ntemp == 0) {
         errcnt++;
         if (errcnt > 5) break;
      } else {
         errcnt = 0;
         nread += ntemp;
/*         if (ntemp < nwant) sleep(1); */
      }
   }
      
   return nread;
}   

#endif
