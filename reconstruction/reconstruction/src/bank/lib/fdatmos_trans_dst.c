#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdatmos_trans_dst.h"

fdatmos_trans_dst_common fdatmos_trans_;
static integer4 fdatmos_trans_blen = 0;
static integer4 fdatmos_trans_maxlen = sizeof(integer4)*2 + sizeof(fdatmos_trans_dst_common);
static integer1* fdatmos_trans_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdatmos_trans_bank_buffer_ (integer4* fdatmos_trans_bank_buffer_size)
{
  (*fdatmos_trans_bank_buffer_size) = fdatmos_trans_blen;
  return fdatmos_trans_bank;
}



static void fdatmos_trans_bank_init();
static integer4 fdatmos_trans_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdatmos_trans_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdatmos_trans_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdatmos_trans_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   if(version >= 4){
      rcode += dst_unpacki4_(&fdatmos_trans_.uniqID,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   }else{
      fdatmos_trans_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdatmos_trans_.dateFrom,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpacki4_(&fdatmos_trans_.dateTo,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpacki4_(&fdatmos_trans_.nItem,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(&fdatmos_trans_.normHeight,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(&fdatmos_trans_.availableHeight,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(&fdatmos_trans_.lowestHeight,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpacki4_(&fdatmos_trans_.badFlag,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpacki4_(&fdatmos_trans_.lowerFlag,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpacki4_(&fdatmos_trans_.cloudFlag,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);

   nobj=fdatmos_trans_.nItem;

   rcode += dst_unpackr4_(fdatmos_trans_.height,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.measuredAlpha,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.measuredAlphaError_P,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.measuredAlphaError_M,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.rayleighAlpha,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.rayleighAlphaError,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.mieAlpha,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.mieAlphaError_P,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.mieAlphaError_M,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.vaod,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.vaodError_P,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_unpackr4_(fdatmos_trans_.vaodError_M,&nobj,bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);

   return rcode;
}

/* write method */
integer4 fdatmos_trans_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdatmos_trans_ -> unit
   integer4 rcode=0;
   if ((rcode += fdatmos_trans_common_to_bank_()) != 0){
      fprintf (stderr, "fdatmos_trans_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdatmos_trans_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdatmos_trans_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdatmos_trans_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdatmos_trans_dst\n");
   fprintf(fp,"uniq ID: %d ",fdatmos_trans_.uniqID);
   if(fdatmos_trans_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdatmos_trans_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");
   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(fdatmos_trans_.dateFrom,dateFromLine);
   convertSec2DateLine(fdatmos_trans_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
   fprintf(fp,"nItem:%d normHeight:%f availableHeight:%f lowestHeight:%f\n",fdatmos_trans_.nItem,fdatmos_trans_.normHeight,fdatmos_trans_.availableHeight,fdatmos_trans_.lowestHeight);
   fprintf(fp,"badFlag:%d lowerFlag:%d cloudFlag:%d\n",fdatmos_trans_.badFlag,fdatmos_trans_.lowerFlag,fdatmos_trans_.cloudFlag);

   if(fdatmos_trans_.nItem==0){
      return 0;
   }

   fprintf(fp,"height measuredAlpha measuredAlphaE+ measuredAlphaE- RayleightAlpha RayleightAlphaE MieAlpha MieAlphaE+ MieAlphaE- VAOD VAODE+ VAODE-\n");
   integer4 i;
   for(i=0;i<fdatmos_trans_.nItem;i++){
      fprintf(fp,"%f %e %e %e %e %e %e %e %e %e %e %e\n",fdatmos_trans_.height[i],fdatmos_trans_.measuredAlpha[i],fdatmos_trans_.measuredAlphaError_P[i],fdatmos_trans_.measuredAlphaError_M[i],fdatmos_trans_.rayleighAlpha[i],fdatmos_trans_.rayleighAlphaError[i],fdatmos_trans_.mieAlpha[i],fdatmos_trans_.mieAlphaError_P[i],fdatmos_trans_.mieAlphaError_M[i],fdatmos_trans_.vaod[i],fdatmos_trans_.vaodError_P[i],fdatmos_trans_.vaodError_M[i]);
   }

   return 0;
}

/* static method */
static void fdatmos_trans_bank_init(){
   fdatmos_trans_bank = (integer1*) calloc(fdatmos_trans_maxlen,sizeof(integer1));
   if(fdatmos_trans_bank==NULL){
      fprintf (stderr,"fdatmos_trans_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdatmos_trans_common_to_bank_(){
   static integer4 id = FDATMOS_TRANS_BANKID;
   static integer4 ver = FDATMOS_TRANS_BANKVERSION;
   integer4 rcode,nobj;
   if(fdatmos_trans_bank == NULL){
      fdatmos_trans_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdatmos_trans_blen,&fdatmos_trans_maxlen,fdatmos_trans_bank);

   nobj=1;
   rcode += dst_packi4_(&fdatmos_trans_.uniqID,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packi4_(&fdatmos_trans_.dateFrom,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packi4_(&fdatmos_trans_.dateTo,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packi4_(&fdatmos_trans_.nItem,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(&fdatmos_trans_.normHeight,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(&fdatmos_trans_.availableHeight,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(&fdatmos_trans_.lowestHeight,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packi4_(&fdatmos_trans_.badFlag,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packi4_(&fdatmos_trans_.lowerFlag,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packi4_(&fdatmos_trans_.cloudFlag,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);

   nobj=fdatmos_trans_.nItem;

   rcode += dst_packr4_(fdatmos_trans_.height,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.measuredAlpha,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.measuredAlphaError_P,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.measuredAlphaError_M,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.rayleighAlpha,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.rayleighAlphaError,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.mieAlpha,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.mieAlphaError_P,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.mieAlphaError_M,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.vaod,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.vaodError_P,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);
   rcode += dst_packr4_(fdatmos_trans_.vaodError_M,&nobj,fdatmos_trans_bank,&fdatmos_trans_blen,&fdatmos_trans_maxlen);

   return rcode;
}

static integer4 fdatmos_trans_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdatmos_trans_blen,fdatmos_trans_bank);
   free(fdatmos_trans_bank);
   fdatmos_trans_bank = NULL;
   return rcode;
}

