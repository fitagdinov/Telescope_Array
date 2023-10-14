#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdatmos_param_dst.h"

fdatmos_param_dst_common fdatmos_param_;
static integer4 fdatmos_param_blen = 0;
static integer4 fdatmos_param_maxlen = sizeof(integer4)*2 + sizeof(fdatmos_param_dst_common);
static integer1* fdatmos_param_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdatmos_param_bank_buffer_ (integer4* fdatmos_param_bank_buffer_size)
{
  (*fdatmos_param_bank_buffer_size) = fdatmos_param_blen;
  return fdatmos_param_bank;
}



static void fdatmos_param_bank_init();
static integer4 fdatmos_param_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdatmos_param_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdatmos_param_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdatmos_param_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdatmos_param_.uniqID,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   }else{
      fdatmos_param_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdatmos_param_.dateFrom,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpacki4_(&fdatmos_param_.dateTo,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpacki4_(&fdatmos_param_.nItem,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);

   nobj=fdatmos_param_.nItem;

   //todo
   rcode += dst_unpackr4_(fdatmos_param_.height,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpackr4_(fdatmos_param_.pressure,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpackr4_(fdatmos_param_.pressureError,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpackr4_(fdatmos_param_.temperature,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpackr4_(fdatmos_param_.temperatureError,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpackr4_(fdatmos_param_.dewPoint,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_unpackr4_(fdatmos_param_.dewPointError,&nobj,bank,&fdatmos_param_blen,&fdatmos_param_maxlen);

   return rcode;
}

/* write method */
integer4 fdatmos_param_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdatmos_param_ -> unit
   integer4 rcode=0;
   if ((rcode += fdatmos_param_common_to_bank_()) != 0){
      fprintf (stderr, "fdatmos_param_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdatmos_param_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdatmos_param_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdatmos_param_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdatmos_param_dst\n");
   fprintf(fp,"uniq ID: %d ",fdatmos_param_.uniqID);
   if(fdatmos_param_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdatmos_param_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");
   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(fdatmos_param_.dateFrom,dateFromLine);
   convertSec2DateLine(fdatmos_param_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
   fprintf(fp,"nItem:%d\n",fdatmos_param_.nItem);

   if(fdatmos_param_.nItem==0){
      return 0;
   }

   fprintf(fp,"height press pressE temp tempE dew dewE\n");
   integer4 i;
   for(i=0;i<fdatmos_param_.nItem;i++){
      fprintf(fp,"%f %f %f %f %f %f %f\n",fdatmos_param_.height[i],fdatmos_param_.pressure[i],fdatmos_param_.pressureError[i],fdatmos_param_.temperature[i],fdatmos_param_.temperatureError[i],fdatmos_param_.dewPoint[i],fdatmos_param_.dewPointError[i]);
   }

   return 0;
}

/* static method */
static void fdatmos_param_bank_init(){
   fdatmos_param_bank = (integer1*) calloc(fdatmos_param_maxlen,sizeof(integer1));
   if(fdatmos_param_bank==NULL){
      fprintf (stderr,"fdatmos_param_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdatmos_param_common_to_bank_(){
   static integer4 id = FDATMOS_PARAM_BANKID;
   static integer4 ver = FDATMOS_PARAM_BANKVERSION;
   integer4 rcode,nobj;
   if(fdatmos_param_bank == NULL){
      fdatmos_param_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdatmos_param_blen,&fdatmos_param_maxlen,fdatmos_param_bank);

   nobj=1;
   rcode += dst_packi4_(&fdatmos_param_.uniqID,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packi4_(&fdatmos_param_.dateFrom,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packi4_(&fdatmos_param_.dateTo,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packi4_(&fdatmos_param_.nItem,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);

   nobj=fdatmos_param_.nItem;

   //todo
   rcode += dst_packr4_(fdatmos_param_.height,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packr4_(fdatmos_param_.pressure,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packr4_(fdatmos_param_.pressureError,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packr4_(fdatmos_param_.temperature,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packr4_(fdatmos_param_.temperatureError,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packr4_(fdatmos_param_.dewPoint,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);
   rcode += dst_packr4_(fdatmos_param_.dewPointError,&nobj,fdatmos_param_bank,&fdatmos_param_blen,&fdatmos_param_maxlen);

   return rcode;
}

static integer4 fdatmos_param_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdatmos_param_blen,fdatmos_param_bank);
   free(fdatmos_param_bank);
   fdatmos_param_bank = NULL;
   return rcode;
}

