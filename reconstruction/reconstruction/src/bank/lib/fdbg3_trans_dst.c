#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdbg3_trans_dst.h"

fdbg3_trans_dst_common fdbg3_trans_;
static integer4 fdbg3_trans_blen = 0;
static integer4 fdbg3_trans_maxlen = sizeof(integer4)*2 + sizeof(fdbg3_trans_dst_common);
static integer1* fdbg3_trans_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdbg3_trans_bank_buffer_ (integer4* fdbg3_trans_bank_buffer_size)
{
  (*fdbg3_trans_bank_buffer_size) = fdbg3_trans_blen;
  return fdbg3_trans_bank;
}



static void fdbg3_trans_bank_init();
static integer4 fdbg3_trans_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdbg3_trans_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdbg3_trans_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdbg3_trans_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdbg3_trans_.uniqID,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   }else{
      fdbg3_trans_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdbg3_trans_.nMeasuredData,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_unpacki4_(&fdbg3_trans_.nLambda,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_unpackr4_(&fdbg3_trans_.minLambda,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_unpackr4_(&fdbg3_trans_.deltaLambda,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);


   nobj = fdbg3_trans_.nLambda;
   rcode += dst_unpackr4_(fdbg3_trans_.transmittance,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_unpackr4_(fdbg3_trans_.transmittanceDevLower,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_unpackr4_(fdbg3_trans_.transmittanceDevUpper,&nobj,bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   
   return rcode;
}

/* write method */
integer4 fdbg3_trans_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdbg3_trans_ -> unit
   integer4 rcode=0;
   if ((rcode += fdbg3_trans_common_to_bank_()) != 0){
      fprintf (stderr, "fdbg3_trans_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdbg3_trans_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdbg3_trans_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdbg3_trans_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdbg3_trans_dst\n");
   fprintf(fp,"uniq ID: %d ",fdbg3_trans_.uniqID);
   if(fdbg3_trans_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdbg3_trans_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");

   fprintf(fp,"nMeasuredData=%d\n",fdbg3_trans_.nMeasuredData);
   fprintf(fp,"nLambda=%d minLambda=%f deltaLambda=%f\n",fdbg3_trans_.nLambda,fdbg3_trans_.minLambda,fdbg3_trans_.deltaLambda);

   fprintf(fp,"lambda trans transELower transEUpper\n");
   integer4 i;
   for(i=0;i<fdbg3_trans_.nLambda;i++){
      float lambda=fdbg3_trans_.minLambda + i*fdbg3_trans_.deltaLambda;
      fprintf(fp,"%f %f %f %f\n",lambda,fdbg3_trans_.transmittance[i],fdbg3_trans_.transmittanceDevLower[i],fdbg3_trans_.transmittanceDevUpper[i]);
   }
   return 0;
}

/* static method */
static void fdbg3_trans_bank_init(){
   fdbg3_trans_bank = (integer1*) calloc(fdbg3_trans_maxlen,sizeof(integer1));
   if(fdbg3_trans_bank==NULL){
      fprintf (stderr,"fdbg3_trans_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdbg3_trans_common_to_bank_(){
   static integer4 id = FDBG3_TRANS_BANKID;
   static integer4 ver = FDBG3_TRANS_BANKVERSION;
   integer4 rcode,nobj;
   if(fdbg3_trans_bank == NULL){
      fdbg3_trans_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdbg3_trans_blen,&fdbg3_trans_maxlen,fdbg3_trans_bank);

   nobj=1;
   rcode += dst_packi4_(&fdbg3_trans_.uniqID,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_packi4_(&fdbg3_trans_.nMeasuredData,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_packi4_(&fdbg3_trans_.nLambda,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_packr4_(&fdbg3_trans_.minLambda,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_packr4_(&fdbg3_trans_.deltaLambda,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   nobj = fdbg3_trans_.nLambda;
   rcode += dst_packr4_(fdbg3_trans_.transmittance,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_packr4_(fdbg3_trans_.transmittanceDevLower,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   rcode += dst_packr4_(fdbg3_trans_.transmittanceDevUpper,&nobj,fdbg3_trans_bank,&fdbg3_trans_blen,&fdbg3_trans_maxlen);
   return rcode;
}

static integer4 fdbg3_trans_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdbg3_trans_blen,fdbg3_trans_bank);
   free(fdbg3_trans_bank);
   fdbg3_trans_bank = NULL;
   return rcode;
}

