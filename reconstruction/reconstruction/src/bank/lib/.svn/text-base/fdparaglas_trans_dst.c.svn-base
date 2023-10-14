#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdparaglas_trans_dst.h"

fdparaglas_trans_dst_common fdparaglas_trans_;
static integer4 fdparaglas_trans_blen = 0;
static integer4 fdparaglas_trans_maxlen = sizeof(integer4)*2 + sizeof(fdparaglas_trans_dst_common);
static integer1* fdparaglas_trans_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdparaglas_trans_bank_buffer_ (integer4* fdparaglas_trans_bank_buffer_size)
{
  (*fdparaglas_trans_bank_buffer_size) = fdparaglas_trans_blen;
  return fdparaglas_trans_bank;
}



static void fdparaglas_trans_bank_init();
static integer4 fdparaglas_trans_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdparaglas_trans_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdparaglas_trans_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdparaglas_trans_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdparaglas_trans_.uniqID,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   }else{
      fdparaglas_trans_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdparaglas_trans_.nMeasuredData,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_unpacki4_(&fdparaglas_trans_.nLambda,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_unpackr4_(&fdparaglas_trans_.minLambda,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_unpackr4_(&fdparaglas_trans_.deltaLambda,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);

   nobj = fdparaglas_trans_.nLambda;
   rcode += dst_unpackr4_(fdparaglas_trans_.transmittance,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_unpackr4_(fdparaglas_trans_.transmittanceDev,&nobj,bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   
   return rcode;
}

/* write method */
integer4 fdparaglas_trans_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdparaglas_trans_ -> unit
   integer4 rcode=0;
   if ((rcode += fdparaglas_trans_common_to_bank_()) != 0){
      fprintf (stderr, "fdparaglas_trans_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdparaglas_trans_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdparaglas_trans_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdparaglas_trans_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdparaglas_trans_dst\n");
   fprintf(fp,"uniq ID: %d ",fdparaglas_trans_.uniqID);
   if(fdparaglas_trans_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdparaglas_trans_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");
   fprintf(fp,"nMeasuredData=%d\n",fdparaglas_trans_.nMeasuredData);
   fprintf(fp,"nLambda=%d minLambda=%f deltaLambda=%f\n",fdparaglas_trans_.nLambda,fdparaglas_trans_.minLambda,fdparaglas_trans_.deltaLambda);

   fprintf(fp,"lambda trans transELower transEUpper\n");
   integer4 i;
   for(i=0;i<fdparaglas_trans_.nLambda;i++){
      float lambda=fdparaglas_trans_.minLambda + i*fdparaglas_trans_.deltaLambda;
      fprintf(fp,"%f %f %f\n",lambda,fdparaglas_trans_.transmittance[i],fdparaglas_trans_.transmittanceDev[i]);
   }
   return 0;
}

/* static method */
static void fdparaglas_trans_bank_init(){
   fdparaglas_trans_bank = (integer1*) calloc(fdparaglas_trans_maxlen,sizeof(integer1));
   if(fdparaglas_trans_bank==NULL){
      fprintf (stderr,"fdparaglas_trans_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdparaglas_trans_common_to_bank_(){
   static integer4 id = FDPARAGLAS_TRANS_BANKID;
   static integer4 ver = FDPARAGLAS_TRANS_BANKVERSION;
   integer4 rcode,nobj;
   if(fdparaglas_trans_bank == NULL){
      fdparaglas_trans_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen,fdparaglas_trans_bank);

   nobj=1;
   rcode += dst_packi4_(&fdparaglas_trans_.uniqID,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_packi4_(&fdparaglas_trans_.nMeasuredData,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_packi4_(&fdparaglas_trans_.nLambda,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_packr4_(&fdparaglas_trans_.minLambda,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_packr4_(&fdparaglas_trans_.deltaLambda,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);

   nobj = fdparaglas_trans_.nLambda;
   rcode += dst_packr4_(fdparaglas_trans_.transmittance,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);
   rcode += dst_packr4_(fdparaglas_trans_.transmittanceDev,&nobj,fdparaglas_trans_bank,&fdparaglas_trans_blen,&fdparaglas_trans_maxlen);

   return rcode;
}

static integer4 fdparaglas_trans_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdparaglas_trans_blen,fdparaglas_trans_bank);
   free(fdparaglas_trans_bank);
   fdparaglas_trans_bank = NULL;
   return rcode;
}

