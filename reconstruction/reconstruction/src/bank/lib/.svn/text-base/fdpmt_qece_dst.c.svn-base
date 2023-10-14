#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdpmt_qece_dst.h"

fdpmt_qece_dst_common fdpmt_qece_;
static integer4 fdpmt_qece_blen = 0;
static integer4 fdpmt_qece_maxlen = sizeof(integer4)*2 + sizeof(fdpmt_qece_dst_common);
static integer1* fdpmt_qece_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdpmt_qece_bank_buffer_ (integer4* fdpmt_qece_bank_buffer_size)
{
  (*fdpmt_qece_bank_buffer_size) = fdpmt_qece_blen;
  return fdpmt_qece_bank;
}



static void fdpmt_qece_bank_init();
integer4 fdpmt_qece_common_to_bank_();
static integer4 fdpmt_qece_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdpmt_qece_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdpmt_qece_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdpmt_qece_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdpmt_qece_.uniqID,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   }else{
      fdpmt_qece_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdpmt_qece_.ceNMeasuredData,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(&fdpmt_qece_.ce,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(&fdpmt_qece_.ceDevLower,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(&fdpmt_qece_.ceDevUpper,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpacki4_(&fdpmt_qece_.qeNMeasuredData,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpacki4_(&fdpmt_qece_.qeNLambda,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(&fdpmt_qece_.qeMinLambda,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(&fdpmt_qece_.qeDeltaLambda,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);

   nobj = fdpmt_qece_.qeNLambda;
   rcode += dst_unpackr4_(fdpmt_qece_.qe,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(fdpmt_qece_.qeDevLower,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_unpackr4_(fdpmt_qece_.qeDevUpper,&nobj,bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   
   return rcode;
}

/* write method */
integer4 fdpmt_qece_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdpmt_qece_ -> unit
   integer4 rcode=0;
   if ((rcode += fdpmt_qece_common_to_bank_()) != 0){
      fprintf (stderr, "fdpmt_qece_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdpmt_qece_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdpmt_qece_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdpmt_qece_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdpmt_qece_dst\n");
   fprintf(fp,"uniq ID: %d ",fdpmt_qece_.uniqID);
   if(fdpmt_qece_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdpmt_qece_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");

   fprintf(fp,"collection efficiency(CE)\n");
   fprintf(fp,"CE:%f CEDevLower:%f CEDevUpper:%f nMeasured:%d\n",fdpmt_qece_.ce,fdpmt_qece_.ceDevLower,fdpmt_qece_.ceDevUpper,fdpmt_qece_.ceNMeasuredData);
   fprintf(fp,"quantum efficiency(QE)\n");
   fprintf(fp,"nLambad=%d minLambda=%f deltaLamda=%f\n",fdpmt_qece_.qeNLambda,fdpmt_qece_.qeMinLambda,fdpmt_qece_.qeDeltaLambda);
   fprintf(fp,"lambda QE QEDevLower QEDevUpper\n");

   integer4 i;
   for(i=0;i<fdpmt_qece_.qeNLambda;i++){
      real4 lambda=fdpmt_qece_.qeMinLambda + i*fdpmt_qece_.qeDeltaLambda;
      fprintf(fp,"%f %f %f %f\n",lambda,fdpmt_qece_.qe[i],fdpmt_qece_.qeDevLower[i],fdpmt_qece_.qeDevUpper[i]);
   }

   return 0;
}

/* static method */
static void fdpmt_qece_bank_init(){
   fdpmt_qece_bank = (integer1*) calloc(fdpmt_qece_maxlen,sizeof(integer1));
   if(fdpmt_qece_bank==NULL){
      fprintf (stderr,"fdpmt_qece_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdpmt_qece_common_to_bank_(){
   static integer4 id = FDPMT_QECE_BANKID;
   static integer4 ver = FDPMT_QECE_BANKVERSION;
   integer4 rcode,nobj;
   if(fdpmt_qece_bank == NULL){
      fdpmt_qece_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdpmt_qece_blen,&fdpmt_qece_maxlen,fdpmt_qece_bank);

   nobj=1;
   rcode += dst_packi4_(&fdpmt_qece_.uniqID,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packi4_(&fdpmt_qece_.ceNMeasuredData,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(&fdpmt_qece_.ce,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(&fdpmt_qece_.ceDevLower,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(&fdpmt_qece_.ceDevUpper,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packi4_(&fdpmt_qece_.qeNMeasuredData,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packi4_(&fdpmt_qece_.qeNLambda,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(&fdpmt_qece_.qeMinLambda,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(&fdpmt_qece_.qeDeltaLambda,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);

   nobj = fdpmt_qece_.qeNLambda;
   rcode += dst_packr4_(fdpmt_qece_.qe,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(fdpmt_qece_.qeDevLower,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);
   rcode += dst_packr4_(fdpmt_qece_.qeDevUpper,&nobj,fdpmt_qece_bank,&fdpmt_qece_blen,&fdpmt_qece_maxlen);

   return rcode;
}

static integer4 fdpmt_qece_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdpmt_qece_blen,fdpmt_qece_bank);
   free(fdpmt_qece_bank);
   fdpmt_qece_bank = NULL;
   return rcode;
}

