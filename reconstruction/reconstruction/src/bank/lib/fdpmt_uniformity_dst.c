#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdpmt_uniformity_dst.h"

fdpmt_uniformity_dst_common fdpmt_uniformity_;
static integer4 fdpmt_uniformity_blen = 0;
static integer4 fdpmt_uniformity_maxlen = sizeof(integer4)*2 + sizeof(fdpmt_uniformity_dst_common);
static integer1* fdpmt_uniformity_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdpmt_uniformity_bank_buffer_ (integer4* fdpmt_uniformity_bank_buffer_size)
{
  (*fdpmt_uniformity_bank_buffer_size) = fdpmt_uniformity_blen;
  return fdpmt_uniformity_bank;
}



static void fdpmt_uniformity_bank_init();
integer4 fdpmt_uniformity_common_to_bank_();
static integer4 fdpmt_uniformity_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdpmt_uniformity_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdpmt_uniformity_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdpmt_uniformity_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdpmt_uniformity_.uniqID,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   }else{
      fdpmt_uniformity_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdpmt_uniformity_.xNDivision,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_unpackr4_(&fdpmt_uniformity_.xMinimum,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_unpackr4_(&fdpmt_uniformity_.xMaximum,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_unpacki4_(&fdpmt_uniformity_.yNDivision,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_unpackr4_(&fdpmt_uniformity_.yMinimum,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_unpackr4_(&fdpmt_uniformity_.yMaximum,&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);

   nobj = fdpmt_uniformity_.yNDivision;
   integer4 i;
   for(i=0;i<fdpmt_uniformity_.xNDivision;i++){
      rcode += dst_unpackr4_(fdpmt_uniformity_.uniformity[i],&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
      rcode += dst_unpackr4_(fdpmt_uniformity_.uniformityDev[i],&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
      rcode += dst_unpacki4_(fdpmt_uniformity_.nData[i],&nobj,bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   }

   return rcode;
}

/* write method */
integer4 fdpmt_uniformity_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdpmt_uniformity_ -> unit
   integer4 rcode=0;
   if ((rcode += fdpmt_uniformity_common_to_bank_()) != 0){
      fprintf (stderr, "fdpmt_uniformity_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdpmt_uniformity_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdpmt_uniformity_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdpmt_uniformity_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdpmt_uniformity_dst\n");
   fprintf(fp,"uniq ID: %d ",fdpmt_uniformity_.uniqID);
   if(fdpmt_uniformity_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdpmt_uniformity_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");
   fprintf(fp,"xDiv:%d xMin:%f xMax:%f ",fdpmt_uniformity_.xNDivision,fdpmt_uniformity_.xMinimum,fdpmt_uniformity_.xMaximum);
   fprintf(fp,"yDiv:%d yMin:%f yMax:%f\n",fdpmt_uniformity_.yNDivision,fdpmt_uniformity_.yMinimum,fdpmt_uniformity_.yMaximum);

   fprintf(fp,"x[mm] y[mm] uniformity uniformityDev nData\n");
   real4 deltaX = (fdpmt_uniformity_.xMaximum - fdpmt_uniformity_.xMinimum)/fdpmt_uniformity_.xNDivision;
   real4 deltaY = (fdpmt_uniformity_.yMaximum - fdpmt_uniformity_.yMinimum)/fdpmt_uniformity_.yNDivision;
   integer4 xBin;
   integer4 yBin;
   for(xBin=0;xBin<fdpmt_uniformity_.xNDivision;xBin++){
      for(yBin=0;yBin<fdpmt_uniformity_.yNDivision;yBin++){
         real4 x = fdpmt_uniformity_.xMinimum + xBin*deltaX + deltaX/2;
         real4 y = fdpmt_uniformity_.yMinimum + yBin*deltaY + deltaY/2;
         fprintf(fp,"%f %f %f %f %d\n",x,y,fdpmt_uniformity_.uniformity[xBin][yBin],fdpmt_uniformity_.uniformityDev[xBin][yBin],fdpmt_uniformity_.nData[xBin][yBin]);
      }
   }
   return 0;
}

/* static method */
static void fdpmt_uniformity_bank_init(){
   fdpmt_uniformity_bank = (integer1*) calloc(fdpmt_uniformity_maxlen,sizeof(integer1));
   if(fdpmt_uniformity_bank==NULL){
      fprintf (stderr,"fdpmt_uniformity_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdpmt_uniformity_common_to_bank_(){
   static integer4 id = FDPMT_UNIFORMITY_BANKID;
   static integer4 ver = FDPMT_UNIFORMITY_BANKVERSION;
   integer4 rcode,nobj;
   if(fdpmt_uniformity_bank == NULL){
      fdpmt_uniformity_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen,fdpmt_uniformity_bank);

   nobj=1;
   rcode += dst_packi4_(&fdpmt_uniformity_.uniqID,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_packi4_(&fdpmt_uniformity_.xNDivision,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_packr4_(&fdpmt_uniformity_.xMinimum,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_packr4_(&fdpmt_uniformity_.xMaximum,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_packi4_(&fdpmt_uniformity_.yNDivision,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_packr4_(&fdpmt_uniformity_.yMinimum,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   rcode += dst_packr4_(&fdpmt_uniformity_.yMaximum,&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);

   nobj = fdpmt_uniformity_.yNDivision;
   integer4 i;
   for(i=0;i<fdpmt_uniformity_.xNDivision;i++){
      rcode += dst_packr4_(fdpmt_uniformity_.uniformity[i],&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
      rcode += dst_packr4_(fdpmt_uniformity_.uniformityDev[i],&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
      rcode += dst_packi4_(fdpmt_uniformity_.nData[i],&nobj,fdpmt_uniformity_bank,&fdpmt_uniformity_blen,&fdpmt_uniformity_maxlen);
   }

   return rcode;
}

static integer4 fdpmt_uniformity_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdpmt_uniformity_blen,fdpmt_uniformity_bank);
   free(fdpmt_uniformity_bank);
   fdpmt_uniformity_bank = NULL;
   return rcode;
}

