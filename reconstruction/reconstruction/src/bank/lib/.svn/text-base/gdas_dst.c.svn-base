#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "gdas_dst.h"

gdas_dst_common gdasbank_;
static integer4 gdas_blen = 0;
static integer4 gdas_maxlen = sizeof(integer4)*2 + sizeof(gdas_dst_common);
static integer1* gdas_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* gdas_bank_buffer_ (integer4* gdas_bank_buffer_size)
{
  (*gdas_bank_buffer_size) = gdas_blen;
  return gdas_bank;
}



static void gdas_bank_init();
static integer4 gdas_bank_to_dst_(integer4 *unit);

/* read method */
integer4 gdas_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> gdas_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   gdas_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&gdas_blen,&gdas_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&gdasbank_.uniqID,&nobj,bank,&gdas_blen,&gdas_maxlen);
   }else{
      gdasbank_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&gdasbank_.dateFrom,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpacki4_(&gdasbank_.dateTo,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpacki4_(&gdasbank_.nItem,&nobj,bank,&gdas_blen,&gdas_maxlen);

   nobj=gdasbank_.nItem;

   //todo
   /*  Comment out 2015/Mar/12
   int k;
   for (k = 0; k < GDAS_NGRID; k++) {
     rcode += dst_unpackr4_(gdasbank_.height[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_unpackr4_(gdasbank_.pressure[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_unpackr4_(gdasbank_.pressureError[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_unpackr4_(gdasbank_.temperature[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_unpackr4_(gdasbank_.temperatureError[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_unpackr4_(gdasbank_.dewPoint[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_unpackr4_(gdasbank_.dewPointError[k],&nobj,bank,&gdas_blen,&gdas_maxlen);
   }
   */
   rcode += dst_unpackr4_(gdasbank_.height,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpackr4_(gdasbank_.pressure,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpackr4_(gdasbank_.pressureError,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpackr4_(gdasbank_.temperature,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpackr4_(gdasbank_.temperatureError,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpackr4_(gdasbank_.dewPoint,&nobj,bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_unpackr4_(gdasbank_.dewPointError,&nobj,bank,&gdas_blen,&gdas_maxlen);
   return rcode;
}

/* write method */
integer4 gdas_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //gdas_ -> unit
   integer4 rcode=0;
   if ((rcode += gdas_common_to_bank_()) != 0){
      fprintf (stderr, "gdas_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += gdas_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "gdas_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 gdas_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"gdas_dst\n");
   fprintf(fp,"uniq ID: %d ",gdasbank_.uniqID);
   if(gdasbank_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(gdasbank_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");
   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(gdasbank_.dateFrom,dateFromLine);
   convertSec2DateLine(gdasbank_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
   fprintf(fp,"nItem:%d\n",gdasbank_.nItem);

   if(gdasbank_.nItem==0){
      return 0;
   }

   fprintf(fp,"height press pressE temp tempE dew dewE\n");
   integer4 i;
   for(i=0;i<gdasbank_.nItem;i++){
     //      fprintf(fp,"%f %f %f %f %f %f %f\n",gdasbank_.height[17][i],gdasbank_.pressure[17][i],gdasbank_.pressureError[17][i],gdasbank_.temperature[17][i],gdasbank_.temperatureError[17][i],gdasbank_.dewPoint[17][i],gdasbank_.dewPointError[17][i]);
      fprintf(fp,"%f %f %f %f %f %f %f\n",gdasbank_.height[i],gdasbank_.pressure[i],gdasbank_.pressureError[i],gdasbank_.temperature[i],gdasbank_.temperatureError[i],gdasbank_.dewPoint[i],gdasbank_.dewPointError[i]);
   }

   return 0;
}

/* static method */
static void gdas_bank_init(){
   gdas_bank = (integer1*) calloc(gdas_maxlen,sizeof(integer1));
   if(gdas_bank==NULL){
      fprintf (stderr,"gdas_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 gdas_common_to_bank_(){
   static integer4 id = GDAS_BANKID;
   static integer4 ver = GDAS_BANKVERSION;
   integer4 rcode,nobj;
   if(gdas_bank == NULL){
      gdas_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&gdas_blen,&gdas_maxlen,gdas_bank);

   nobj=1;
   rcode += dst_packi4_(&gdasbank_.uniqID,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packi4_(&gdasbank_.dateFrom,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packi4_(&gdasbank_.dateTo,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packi4_(&gdasbank_.nItem,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);

   nobj=gdasbank_.nItem;

   //todo
   /* Comment out 2015/Mar/12
   int k;
   for (k = 0; k < GDAS_NGRID; k++) {
     rcode += dst_packr4_(gdasbank_.height[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_packr4_(gdasbank_.pressure[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_packr4_(gdasbank_.pressureError[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_packr4_(gdasbank_.temperature[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_packr4_(gdasbank_.temperatureError[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_packr4_(gdasbank_.dewPoint[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
     rcode += dst_packr4_(gdasbank_.dewPointError[k],&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   }
   */
   rcode += dst_packr4_(gdasbank_.height,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packr4_(gdasbank_.pressure,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packr4_(gdasbank_.pressureError,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packr4_(gdasbank_.temperature,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packr4_(gdasbank_.temperatureError,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packr4_(gdasbank_.dewPoint,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   rcode += dst_packr4_(gdasbank_.dewPointError,&nobj,gdas_bank,&gdas_blen,&gdas_maxlen);
   return rcode;
}

static integer4 gdas_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&gdas_blen,gdas_bank);
   free(gdas_bank);
   gdas_bank = NULL;
   return rcode;
}

/** Returns the grid index of a grid (lat, lon) */
int latlon2index(int lat, int lon) {
  return (lat - 37)*7 + (lon - 110);
}
