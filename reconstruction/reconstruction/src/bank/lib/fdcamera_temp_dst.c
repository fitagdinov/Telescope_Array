#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdcamera_temp_dst.h"

fdcamera_temp_dst_common fdcamera_temp_;
static integer4 fdcamera_temp_blen = 0;
static integer4 fdcamera_temp_maxlen = sizeof(integer4)*2 + sizeof(fdcamera_temp_dst_common);
static integer1* fdcamera_temp_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdcamera_temp_bank_buffer_ (integer4* fdcamera_temp_bank_buffer_size)
{
  (*fdcamera_temp_bank_buffer_size) = fdcamera_temp_blen;
  return fdcamera_temp_bank;
}



static void fdcamera_temp_bank_init();
static integer4 fdcamera_temp_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdcamera_temp_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdcamera_temp_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdcamera_temp_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdcamera_temp_.uniqID,&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   }else{
      fdcamera_temp_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdcamera_temp_.dateFrom,&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_unpacki4_(&fdcamera_temp_.dateTo,&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_unpacki2_(&fdcamera_temp_.nSite,&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_unpacki2_(&fdcamera_temp_.nTelescope,&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);

   nobj=fdcamera_temp_.nTelescope;
   integer2 site;
   for(site=0;site<fdcamera_temp_.nSite;site++){
      rcode += dst_unpacki4_(fdcamera_temp_.badFlag[site],&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
      rcode += dst_unpackr4_(fdcamera_temp_.temp[site],&nobj,bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   }

   return rcode;
}

/* write method */
integer4 fdcamera_temp_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdcamera_temp_ -> unit
   integer4 rcode=0;
   if ((rcode += fdcamera_temp_common_to_bank_()) != 0){
      fprintf (stderr, "fdcamera_temp_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdcamera_temp_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdcamera_temp_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdcamera_temp_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
  (void)(long_output);
   fprintf(fp,"fdcamera_temp_dst\n");
   fprintf(fp,"uniq ID: %d ",fdcamera_temp_.uniqID);
   if(fdcamera_temp_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdcamera_temp_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");
   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(fdcamera_temp_.dateFrom,dateFromLine);
   convertSec2DateLine(fdcamera_temp_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
   fprintf(fp,"nSite:%d nTelescope:%d\n",fdcamera_temp_.nSite,fdcamera_temp_.nTelescope);

   fprintf(fp,"site telescope pmt badFlag gain gainError\n");
   integer2 site;
   integer2 tele;
   for(site=0;site<fdcamera_temp_.nSite;site++){
      for(tele=0;tele<fdcamera_temp_.nTelescope;tele++){
         fprintf(fp,"%d %d %d %f\n",site,tele,fdcamera_temp_.badFlag[site][tele],fdcamera_temp_.temp[site][tele]);
      }
   }

   return 0;
}

/* static method */
static void fdcamera_temp_bank_init(){
   fdcamera_temp_bank = (integer1*) calloc(fdcamera_temp_maxlen,sizeof(integer1));
   if(fdcamera_temp_bank==NULL){
      fprintf (stderr,"fdcamera_temp_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdcamera_temp_common_to_bank_(){
   static integer4 id = FDCAMERA_TEMP_BANKID;
   static integer4 ver = FDCAMERA_TEMP_BANKVERSION;
   integer4 rcode,nobj;
   if(fdcamera_temp_bank == NULL){
      fdcamera_temp_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdcamera_temp_blen,&fdcamera_temp_maxlen,fdcamera_temp_bank);

   nobj=1;
   rcode += dst_packi4_(&fdcamera_temp_.uniqID,&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_packi4_(&fdcamera_temp_.dateFrom,&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_packi4_(&fdcamera_temp_.dateTo,&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_packi2_(&fdcamera_temp_.nSite,&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   rcode += dst_packi2_(&fdcamera_temp_.nTelescope,&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);

   integer2 site;
   nobj=fdcamera_temp_.nTelescope;
   for(site=0;site<fdcamera_temp_.nSite;site++){
      rcode += dst_packi4_(fdcamera_temp_.badFlag[site],&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
      rcode += dst_packr4_(fdcamera_temp_.temp[site],&nobj,fdcamera_temp_bank,&fdcamera_temp_blen,&fdcamera_temp_maxlen);
   }

   return rcode;
}

static integer4 fdcamera_temp_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdcamera_temp_blen,fdcamera_temp_bank);
   free(fdcamera_temp_bank);
   fdcamera_temp_bank = NULL;
   return rcode;
}

