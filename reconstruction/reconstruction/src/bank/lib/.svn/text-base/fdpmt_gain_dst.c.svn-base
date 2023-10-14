#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdpmt_gain_dst.h"

fdpmt_gain_dst_common fdpmt_gain_;
static integer4 fdpmt_gain_blen = 0;
static integer4 fdpmt_gain_maxlen = sizeof(integer4)*2 + sizeof(fdpmt_gain_dst_common);
static integer1* fdpmt_gain_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdpmt_gain_bank_buffer_ (integer4* fdpmt_gain_bank_buffer_size)
{
  (*fdpmt_gain_bank_buffer_size) = fdpmt_gain_blen;
  return fdpmt_gain_bank;
}



static void fdpmt_gain_bank_init();
static integer4 fdpmt_gain_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdpmt_gain_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdpmt_gain_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdpmt_gain_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdpmt_gain_.uniqID,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   }else{
      fdpmt_gain_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdpmt_gain_.dateFrom,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_unpacki4_(&fdpmt_gain_.dateTo,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_unpacki2_(&fdpmt_gain_.nSite,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_unpacki2_(&fdpmt_gain_.nTelescope,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_unpacki2_(&fdpmt_gain_.nPmt,&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);

   integer2 site;
   integer2 tele;
   integer2 pmt;
   for(site=0;site<fdpmt_gain_.nSite;site++){
      for(tele=0;tele<fdpmt_gain_.nTelescope;tele++){
         for(pmt=0;pmt<fdpmt_gain_.nPmt;pmt++){
            fdpmt_gain_data* data = &(fdpmt_gain_.gainData[site][tele][pmt]); 
            rcode += dst_unpacki4_(&(data->badFlag),&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
            rcode += dst_unpackr4_(&(data->gain),&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
            rcode += dst_unpackr4_(&(data->gainError),&nobj,bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
         }
      }
   }

   return rcode;
}

/* write method */
integer4 fdpmt_gain_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdpmt_gain_ -> unit
   integer4 rcode=0;
   if ((rcode += fdpmt_gain_common_to_bank_()) != 0){
      fprintf (stderr, "fdpmt_gain_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdpmt_gain_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdpmt_gain_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdpmt_gain_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   fprintf(fp,"fdpmt_gain_dst\n");
   fprintf(fp,"uniq ID: %d ",fdpmt_gain_.uniqID);
   if(fdpmt_gain_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdpmt_gain_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");

   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(fdpmt_gain_.dateFrom,dateFromLine);
   convertSec2DateLine(fdpmt_gain_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
  if ( (*long_output) == 1) {
   fprintf(fp,"site telescope pmt badFlag gain gainError\n");
   integer2 site;
   integer2 tele;
   integer2 pmt;
   for(site=0;site<fdpmt_gain_.nSite;site++){
      for(tele=0;tele<fdpmt_gain_.nTelescope;tele++){
         for(pmt=0;pmt<fdpmt_gain_.nPmt;pmt++){
            fdpmt_gain_data* data = &(fdpmt_gain_.gainData[site][tele][pmt]); 
            fprintf(fp,"%d %d %d %d %f %f\n",site,tele,pmt,data->badFlag,data->gain,data->gainError);
         }
      }
   }
  }
    else
    fprintf (fp, "Tube information not displayed in short output\n");

   return 0;
}

/* static method */
static void fdpmt_gain_bank_init(){
   fdpmt_gain_bank = (integer1*) calloc(fdpmt_gain_maxlen,sizeof(integer1));
   if(fdpmt_gain_bank==NULL){
      fprintf (stderr,"fdpmt_gain_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdpmt_gain_common_to_bank_(){
   static integer4 id = FDPMT_GAIN_BANKID;
   static integer4 ver = FDPMT_GAIN_BANKVERSION;
   integer4 rcode,nobj;
   if(fdpmt_gain_bank == NULL){
      fdpmt_gain_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdpmt_gain_blen,&fdpmt_gain_maxlen,fdpmt_gain_bank);

   nobj=1;
   rcode += dst_packi4_(&fdpmt_gain_.uniqID,&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_packi4_(&fdpmt_gain_.dateFrom,&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_packi4_(&fdpmt_gain_.dateTo,&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_packi2_(&fdpmt_gain_.nSite,&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_packi2_(&fdpmt_gain_.nTelescope,&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
   rcode += dst_packi2_(&fdpmt_gain_.nPmt,&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);

   integer2 site;
   integer2 tele;
   integer2 pmt;
   for(site=0;site<fdpmt_gain_.nSite;site++){
      for(tele=0;tele<fdpmt_gain_.nTelescope;tele++){
         for(pmt=0;pmt<fdpmt_gain_.nPmt;pmt++){
            fdpmt_gain_data* data = &(fdpmt_gain_.gainData[site][tele][pmt]); 
            rcode += dst_packi4_(&(data->badFlag),&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
            rcode += dst_packr4_(&(data->gain),&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
            rcode += dst_packr4_(&(data->gainError),&nobj,fdpmt_gain_bank,&fdpmt_gain_blen,&fdpmt_gain_maxlen);
         }
      }
   }

   return rcode;
}

static integer4 fdpmt_gain_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdpmt_gain_blen,fdpmt_gain_bank);
   free(fdpmt_gain_bank);
   fdpmt_gain_bank = NULL;
   return rcode;
}

