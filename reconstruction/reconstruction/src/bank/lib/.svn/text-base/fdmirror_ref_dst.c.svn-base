#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdmirror_ref_dst.h"

fdmirror_ref_dst_common fdmirror_ref_;
static integer4 fdmirror_ref_blen = 0;
static integer4 fdmirror_ref_maxlen = sizeof(integer4)*2 + sizeof(fdmirror_ref_dst_common);
static integer1* fdmirror_ref_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdmirror_ref_bank_buffer_ (integer4* fdmirror_ref_bank_buffer_size)
{
  (*fdmirror_ref_bank_buffer_size) = fdmirror_ref_blen;
  return fdmirror_ref_bank;
}



static void fdmirror_ref_bank_init();
static integer4 fdmirror_ref_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdmirror_ref_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdmirror_ref_
   integer4 rcode = 0;
   integer4 nobj;
   integer4 version;
   fdmirror_ref_blen = sizeof(integer4); //skip id

   nobj=1;
   rcode += dst_unpacki4_(&version,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   if(version >= 1){
      rcode += dst_unpacki4_(&fdmirror_ref_.uniqID,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   }else{
      fdmirror_ref_.uniqID = 0;
   }
   rcode += dst_unpacki4_(&fdmirror_ref_.dateFrom,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpacki4_(&fdmirror_ref_.dateTo,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpacki2_(&fdmirror_ref_.nSite,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpacki2_(&fdmirror_ref_.nTelescope,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpacki2_(&fdmirror_ref_.nMirror,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpackr4_(&fdmirror_ref_.minLambda,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpackr4_(&fdmirror_ref_.deltaLambda,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_unpacki4_(&fdmirror_ref_.nLambda,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);

   integer2 site;
   integer2 tele;
   integer2 mirror;
   for(site=0;site<fdmirror_ref_.nSite;site++){
      for(tele=0;tele<fdmirror_ref_.nTelescope;tele++){
         for(mirror=0;mirror<fdmirror_ref_.nMirror;mirror++){
            fdmirror_ref_data* data = &(fdmirror_ref_.mirrorData[site][tele][mirror]); 
            nobj = 1;
            rcode += dst_unpacki4_(&(data->badFlag),&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
            if(data->badFlag != 0){
               continue;
            }
            nobj = fdmirror_ref_.nLambda;
            rcode += dst_unpackr4_(data->reflection,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
            rcode += dst_unpackr4_(data->reflectionDev,&nobj,bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
         }
      }
   }

   return rcode;
}

/* write method */
integer4 fdmirror_ref_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdmirror_ref_ -> unit
   integer4 rcode=0;
   if ((rcode += fdmirror_ref_common_to_bank_()) != 0){
      fprintf (stderr, "fdmirror_ref_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdmirror_ref_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdmirror_ref_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdmirror_ref_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   fprintf(fp,"fdmirror_ref_dst\n");
   fprintf(fp,"uniq ID: %d ",fdmirror_ref_.uniqID);
   if(fdmirror_ref_.uniqID!=0){
      char dateLine[32];
      convertSec2DateLine(abs(fdmirror_ref_.uniqID),dateLine);
      fprintf(fp,"(%s UTC)",dateLine);
   }
   fprintf(fp,"\n");

   char dateFromLine[32];
   char dateToLine[32];
   convertSec2DateLine(fdmirror_ref_.dateFrom,dateFromLine);
   convertSec2DateLine(fdmirror_ref_.dateTo,dateToLine);
   fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);

  if ( (*long_output) == 1) {
   
    integer2 site;
    integer2 tele;
    integer2 mirror;
    for(site=0;site<fdmirror_ref_.nSite;site++){
        for(tele=0;tele<fdmirror_ref_.nTelescope;tele++){
          for(mirror=0;mirror<fdmirror_ref_.nMirror;mirror++){
              fprintf(fp,"site=%d tele=%d mirror=%d\n",site,tele,mirror);
              fdmirror_ref_data* data = &(fdmirror_ref_.mirrorData[site][tele][mirror]); 
              if(data->badFlag != 0){
                continue;
              }
              fprintf(fp,"lambda reflection reflectionDev\n");
              integer4 i;
              for(i=0;i<fdmirror_ref_.nLambda;i++){
                float lambda = fdmirror_ref_.minLambda + i*fdmirror_ref_.deltaLambda;
                fprintf(fp,"%f %f %f\n",lambda,data->reflection[i],data->reflectionDev[i]);
              }
          }
        }
    }
  }
    else
    fprintf (fp, "Segment information not displayed in short output\n");
   return 0;
}

/* static method */
static void fdmirror_ref_bank_init(){
   fdmirror_ref_bank = (integer1*) calloc(fdmirror_ref_maxlen,sizeof(integer1));
   if(fdmirror_ref_bank==NULL){
      fprintf (stderr,"fdmirror_ref_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdmirror_ref_common_to_bank_(){
   static integer4 id = FDMIRROR_REF_BANKID;
   static integer4 ver = FDMIRROR_REF_BANKVERSION;
   integer4 rcode,nobj;
   if(fdmirror_ref_bank == NULL){
      fdmirror_ref_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdmirror_ref_blen,&fdmirror_ref_maxlen,fdmirror_ref_bank);

   nobj=1;
   rcode += dst_packi4_(&fdmirror_ref_.uniqID,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packi4_(&fdmirror_ref_.dateFrom,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packi4_(&fdmirror_ref_.dateTo,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packi2_(&fdmirror_ref_.nSite,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packi2_(&fdmirror_ref_.nTelescope,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packi2_(&fdmirror_ref_.nMirror,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packr4_(&fdmirror_ref_.minLambda,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packr4_(&fdmirror_ref_.deltaLambda,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
   rcode += dst_packi4_(&fdmirror_ref_.nLambda,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);

   integer2 site;
   integer2 tele;
   integer2 mirror;
   for(site=0;site<fdmirror_ref_.nSite;site++){
      for(tele=0;tele<fdmirror_ref_.nTelescope;tele++){
         for(mirror=0;mirror<fdmirror_ref_.nMirror;mirror++){
            fdmirror_ref_data* data = &(fdmirror_ref_.mirrorData[site][tele][mirror]); 
            nobj = 1;
            rcode += dst_packi4_(&(data->badFlag),&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
            if(data->badFlag != 0){
               continue;
            }
            nobj = fdmirror_ref_.nLambda;
            rcode += dst_packr4_(data->reflection,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
            rcode += dst_packr4_(data->reflectionDev,&nobj,fdmirror_ref_bank,&fdmirror_ref_blen,&fdmirror_ref_maxlen);
         }
      }
   }

   return rcode;
}

static integer4 fdmirror_ref_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdmirror_ref_blen,fdmirror_ref_bank);
   free(fdmirror_ref_bank);
   fdmirror_ref_bank = NULL;
   return rcode;
}

