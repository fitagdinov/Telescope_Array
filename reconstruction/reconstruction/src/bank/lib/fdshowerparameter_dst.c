#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "fdcalib_util.h"
#include "fdshowerparameter_dst.h"

fdshowerparameter_dst_common fdshowerparameter_;
static integer4 fdshowerparameter_blen = 0;
static integer4 fdshowerparameter_maxlen = sizeof(integer4)*2 + sizeof(fdshowerparameter_dst_common);
static integer1* fdshowerparameter_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdshowerparameter_bank_buffer_ (integer4* fdshowerparameter_bank_buffer_size)
{
  (*fdshowerparameter_bank_buffer_size) = fdshowerparameter_blen;
  return fdshowerparameter_bank;
}



static void fdshowerparameter_bank_init();
static integer4 fdshowerparameter_bank_to_dst_(integer4 *unit);

/* read method */
integer4 fdshowerparameter_bank_to_common_(integer1* bank){
   //must be written
   //buffer -> struct
   //bank -> fdshowerparameter_
   integer4 rcode = 0;
   integer4 nobj;
   fdshowerparameter_blen = 2 * sizeof(integer4); //skip id and ver.

   nobj=1;
   rcode += dst_unpacki2_(&fdshowerparameter_.flavor,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpacki2_(&fdshowerparameter_.doublet,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpacki2_(&fdshowerparameter_.massNumber,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpackr4_(&fdshowerparameter_.energy,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpackr4_(&fdshowerparameter_.neMax,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpackr4_(&fdshowerparameter_.xMax,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpackr4_(&fdshowerparameter_.xInt,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpackr4_(&fdshowerparameter_.zenith,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_unpackr4_(&fdshowerparameter_.azimuth,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   nobj=3;
   rcode += dst_unpackr4_(fdshowerparameter_.core,&nobj,bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);

   return rcode;
}

/* write method */
integer4 fdshowerparameter_common_to_dst_(integer4* unit){
   //must be written
   //struct -> dstfile
   //fdshowerparameter_ -> unit
   integer4 rcode=0;
   if ((rcode += fdshowerparameter_common_to_bank_()) != 0){
      fprintf (stderr, "fdshowerparameter_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   if ((rcode += fdshowerparameter_bank_to_dst_(unit)) != 0){
      fprintf (stderr, "fdshowerparameter_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(1);
   }
   return rcode;
}

/* dump method */
integer4 fdshowerparameter_common_to_dumpf_(FILE* fp,integer4* long_output){
   //must be written
   (void)(long_output);
   fprintf(fp,"fdshowerparameter_dst\n");
   fprintf(fp,"flavor %d, doublet %d, massNumber %d\n",fdshowerparameter_.flavor,fdshowerparameter_.doublet,fdshowerparameter_.massNumber);
   fprintf(fp,"energy %e[ev], neMax %e, xMax %f, xInt %f\n",fdshowerparameter_.energy,fdshowerparameter_.neMax,fdshowerparameter_.xMax,fdshowerparameter_.xInt);
   fprintf(fp,"zenith %f[deg], azimuth %f[deg], core(%f, %f, %f)[cm]\n",fdshowerparameter_.zenith,fdshowerparameter_.azimuth,fdshowerparameter_.core[0],fdshowerparameter_.core[1],fdshowerparameter_.core[2]);

   return 0;
}

/* static method */
static void fdshowerparameter_bank_init(){
   fdshowerparameter_bank = (integer1*) calloc(fdshowerparameter_maxlen,sizeof(integer1));
   if(fdshowerparameter_bank==NULL){
      fprintf (stderr,"fdshowerparameter_bank_init : fail to assign memory to bank. Abort.\n");
      exit(1);
   }
}

integer4 fdshowerparameter_common_to_bank_(){
   static integer4 id = FDSHOWERPARAMETER_BANKID;
   static integer4 ver = FDSHOWERPARAMETER_BANKVERSION;
   integer4 rcode,nobj;
   if(fdshowerparameter_bank == NULL){
      fdshowerparameter_bank_init();
   }
   rcode = dst_initbank_(&id,&ver,&fdshowerparameter_blen,&fdshowerparameter_maxlen,fdshowerparameter_bank);

   nobj=1;
   rcode += dst_packi2_(&fdshowerparameter_.flavor,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packi2_(&fdshowerparameter_.doublet,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packi2_(&fdshowerparameter_.massNumber,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packr4_(&fdshowerparameter_.energy,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packr4_(&fdshowerparameter_.neMax,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packr4_(&fdshowerparameter_.xMax,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packr4_(&fdshowerparameter_.xInt,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packr4_(&fdshowerparameter_.zenith,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   rcode += dst_packr4_(&fdshowerparameter_.azimuth,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);
   nobj=3;
   rcode += dst_packr4_(fdshowerparameter_.core,&nobj,fdshowerparameter_bank,&fdshowerparameter_blen,&fdshowerparameter_maxlen);

   return rcode;
}

static integer4 fdshowerparameter_bank_to_dst_(integer4* unit){
   integer4 rcode;
   rcode = dst_write_bank_(unit,&fdshowerparameter_blen,fdshowerparameter_bank);
   free(fdshowerparameter_bank);
   fdshowerparameter_bank = NULL;
   return rcode;
}

