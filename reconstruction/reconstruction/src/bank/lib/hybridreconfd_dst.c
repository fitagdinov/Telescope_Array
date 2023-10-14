// Created 2010/09/17 by D. Ivanov <ivanov@physics.rutgers.edu>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "hybridreconfd_dst.h"
#include "hybridreconbr_dst.h"
#include "hybridreconlr_dst.h"

hybridreconfd_dst_common hybridreconfd_;

integer4 hybridreconfd_blen = 0; /* not static because it needs to be accessed by the c files of the derived banks */
static integer4 hybridreconfd_maxlen = sizeof(integer4) * 2 + sizeof(hybridreconfd_dst_common);
static integer1 *hybridreconfd_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hybridreconfd_bank_buffer_ (integer4* hybridreconfd_bank_buffer_size)
{
  (*hybridreconfd_bank_buffer_size) = hybridreconfd_blen;
  return hybridreconfd_bank;
}



static void hybridreconfd_abank_init(integer1* (*pbank) ) {
  *pbank = (integer1 *)calloc(hybridreconfd_maxlen, sizeof(integer1));
  if (*pbank==NULL) {
    fprintf (stderr,"hybridreconfd_abank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

static void hybridreconfd_bank_init() {hybridreconfd_abank_init(&hybridreconfd_bank);}

integer4 hybridreconfd_common_to_bank_() {
  if (hybridreconfd_bank == NULL) hybridreconfd_bank_init();
  return hybridreconfd_struct_to_abank_(&hybridreconfd_, &hybridreconfd_bank, HYBRIDRECONFD_BANKID, HYBRIDRECONFD_BANKVERSION);
}
integer4 hybridreconfd_bank_to_dst_ (integer4 *unit) {return hybridreconfd_abank_to_dst_(hybridreconfd_bank, unit);}
integer4 hybridreconfd_common_to_dst_(integer4 *unit) {
  if (hybridreconfd_bank == NULL) hybridreconfd_bank_init();
  return hybridreconfd_struct_to_dst_(&hybridreconfd_, hybridreconfd_bank, unit, HYBRIDRECONFD_BANKID, HYBRIDRECONFD_BANKVERSION);
}
integer4 hybridreconfd_bank_to_common_(integer1 *bank) {return hybridreconfd_abank_to_struct_(bank, &hybridreconfd_);}
integer4 hybridreconfd_common_to_dump_(integer4 *opt) {return hybridreconfd_struct_to_dumpf_(-1, &hybridreconfd_, stdout, opt);}
integer4 hybridreconfd_common_to_dumpf_(FILE* fp, integer4 *opt) {return hybridreconfd_struct_to_dumpf_(-1, &hybridreconfd_, fp, opt);}

integer4 hybridreconfd_struct_to_abank_(hybridreconfd_dst_common *hybridreconfd, integer1 *(*pbank), integer4 id, integer4 ver) {
  integer4 rcode, nobj, i, nPmt;
  integer1 *bank;
  
  if (*pbank == NULL) hybridreconfd_abank_init(pbank);
  bank = *pbank;
  rcode = dst_initbank_(&id, &ver, &hybridreconfd_blen, &hybridreconfd_maxlen, bank);  
  
  nobj = 1;
  rcode += dst_packi2_(&hybridreconfd->fHasPre,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  if (hybridreconfd->fHasPre == 1)
    {
      nobj=1;
      rcode += dst_packi2_(&hybridreconfd->fPreFDNStep,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=hybridreconfd->fPreFDNStep;
      rcode += dst_packi4_(&hybridreconfd->fPreFDNPmt[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=1;
      rcode += dst_packr4_(&hybridreconfd->fPreFDTrackLength,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fPreFDTimeExtend,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fPreFDTotalNPE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nPmt = hybridreconfd->fPreFDNPmt[hybridreconfd->fPreFDNStep-1];
      for (i=0; i<nPmt; i++)
	{
	  rcode += dst_packi4_(&hybridreconfd->fPreFDPMT[i].fTeleID,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packi4_(&hybridreconfd->fPreFDPMT[i].fPmtID,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fPhi,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fElev,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fSN,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fPed,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fPedRMS,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fTimeW1,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fTimeW2,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fChi2,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fNPE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fAlpha,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fAlphaE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fTime,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fTimeE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fDepth,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fDepthE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fNPE_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_packr4_(&hybridreconfd->fPreFDPMT[i].fNPE_IMCE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	}
      rcode += dst_packi4_(&hybridreconfd->fPreSDTrigLid,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
    }
  nobj=1;
  rcode += dst_packi2_(&hybridreconfd->fHasGeom,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  if(hybridreconfd->fHasGeom == 1)
    {
      nobj=2;
      rcode += dst_packr4_(&hybridreconfd->fGeomZen[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fGeomAzi[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=3;
      rcode += dst_packr4_(&hybridreconfd->fGeomCore[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fGeomSDP[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=2;
      rcode += dst_packr4_(&hybridreconfd->fGeomRp[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fGeomPsi[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fGeomRcore[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fGeomTcore[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fGeomChi2[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=1;
      rcode += dst_packi4_(&hybridreconfd->fGeomUsedSD,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
    }
  nobj=1;
  rcode += dst_packi2_(&hybridreconfd->fHasLng,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  if (hybridreconfd->fHasLng == 1)
    {
      nobj=2;
      rcode += dst_packr4_(&hybridreconfd->fLngE[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fLngNmax[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fLngXmax[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fLngXint[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packr4_(&hybridreconfd->fLngLambda[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      //rcode += dst_packr4_(&hybridreconfd->fLngObservedDepth[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=4;
      rcode += dst_packr4_(&hybridreconfd->fLngObservedDepth[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=2;
      rcode += dst_packr4_(&hybridreconfd->fLngChi2[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);

      nobj=1;
      rcode += dst_packi2_(&hybridreconfd->fNPhotonType_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_packi4_(&hybridreconfd->fNX_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      if(hybridreconfd->fNX_IMC != 0){
          integer2 type;
          for(type=0;type<hybridreconfd->fNPhotonType_IMC;type++){
              rcode += dst_packr4_(hybridreconfd->fNPE_IMC[type],&hybridreconfd->fNX_IMC,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
          }
      }
      rcode += dst_packr4_(&hybridreconfd->fCVFrac_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
    }
  
  return rcode;
}

integer4 hybridreconfd_abank_to_dst_(integer1 *bank, integer4 *unit) {
  return dst_write_bank_(unit, &hybridreconfd_blen, bank);
}

integer4 hybridreconfd_struct_to_dst_(hybridreconfd_dst_common *hybridreconfd, integer1 *bank, integer4 *unit, integer4 id, integer4 ver) {
  integer4 rcode;
  if ( (rcode = hybridreconfd_struct_to_abank_(hybridreconfd, &bank, id, ver)) ) {
    fprintf(stderr, "hybridreconfd_struct_to_abank_ ERROR : %ld\n", (long)rcode);
    exit(0);
  }
  if ( (rcode = hybridreconfd_abank_to_dst_(bank, unit)) ) {
    fprintf(stderr, "hybridreconfd_abank_to_dst_ ERROR : %ld\n", (long)rcode);
    exit(0);
  }
  return 0;
}

integer4 hybridreconfd_abank_to_struct_(integer1 *bank, hybridreconfd_dst_common *hybridreconfd) {
  integer4 rcode = 0;
  integer4 nobj, i, nPmt;
  integer4 id, ver;
  hybridreconfd_blen = 0; /* do not skip id and version */
  nobj = 1;
  rcode += dst_unpacki4_(&id,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  rcode += dst_unpacki4_(&ver,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  rcode += dst_unpacki2_(&hybridreconfd->fHasPre,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  if (hybridreconfd->fHasPre == 1)
    {
      nobj=1;
      rcode += dst_unpacki2_(&hybridreconfd->fPreFDNStep,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=hybridreconfd->fPreFDNStep;
      rcode += dst_unpacki4_(&hybridreconfd->fPreFDNPmt[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=1;
      rcode += dst_unpackr4_(&hybridreconfd->fPreFDTrackLength,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fPreFDTimeExtend,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fPreFDTotalNPE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nPmt = hybridreconfd->fPreFDNPmt[hybridreconfd->fPreFDNStep-1];
      for (i=0; i<nPmt; i++)
	{
	  rcode += dst_unpacki4_(&hybridreconfd->fPreFDPMT[i].fTeleID,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpacki4_(&hybridreconfd->fPreFDPMT[i].fPmtID,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fPhi,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fElev,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fSN,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fPed,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fPedRMS,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fTimeW1,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fTimeW2,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fChi2,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fNPE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fAlpha,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fAlphaE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fTime,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
	  rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fTimeE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      if(ver >= 4){
          rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fDepth,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
          rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fDepthE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
          rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fNPE_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
          rcode += dst_unpackr4_(&hybridreconfd->fPreFDPMT[i].fNPE_IMCE,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      }
	}
      rcode += dst_unpacki4_(&hybridreconfd->fPreSDTrigLid,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
    }
  nobj=1;
  rcode += dst_unpacki2_(&hybridreconfd->fHasGeom,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  if(hybridreconfd->fHasGeom == 1)
    {
      nobj=2;
      rcode += dst_unpackr4_(&hybridreconfd->fGeomZen[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fGeomAzi[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=3;
      rcode += dst_unpackr4_(&hybridreconfd->fGeomCore[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fGeomSDP[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=2;
      rcode += dst_unpackr4_(&hybridreconfd->fGeomRp[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fGeomPsi[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fGeomRcore[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fGeomTcore[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fGeomChi2[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      nobj=1;
      rcode += dst_unpacki4_(&hybridreconfd->fGeomUsedSD,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
    }
  nobj=1;
  rcode += dst_unpacki2_(&hybridreconfd->fHasLng,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
  if (hybridreconfd->fHasLng == 1)
    {
      nobj=2;
      rcode += dst_unpackr4_(&hybridreconfd->fLngE[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fLngNmax[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fLngXmax[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fLngXint[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      rcode += dst_unpackr4_(&hybridreconfd->fLngLambda[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      if(ver >= 2){
        nobj=4;
        rcode += dst_unpackr4_(&hybridreconfd->fLngObservedDepth[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
        nobj=2;
      }else{
        rcode += dst_unpackr4_(&hybridreconfd->fLngObservedDepth[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      }
      rcode += dst_unpackr4_(&hybridreconfd->fLngChi2[0],&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);

      if(ver >= 3){
        nobj=1;
        rcode += dst_unpacki2_(&hybridreconfd->fNPhotonType_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
        rcode += dst_unpacki4_(&hybridreconfd->fNX_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
        if(hybridreconfd->fNX_IMC != 0){
            integer2 type;
            for(type=0;type<hybridreconfd->fNPhotonType_IMC;type++){
                rcode += dst_unpackr4_(hybridreconfd->fNPE_IMC[type],&hybridreconfd->fNX_IMC,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
            }
        }
        rcode += dst_unpackr4_(&hybridreconfd->fCVFrac_IMC,&nobj,bank,&hybridreconfd_blen,&hybridreconfd_maxlen);
      }
    }

  return rcode;
}

integer4 hybridreconfd_struct_to_dump_(integer4 siteid, hybridreconfd_dst_common *hybridreconfd, integer4 *long_output) {
  return hybridreconfd_struct_to_dumpf_(siteid, hybridreconfd, stdout, long_output);
}

integer4 hybridreconfd_struct_to_dumpf_(integer4 siteid, hybridreconfd_dst_common *hybridreconfd, FILE* fp, integer4 *long_output) 
{
  int i, j, nPmt;

  if ( siteid == 0 ){
    fprintf(fp,"%s :\n", "hybridreconbr");
  }else if( siteid == 1){
    fprintf(fp,"%s :\n", "hybridreconlr");
  }else{
    fprintf(fp,"%s :\n", "hybridreconfd");
  }
  
  if(hybridreconfd->fHasPre==1)
    {
      fprintf(fp,"pre:\n");
      fprintf(fp,"trigSD %04d\n",hybridreconfd->fPreSDTrigLid);
      fprintf(fp,"nPmt ");
      for(i=0;i<hybridreconfd->fPreFDNStep;i++)
	fprintf(fp,"%d ",hybridreconfd->fPreFDNPmt[i]);
      fprintf(fp,"\n");
      fprintf(fp,"track %f, time %f, npe %f\n",
	      hybridreconfd->fPreFDTrackLength,
	      hybridreconfd->fPreFDTimeExtend,
	      hybridreconfd->fPreFDTotalNPE);
      if ((*long_output))
	{
	  nPmt = hybridreconfd->fPreFDNPmt[hybridreconfd->fPreFDNStep-1];
	  fprintf(fp,"%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
		  "fTeleID","fPmtID","fPhi","fElev","fSN","fPed","fPedRMS",
		  "fTimeW1","fTimeW2","fChi2","fNPE","fAlpha","fAlphaE","fTime","fTimeE","fDepth","fDepthE","fNPE_IMC","fNPE_IMCE");
	  for (i=0; i<nPmt; i++)
	    {
	      fprintf(fp,
		      "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
		      hybridreconfd->fPreFDPMT[i].fTeleID,
		      hybridreconfd->fPreFDPMT[i].fPmtID,
		      hybridreconfd->fPreFDPMT[i].fPhi,
		      hybridreconfd->fPreFDPMT[i].fElev,
		      hybridreconfd->fPreFDPMT[i].fSN,
		      hybridreconfd->fPreFDPMT[i].fPed,
		      hybridreconfd->fPreFDPMT[i].fPedRMS,
		      hybridreconfd->fPreFDPMT[i].fTimeW1,
		      hybridreconfd->fPreFDPMT[i].fTimeW2,
		      hybridreconfd->fPreFDPMT[i].fChi2,
		      hybridreconfd->fPreFDPMT[i].fNPE,
		      hybridreconfd->fPreFDPMT[i].fAlpha,
		      hybridreconfd->fPreFDPMT[i].fAlphaE,
		      hybridreconfd->fPreFDPMT[i].fTime,
		      hybridreconfd->fPreFDPMT[i].fTimeE,
		      hybridreconfd->fPreFDPMT[i].fDepth,
		      hybridreconfd->fPreFDPMT[i].fDepthE,
		      hybridreconfd->fPreFDPMT[i].fNPE_IMC,
		      hybridreconfd->fPreFDPMT[i].fNPE_IMCE);
	    }
	}
      else
	fprintf(fp, "PMT information not displayed in short output mode\n");
    }
  if(hybridreconfd->fHasGeom==1)
    {
      fprintf(fp,"geom:\n");
      fprintf(fp,"zen:%f, azi:%f, sdp:(%f, %f, %f) Rp:%f\n",
	      hybridreconfd->fGeomZen[0],
	      hybridreconfd->fGeomAzi[0],
	      hybridreconfd->fGeomSDP[0],
	      hybridreconfd->fGeomSDP[1],
	      hybridreconfd->fGeomSDP[2],
	      hybridreconfd->fGeomRp[0]);
      fprintf(fp,"core: (%f %f %f)\n",
	      hybridreconfd->fGeomCore[0],
	      hybridreconfd->fGeomCore[1],
	      hybridreconfd->fGeomCore[2]);
      fprintf(fp,"psi:%f +- %f, rCore:%f +- %f, tCore:%f +- %f\n",
	      hybridreconfd->fGeomPsi[0],hybridreconfd->fGeomPsi[1],
	      hybridreconfd->fGeomRcore[0],hybridreconfd->fGeomRcore[1],
	      hybridreconfd->fGeomTcore[0],hybridreconfd->fGeomTcore[1]);
      fprintf(fp,"chi2:%f ndf:%d\n",
	      hybridreconfd->fGeomChi2[0],(int)hybridreconfd->fGeomChi2[1]);
      fprintf(fp,"used SD:%04d\n",
	      hybridreconfd->fGeomUsedSD);
    }
  if(hybridreconfd->fHasLng==1)
    {
      fprintf(fp,"lng:\n");
      fprintf(fp,"energy:%e, xmax:%f, xmin:%f, xmax:%f, xmin2:%f, xmax2:%f\n",
	      hybridreconfd->fLngE[0],hybridreconfd->fLngXmax[0],
	      hybridreconfd->fLngObservedDepth[0],hybridreconfd->fLngObservedDepth[1],
	      hybridreconfd->fLngObservedDepth[2],hybridreconfd->fLngObservedDepth[3]);
      fprintf(fp,"nmax:%e, xint:%f, lambda:%f\n",
	      hybridreconfd->fLngNmax[0],hybridreconfd->fLngXint[0],hybridreconfd->fLngLambda[0]);
      fprintf(fp,"chi2:%f ndf:%d\n",
	      hybridreconfd->fLngChi2[0],(int)hybridreconfd->fLngChi2[1]);
      if(hybridreconfd->fNX_IMC > 0){
      fprintf(fp,"nphoton type :%d nX:%d, cvFrac: %f\n",
              hybridreconfd->fNPhotonType_IMC,hybridreconfd->fNX_IMC,hybridreconfd->fCVFrac_IMC);
        for(i=0;i<hybridreconfd->fNX_IMC;i++){
            fprintf(fp,"%4.1f",hybridreconfd->fNPE_IMC[hybridreconfd->fNPhotonType_IMC-1][i]);
            for(j=0;j<hybridreconfd->fNPhotonType_IMC-1;j++){
                fprintf(fp," %5.3f",hybridreconfd->fNPE_IMC[j][i]);
            }
            fprintf(fp,"\n");
        }
      }
  }
  return 0;
}

