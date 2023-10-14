
/**
 * This DST bank has the information of reconstruction for hybrid
 * @author IKEDA Daisuke (ICRR) 2010-06-13
 * C/C++ version added by D. Ivanov <ivanov@physics.rutgers.edu>
 * In C/C++ version, structures are arranged to have the large type first
 * In dst pack/unpack routines, variables are read out / filled
 * in the same order as in the correcponding Java routines. 
 * This is the prototype bank for BR/LR Hybrid ( by D. Ikeda ) bank
 */


#ifndef _HYBRIDRECONFD_DST_
#define _HYBRIDRECONFD_DST_

#define HYBRIDRECONFD_BANKID		12030
#define HYBRIDRECONFD_BANKVERSION	004


// maximum number of reconstruction steps
#define hybridreconfd_fPreFDNStep 16

// maximum number of PMTs (256 tubes x 12 mirrors)
#define hybridreconfd_fPreFDNPmt 3072

//maximum photon type
#define hybridreconfd_fNPhotonType_IMC 16

//maximum slice of the showr axis in the IMC
#define hybridreconfd_maxSlotOfDepth 10000


#ifdef __cplusplus
extern "C" {
#endif
integer4 hybridreconfd_common_to_bank_();
integer4 hybridreconfd_bank_to_dst_(integer4 *unit);
integer4 hybridreconfd_common_to_dst_(integer4 *unit); // combines above 2
integer4 hybridreconfd_bank_to_common_(integer1 *bank);
integer4 hybridreconfd_common_to_dump_(integer4 *opt);
integer4 hybridreconfd_common_to_dumpf_(FILE *fp, integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* hybridreconfd_bank_buffer_ (integer4* hybridreconfd_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct 
{
  real4 fPhi;
  real4 fElev;
  real4 fSN;
  real4 fPed;
  real4 fPedRMS;
  real4 fTimeW1;
  real4 fTimeW2;
  real4 fChi2;
  real4 fNPE;
  real4 fAlpha;
  real4 fAlphaE;
  real4 fTime;
  real4 fTimeE;
  integer4 fTeleID;
  integer4 fPmtID;

  real4 fDepth;
  real4 fDepthE;
  real4 fNPE_IMC;
  real4 fNPE_IMCE;
  
} FDPMTData;

typedef struct 
{ 
  
  // full PMT information at the last step
  FDPMTData fPreFDPMT[hybridreconfd_fPreFDNPmt];
  
  real4 fGeomZen[2];   //[deg]
  real4 fGeomAzi[2];   //[deg] E to N
  real4 fGeomCore[3];  //[km] (x,y,z)
  real4 fGeomSDP[3];   //[km] unit vector (x,y,z) in CLF coordinate
  real4 fGeomRp[2];    //[km] 
  real4 fGeomPsi[2];   //[rad] fitting param
  real4 fGeomRcore[2]; //[km] fitting param
  real4 fGeomTcore[2]; //[ns] fitting param
  real4 fGeomChi2[2];  //0:chi2 or other qual, 1:ndf
  real4 fLngE[2];      //[eV]
  real4 fLngNmax[2];
  real4 fLngXmax[2];
  real4 fLngXint[2];
  real4 fLngLambda[2];
  real4 fLngObservedDepth[4]; //0:first depth, 1:last depth, 2:new first depth, 3:new last depth
  real4 fLngChi2[2];          //0:chi2 or other qual, 1:ndf
  real4 fPreFDTrackLength;    // track length
  real4 fPreFDTimeExtend;     // crossing time
  real4 fPreFDTotalNPE;       // total NPE

  real4 fNPE_IMC[hybridreconfd_fNPhotonType_IMC][hybridreconfd_maxSlotOfDepth]; //npe from each depth in IMC
  real4 fCVFrac_IMC; //cherenkov fraction

  // number of PMTs present at each step 
  // (full PMT info stored for the last step)
  integer4 fPreFDNPmt[hybridreconfd_fPreFDNStep];
  
  integer4 fPreSDTrigLid;
  integer4 fGeomUsedSD; //lid
  integer4 fNX_IMC;
  integer2 fHasPre;
  integer2 fPreFDNStep; //usually 5(0th to 4th step)
  integer2 fNPhotonType_IMC; //usually 5, 0:fluorescence, 1:direct cv, 2:rayleigh cv, 3: mie cv, 4: depth
  integer2 fHasGeom;
  integer2 fHasLng;  
  
} hybridreconfd_dst_common;

extern hybridreconfd_dst_common hybridreconfd_;
extern integer4 hybridreconfd_blen; /* needs to be accessed by the c files of the derived banks */ 

integer4 hybridreconfd_struct_to_abank_(hybridreconfd_dst_common *hybridreconfd, integer1 *(*pbank), integer4 id, integer4 ver);
integer4 hybridreconfd_abank_to_dst_(integer1 *bank, integer4 *unit);
integer4 hybridreconfd_struct_to_dst_(hybridreconfd_dst_common *hybridreconfd, integer1 *bank, integer4 *unit, integer4 id, integer4 ver);
integer4 hybridreconfd_abank_to_struct_(integer1 *bank, hybridreconfd_dst_common *hybridreconfd);
//integer4 hybridreconfd_struct_to_dump_(hybridreconfd_dst_common *hybridreconfd, integer4 *opt);
//integer4 hybridreconfd_struct_to_dumpf_(hybridreconfd_dst_common *hybridreconfd, FILE *fp, integer4 *opt);
integer4 hybridreconfd_struct_to_dump_(integer4 siteid, hybridreconfd_dst_common *hybridreconfd, integer4 *opt);
integer4 hybridreconfd_struct_to_dumpf_(integer4 siteid, hybridreconfd_dst_common *hybridreconfd, FILE *fp, integer4 *opt);

#endif
