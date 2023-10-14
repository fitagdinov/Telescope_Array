#ifndef _FDSHOWERPARAMETER_
#define _FDSHOWERPARAMETER_

#define FDSHOWERPARAMETER_BANKID 12804
#define FDSHOWERPARAMETER_BANKVERSION 000

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdshowerparameter_bank_to_common_(integer1 *bank);
integer4 fdshowerparameter_common_to_dst_(integer4 *unit);
integer4 fdshowerparameter_common_to_bank_();
integer4 fdshowerparameter_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdshowerparameter_bank_buffer_ (integer4* fdshowerparameter_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdshowerparameter_data_t{
   /* Parameters for primary particle */
   integer2 flavor; // See ---> telescopeArray.dataClass.particle.Particle
   integer2 doublet;
   integer2 massNumber;

   /* Parameters for longitudinal development */
   real4 energy; // [eV]
   real4 neMax; // See ---> telescopeArray.physics.EAScascade
   real4 xMax; // [g/cm^2]
   real4 xInt; // [g/cm^2]

   /* Parameters for geometry (in TACoordinate) */
   real4 zenith; // [deg]
   real4 azimuth; // [deg] (E=0,N=90,...)
   real4 core[3]; // [x,y,z] [cm] 

} fdshowerparameter_dst_common;

extern fdshowerparameter_dst_common fdshowerparameter_;


#endif
