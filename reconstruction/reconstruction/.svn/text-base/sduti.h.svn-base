#ifndef _sduti_h_
#define _sduti_h_

#include "sdanalysis_icc_settings.h"

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <string>
#include "dst_std_types.h"        
#include "dst_err_codes.h"
#include "filestack.h"		// parsing the want-files routines

#if !defined(SDUTI_FORMAT_CHECK)
#if defined(__GNUC__)
#define SDUTI_FORMAT_CHECK(format_index, args_index) __attribute__ ((__format__(printf, format_index, args_index)))
#else
#define SDUTI_FORMAT_CHECK(format_index, args_index)
#endif // __GNUC__
#endif // SDUTI_FORMAT_CHECK



// If the output of the routine is less than
// this, then the fit did not converge
#define GETFITSTATUS_GOOD 4


/************ SD I/O *************************************/
namespace SDIO
{
  // This is for the case when we either expected a variety of input suffixes
  // Replace the input suffix with the output suffix
  // and obtain the full path for the output file
  int makeOutFileName(const char *inFile, const char *outDir, 
		      const char *outSuffix, char *outFile);
  
  // Checks if a DST file has a valid suffix.
  // Returs 1 if a DST file has a valid suffix, returns 0 otherwise.
  int check_dst_suffix(const char *dstname);
  
  // Gets the DST suffix of a DST file into the string suffix
  // returns 0 if fails, 1 if success
  int getDSTsuffix(const char *inFile, char *suffix);
  
  // To produce the output file name,
  // with correct suffixes, etc.  Expects the input suffix.
  int GetOutFileName (const char *inFile, const char *outDir,
		      const char *inSuffix, const char *outSuffix, 
		      char *outFile);
  
  // To produce the output file name,
  // with correct suffixes, etc.  No input suffix.
  int GetOutFileName (const char *inFile, const char *outDir,
		      const char *outSuffix, char *outFile);
  
  // Substitude pattern 1 in str1 for pattern2 and gives the result in str2.  
  int patternSubst(const char *str1, const char *pattern1, const char *pattern2, char *str2);
  
  // Gets the suffix after the last '.'. 
  // If file name doesn't contain '.' then returns 0.
  // returns value is the position in fname string of the last '.'
  int getSuffix(const char *fname, char *suffix);
  
  // return 1 if something is available on stdin, 0 otherwise
  // poll_timeout_ms is time in milliseconds to wait for data to become
  // available on standard input; if this is -1 then wait until data
  // becomes available through standard input
  int  have_stdin(int poll_timeout_ms = 0);

  // customizable prototype function for message printing
  void vprintMessage(FILE *fp, const char *who, const char *what, va_list args);
  
  // customized message printing that fits most cases
  void printMessage(FILE *fp, const char *who, const char *what, ...) SDUTI_FORMAT_CHECK(3,4);

  // to format a C++ string using the usual printf format
  std::string strprintf(const char* what, ...) SDUTI_FORMAT_CHECK(1,2);
  
  // print warning / error messages by SDIO
  void printErr(const char *form, ...) SDUTI_FORMAT_CHECK(1,2);
  
}; 

/********* TIME, COORDINATES ****************************************/

namespace SDGEN
{
  
  ///////////////// CORSIKA  < ----- > PDG pagricle names, IDs, and masses ////////////////
  
  // Get corsika ID from particle name 
  int get_corid (const char* pname);
  
  // Get PDG ID and PDG mass from corsika ID
  bool get_pdginfo(int corID, int* pdgid, double* pdgm);
  
  // Get PDG ID from CORSIKA ID
  int corid2pdgid(int corid);
  
  // Get CORSIKA ID from PDG ID
  int pdgid2corid(int pdgid);
  
  // return name of the particle from PDG ID
  const char* pdgid2name(int pdgid);
  
  // for getting the sd coordinates
  void xxyy2xy(int xxyy, int *x, int *y);
  
  // To get year,month,day from yymmdd, or hour,minute second from hhmmss
  void parseAABBCC(int aabbcc, int *aa, int *bb, int *cc);
  
  void toAABBCC(int aa, int bb, int cc, int *aabbcc);
  
  // returns the time after midnight in seconds
  int timeAftMNinSec(int hhmmss);
  
  // Convert year, month, day to julian days since 1/1/2000
  int greg2jd(int year, int month, int day);
  
  // Obtain number of days since midnight of Jan 1, 2000 from date in yymmdd format
  int greg2jd(int yymmdd);
  
  // Convert julian days corresponding to midnight since Jan 1, 2000 to gregorian date
  void jd2greg(double julian, int *year, int *month, int *day);
  
  // Convert julian days corresponding to midnight since Jan 1, 2000 to yymmdd format
  int jd2yymmdd(double julian);
  
  // Change second by an integer ammount, original date and time variables
  // will be overwritten with those corresponding to new second
  void change_second(int *year, int *month, int *day, int *hr,
		     int *min, int *sec, int correction_sec);
  
  // Get time in seconds since midnight of Jan 2000
  int time_in_sec_j2000(int year, int month, int day, int hour, int minute, int second);
  
  // Get time in seconds since midnight of Jan 2000
  int time_in_sec_j2000(int yymmdd, int hhmmss);
  
  // Get time in seconds since midnight of Jan 1, 2000 including second fraction
  double time_in_sec_j2000f(int year, int month, int day, int hour, int minute, int second, int usec);
  
  // Get time in seconds since midnight of Jan 1, 2000 including second fraction
  double time_in_sec_j2000f(int yymmdd, int hhmmss, int usec);
  
  // Get calendar date from time in seconds since midnight of Jan 2000
  void j2000sec2greg(int j2000sec, int *year, int *month, int *day);
  
  // Get calendar date in yymmdd format from time in seconds since midnight of Jan 2000
  int j2000sec2yymmdd(int j2000sec);
  
  //////////////// For parsing Minuit output /////////////////////////////
  
  // Read migrad status string, return 
  // -1 - Did not understand the status string
  // 0 - FAILED
  // 1 - PROBLEMS
  // 2 - CALL LIMIT
  // 3 - NOT POSDEF
  // 4 - CONVERGED
  // 5 - SUCCESSFUL
  int getFitStatus (char *Migrad);
  
  
  ////////////////// Atmoshperic Models //////////////////////////////////
  
  // CORSIKA ATMOSPHERIC MODELS (B. T. Stokes adopted these from CORSIKA)
  // h = height in cm
  // mo = (vertical) mass overburden in g/cm^2
  // model = model number, model must be in [0 to 22] range
  double h2mo(double h, int model);  // get the height (in cm) from the mass overburden in g/cm^2
  double mo2h(double mo, int model); // get the mass overburden in g/cm^2 from the height in cm
  

  // Based on U.S. Standard Atmosphere, 1976
  // Ported to C/C++ from mc04(stdz76.f) and successfuly tested.
  // Dmitri Ivanov, <ivanov@physics.rutgers.edu>
  // Last modified: Jun 1, 2010
  // INPUTS:
  // h:      altitude [meter]
  // OUTPUTS:
  // pres:   pressure [millibar]
  // xdepth: vertical depth [g/cm^2]
  // tcent:  temperature in [degree Celsius]
  // rho:    density [g/cm^3]
  // RETURNS: 
  // 1:      h < 84852m, calculated variables are non-trivial
  // 0:      h > 84852m, calculated variables are at absolute zero
  int stdz76(double h,double *pres,double *xdepth,double *tcent,double *rho);
  
  
  // Get the vertical depth for a given altitude 
  // (U.S. Standard Atmosphere, 1976)
  // INPUTS:
  // h:      altitude  [meter]
  // OUTPUTS (RETURNS):
  // xdepth: vertical depth  [g/cm^2]
  double stdz76_xdepth(double h);
  
  
  // iverse of stdz76
  // INPUTS: 
  // xdepth: vertical depth [g/cm^2]
  // OUTPUTS (RETURN): 
  // altitude [m] in 0-84852m range
  double stdz76_inv(double xdepth);
  
  // return minimum and maximum height [meter] for which
  // the answers are non-trivial
  inline double stdz76_h_min()      { return 0.0;      }
  inline double stdz76_h_max()      { return 84852.0;  }
  
  // return minimum and maximum vertical slant depth for which
  // the answers are non-trivial
  inline double stdz76_xdepth_min() { return 0.0063564; }
  inline double stdz76_xdepth_max() { return 1033.23;   }


  ////// ATMOSPHERIC FUNCTIONS //////////////////////////
  
  // Saturated water vapor pressure as function of temperature
  // input: T in Kelvin
  // output: pressure in Pa 
  double Get_H20_Saturated_Vapor_Pressure(double T);
  
  // Density of air / water mixture
  // P_Pa:    Pressure of air/water vapor mixture [ K ]
  // T_K:     Temperature of air/water vapor mixture [ K ]
  // T_Dew_K: Dew point temperature [ K ]
  // Return: air density in g/cm3 units
  double Get_Air_Density_g_cm3(double P_Pa, double T_K, double T_Dew_K);
  

  // checks the current gdas DST bank for goodness
  // (if number of good points is less than 3 return false, otherwise return true)
  bool check_gdas();

  // Numerical calculation of the vertical mass overburden (g/cm^2)
  // given gdas atmospheric parameters such as pressure, temperature,
  // and the dew point at various pressure points
  double get_gdas_mo_numerically(double h_cm);

  // Numerical calculation of the density (g/cm^3)
  // given gdas atmospheric parameters such as pressure, temperature,
  // and the dew point at various pressure points
  double get_gdas_rho_numerically(double h_cm);
  
  // Numerical calculation (interpolation) to find gdas temperature
  // [ K ] at a given height
  double get_gdas_temp(double h_cm);

  //////////////////// Templates //////////////////////////////////////
  
  // Flipping the 1D - arrays of any type
  template < class T > static void
  flipArray (int asize, T * a)
  {
    T a1;
    int i, imax;
    imax = asize / 2;
    for (i = 0; i < imax; i++)
      {
	a1 = a[asize - 1 - i];
	a[asize - 1 - i] = a[i];
	a[i] = a1;
      }
  }
  
  
  // Removes points from 1D - arrays of any type
  // and shifts the remaining entries up
  template < class T > static void
  remArrayPoint (int *asize, int ipoint, T * a)
  {
    int i;
    (*asize)-=1;
    for (i = ipoint; i < (*asize); i++)
      a[i] = a[i+1];
  }
  
  
  // Removes points in two coupled 1D - arrays of any type but of
  // [][2] indexing form,  and shifts the remaining entries up
  template < class T > static void
  remArrayPoint2 (int *asize, int ipoint, T a[][2])
  {
    int i,k;
    (*asize) -= 1;
    for (i = ipoint; i < (*asize); i++)
      for (k=0; k<2; k++) a[i][k] = a[i+1][k];
  }
  

  ////////////////// NUMERICAL FUNCTIONS /////////////
  double linear_interpolation( int n, double *x, double *y, double t );
  

};



#endif
