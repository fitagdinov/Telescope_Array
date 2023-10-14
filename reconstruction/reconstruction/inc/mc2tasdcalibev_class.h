/*
 * mc2tasdcalibev_class.h
 *
 *  Dmitri Ivanov <ivanov@physics.rutgers.edu>
 */

#ifndef MC2TASDCALIBEV_CLASS_H_
#define MC2TASDCALIBEV_CLASS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include "event.h"
#include "TMath.h"
#include <map>

using namespace std;


// radius within which we fill out the alive detector information, km
#define alivedet_radius 7.5

// class to hold SD indices within each calibration cycle
class calibInd_class
{
  public:
    calibInd_class();
    virtual ~calibInd_class();
    void clear();
    int getInd(int xxyy);
    bool putInd(int ind, int xxyy);
  private:
    int calibInd[30][30];
};

class mc2tasdcalibev_class
{
  public:
    mc2tasdcalibev_class();
    virtual ~mc2tasdcalibev_class();
    void add_cnst_info(tasdconst_dst_common *x); // to add tasd constants information
    bool add_calib_info(tasdcalib_dst_common *x); // to add tasd calibration information; failure if false
    bool Convert(); // do all the conversions; true = success, false = failure
    // return the date for the calibration cycles
    int getCalibDate()
    {
       return yymmdd_calib;
    }

  private:
    int yymmdd_calib; // calibration date, should be same for all calibration cycles
    int hhmmss_calib; // time at the beginning of the calibration cycle used for calibrating this event
    map<int, tasdconst_dst_common> cnst; // latest tasd constants, index=lid
    map<int, tasdcalib_dst_common> calib; // calibration information indexed by time (hhmmss)
    map<int, calibInd_class> calibInd; // map to store the indices of SDs for each calibration cycle
    tasdcalibev_dst_common *icrr; // tasdcalibev (ICRR) event bank
    rusdraw_dst_common *ru; // rusdraw (Rutgers) event bank
    rusdmc_dst_common *rumc; // Rutgers MC bank
    void convThrown(); // convert the thrown part
    bool findCalib(); // find the appropriate calibration information; true if found, false if failed
    void convWf();   // convert the waveforms
    bool is_alive_det_relevant(int xxyy); // returns true if the detector is within 7km of the triggered position
    void fillDeadAlive(); // fill dead / alive detector information
    void fillWeather(); // fill the weather information
    double get_xdepth(double h); // get the vertical depth in g/cm^2 from vertical depth in m
    int get_const_ind(int xxyy); // get SD index in the constants bank
    void printErr(const char *form, ...);
};

#endif /* MC2TASDCALIBEV_CLASS_H_ */
