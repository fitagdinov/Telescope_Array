/*
 * CLASS TO HANDLE INDEXING IN CALIBRATION AND MONITORING CYCLES.  KNOWS WHAT 
 * SDs ARE PRESENT (SEE SOURCE FILE FOR MORE DEFINITIONS) AND WILLL PUT THEM INTO MONITORING 
 * ARRAYS CORRESPONDINGLY
 */

#ifndef SDINDEXING_H_
#define SDINDEXING_H_

#include "rusdpass0_const.h"

class sdindex_class
  {
public:
  sdindex_class(int yymmdd);
  virtual ~sdindex_class();
  // These return -1 in case of failure
  int getInd(int xxyy); // Get monitoring index in full array indexing
  int getInd(int itower, int xxyy); // Get monitoring indedx in one tower indexing
  int getNsds(); // Get current number of SDs for the entire array
  int getNsds(int itower); // Get current number of SDs for a given tower
  int getMaxInd(); // Get the maximum value of index for all array
  int getMaxInd(int itower); // Get the maximum value of index for a given tower
  int getTowerID(int xxyy); // Get tower name corresponding to a given XXYY
  int getXXYYfromInd(int ind); // Get XXYY from index in full array indexing
  int getXXYYfromInd(int itower, int ind); // Get XXYY from index in one tower indexing
  bool addTowerID(int *com_tid, int tid); // Get combined tower ID for BR,LR,SK,BRLR,BRSK,LRSK,BRLRSK
private:
  int n_det[3];      // number of detectors responding to each tower
  int* tower_det[3]; // arrays of sd position IDs responding to each tower
  int ind2xxyy[SDMON_MAXSDS][4];
  int xxyy2tid[SDMON_X_MAX][SDMON_Y_MAX];
  int xxyy2ind[SDMON_X_MAX][SDMON_Y_MAX][4];
  int combined_tower_id[RUSDRAW_BRLRSK+1][RUSDRAW_BRLRSK+1]; // Holds combined tower ID values
  };
#endif /*SDINDEXING_H_*/
