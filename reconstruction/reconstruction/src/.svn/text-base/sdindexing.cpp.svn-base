#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "event.h"
#include "sdindexing.h"

#include "ds1_2_detector_layout.h" // dates  < DS2_TO_DS3
#include "ds3_detector_layout.h"   // dates >= DS2_TO_DS3

// Initialize the internal arrays and fill them with counter index information
// depending on the intialization date different sets of counters will be 
// associated with each tower
sdindex_class::sdindex_class(int yymmdd)
{
  int itower, isd;
  int ix, iy;
  int xxyy;
  int i, j;

  ////////////////// Determine which detector layouts to use ////////////////
    
  // date when data set 3 layout should be used,
  // until data set 3 detector layout changes
  if (yymmdd >= DS2_TO_DS3)
    {
      n_det[0] = n_br_det_ds3;
      n_det[1] = n_lr_det_ds3;
      n_det[2] = n_sk_det_ds3;
      tower_det[0] = br_det_ds3;
      tower_det[1] = lr_det_ds3;
      tower_det[2] = sk_det_ds3;
    }
  // all data before data set 3 has the same detector layout
  else
    {
      n_det[0] = n_br_det_ds1_2;
      n_det[1] = n_lr_det_ds1_2;
      n_det[2] = n_sk_det_ds1_2;
      tower_det[0] = br_det_ds1_2;
      tower_det[1] = lr_det_ds1_2;
      tower_det[2] = sk_det_ds1_2;
    }
    
  ///////////// INITIALIZE ALL ARRAYS WITH -1 ////////////////////

  /*
   * itower=0: BR indexing
   * itower=1: LR indexing
   * itower=2: SK indexing
   * itower=3: full SD array indexing
   */
  for (isd=0; isd<SDMON_MAXSDS; isd++)
    {
      for (itower=0; itower<4; itower++)
	ind2xxyy[isd][itower]=-1;
    }
  for (ix=0; ix<SDMON_X_MAX; ix++)
    {
      for (iy=0; iy<SDMON_Y_MAX; iy++)
	{
	  xxyy2tid[ix][iy] = -1;
	  for (itower=0; itower<4; itower++)
	    xxyy2ind[ix][iy][itower] = -1;
	}
    }
  //////////////// FILL THE ARRAYS WITH COUNTER INDEX INFORMATION //////////////////////
  /* itower=0,1,2 is for BR,LR,SK, correspondingly.  
     itower=3 is special and means for the whole array
  */
  for (itower = 0; itower < 3; itower++)
    {
      for (isd=0; isd<n_det[itower]; isd++)
	{
	  xxyy = tower_det[itower][isd];
	  ix=xxyy/100 - 1;
	  iy=xxyy%100 - 1;
	  if (ix < 0 || ix >= SDMON_X_MAX || iy<0 || iy>=SDMON_Y_MAX)
	    {
	      fprintf(stderr,"sdindex_class: %04d is an invaling detector ID\n",xxyy);
	      exit(1);
	    }
	  ind2xxyy[isd][itower]=xxyy;
	  ind2xxyy[isd][3]=xxyy;
	  xxyy2tid[ix][iy] = 0;
	  xxyy2ind[ix][iy][itower] = isd;
	  xxyy2ind[ix][iy][3] = isd;
	}
    }
    
  // PREPARE ARRAYS FOR COMBINING TOWER IDS WHEN WRITING OUT COMBINED EVENTS
  for (i=0; i<=RUSDRAW_BRLRSK; i++)
    {
      for (j=0; j<=RUSDRAW_BRLRSK; j++)
	combined_tower_id[i][j] = -1;
    }

  // BR-LR combinations
  combined_tower_id[RUSDRAW_BR][RUSDRAW_LR] = RUSDRAW_BRLR;
  combined_tower_id[RUSDRAW_LR][RUSDRAW_BR] = RUSDRAW_BRLR;

  // BR-SK combinations
  combined_tower_id[RUSDRAW_BR][RUSDRAW_SK] = RUSDRAW_BRSK;
  combined_tower_id[RUSDRAW_SK][RUSDRAW_BR] = RUSDRAW_BRSK;

  // LR-SK combinations
  combined_tower_id[RUSDRAW_LR][RUSDRAW_SK] = RUSDRAW_LRSK;
  combined_tower_id[RUSDRAW_SK][RUSDRAW_LR] = RUSDRAW_LRSK;

  // BR-LR-SK combinations
  combined_tower_id[RUSDRAW_BR][RUSDRAW_LRSK] = RUSDRAW_BRLRSK;
  combined_tower_id[RUSDRAW_LRSK][RUSDRAW_BR] = RUSDRAW_BRLRSK;
  combined_tower_id[RUSDRAW_LR][RUSDRAW_BRSK] = RUSDRAW_BRLRSK;
  combined_tower_id[RUSDRAW_BRSK][RUSDRAW_LR] = RUSDRAW_BRLRSK;
  combined_tower_id[RUSDRAW_SK][RUSDRAW_BRLR] = RUSDRAW_BRLRSK;
  combined_tower_id[RUSDRAW_BRLR][RUSDRAW_SK] = RUSDRAW_BRLRSK;

}

sdindex_class::~sdindex_class()
{
}
int sdindex_class::getInd(int xxyy)
{
  int ix, iy;
  ix=xxyy/100 - 1;
  iy=xxyy%100 - 1;
  if (ix < 0 || ix >= SDMON_X_MAX || iy<0 || iy>=SDMON_Y_MAX)
    return -1;
  return xxyy2ind[ix][iy][3];
}
int sdindex_class::getInd(int itower, int xxyy)
{
  int ix, iy;
  if ((itower < 0) || (itower > 2))
    {
      fprintf(stderr, "sdindex_class::getInd: %d is an invalid tower id\n",itower);
      exit(1);
    }
  ix=xxyy/100 - 1;
  iy=xxyy%100 - 1;
  if ((ix < 0) || (ix >= SDMON_X_MAX) || (iy<0) || (iy>=SDMON_Y_MAX))
    return -1;
  return xxyy2ind[ix][iy][itower];
}
int sdindex_class::getNsds()
{
  return n_det[0]+n_det[1]+n_det[2];
}
int sdindex_class::getNsds(int itower)
{
  switch (itower)
    {
    case 0:
      {
        return n_det[0];
        break;
      }
    case 1:
      {
        return n_det[1];
        break;
      }
    case 2:
      {
        return n_det[2];
        break;
      }
    default:
      fprintf(stderr, "sdindex_class::getInd: %d is an invalid tower id\n",itower);
      exit(1);
    }
  return 0;
}
int sdindex_class::getMaxInd()
{
  return getNsds() - 1;
}
int sdindex_class::getMaxInd(int itower)
{
  return getNsds(itower) - 1;
}
int sdindex_class::getTowerID(int xxyy)
{
  int ix, iy;
  ix=xxyy/100 - 1;
  iy=xxyy%100 - 1;
  if (ix < 0 || ix >= SDMON_X_MAX || iy<0 || iy>=SDMON_Y_MAX)
    return -1;
  return xxyy2tid[ix][iy];
}
int sdindex_class::getXXYYfromInd(int ind)
{
  if (ind < 0 || ind >= SDMON_MAXSDS)
    {
      fprintf(stderr, "sdindex_class: Ivalind full-array index\n");
      exit(1);
    }
  return ind2xxyy[ind][3];
}
int sdindex_class::getXXYYfromInd(int itower, int ind)
{
  if (itower < 0 || itower > 2){
    fprintf (stderr, "sdindex_class::getInd: %d is an invalid tower id\n",itower);
    exit(1);
  }
  if (ind < 0 || ind >= SDMON_MAXSDS)
    {
      fprintf (stderr, "sdindex_class: Ivalind full-array index\n");
      exit(1);
    }
  return ind2xxyy[ind][itower];
}

bool sdindex_class::addTowerID(int *com_tid, int tid)
{
  // If tower ID can't be added
  if ((tid < 0) || (tid > RUSDRAW_BRLRSK))
    {
      (*com_tid) = -1;
      return false;
    }
  // Can't add the entire BR-LR-SK combination to something already existing
  if ( ((*com_tid) != -1) && (tid == RUSDRAW_BRLRSK) )
    {
      (*com_tid) = -1;
      return false;
    }
    
  // If can't add any more tower IDs
  if ((*com_tid) >= RUSDRAW_BRLRSK)
    {
      (*com_tid) = -1;
      return false;
    }
    
  // If combined tower id was initially empty
  if ((*com_tid) < 0)
    {
      (*com_tid) = tid;
      return true;
    }
    
  // Get the combined ID from the array
  (*com_tid) = combined_tower_id[(*com_tid)][tid];
    
  // Check if it's reasonable
  if (((*com_tid) < 0) || ((*com_tid) > RUSDRAW_BRLRSK))
    {
      (*com_tid) = -1;
      return false;
    }
    
  return true;
}
