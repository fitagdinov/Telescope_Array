#include "sdxyzclf_class.h"
#include "sdxyzclf_raw.h"
#include "br_xxyy.h"
#include "lr_xxyy.h"
#include "sk_xxyy.h"

#define MAX_NSD_PTOWER 256 // max. number of SDs per tower

using namespace std;

sdxyzclf_class::sdxyzclf_class()
  {
    int i, j, xxyy, x, y;

    for (x = 0; x < SDMON_X_MAX; x++)
      {
        for (y = 0; y < SDMON_Y_MAX; y++)
          {
            towerids_ds1[x][y] = -1;
            towerids_ds2[x][y] = -1;
	    towerids_ds3[x][y] = -1;
            for (i = 0; i < 3; i++)
              {
                sdxyzclf_ds1[x][y][i] = -1.e6;
                sdxyzclf_ds2[x][y][i] = -1.e6;
		sdxyzclf_ds3[x][y][i] = -1.e6;
              }
          }
      }
    // Make tower id look-up table
    for (i = 0; i < MAX_NSD_PTOWER; i++)
      {
        // BR
        if (i < NBR_DS1)
          {
            x = br_xxyy_ds1[i] / 100;
            y = br_xxyy_ds1[i] % 100;
            towerids_ds1[x - 1][y - 1] = RUSDRAW_BR;
            // In DS1, this counter is either present or not present
            // and for some tome it was reporting to a wrong tower.
            // Also it didn't have valid GPS coordinates in DS1.
            // Keep it out of the analysis in DS1.
            if (x == 22 && y == 8)
              towerids_ds1[x - 1][y - 1] = -1;
          }
        if (i < NBR_DS2)
          {
            x = br_xxyy_ds2[i] / 100;
            y = br_xxyy_ds2[i] % 100;
            towerids_ds2[x - 1][y - 1] = RUSDRAW_BR;
          }
	if (i < NBR_DS3)
          {
            x = br_xxyy_ds3[i] / 100;
            y = br_xxyy_ds3[i] % 100;
            towerids_ds3[x - 1][y - 1] = RUSDRAW_BR;
          }
        // LR
        if (i < NLR_DS1)
          {
            x = lr_xxyy_ds1[i] / 100;
            y = lr_xxyy_ds1[i] % 100;
            towerids_ds1[x - 1][y - 1] = RUSDRAW_LR;
          }
        if (i < NLR_DS2)
          {
            x = lr_xxyy_ds2[i] / 100;
            y = lr_xxyy_ds2[i] % 100;
            towerids_ds2[x - 1][y - 1] = RUSDRAW_LR;
          }
	if (i < NLR_DS3)
          {
            x = lr_xxyy_ds3[i] / 100;
            y = lr_xxyy_ds3[i] % 100;
            towerids_ds3[x - 1][y - 1] = RUSDRAW_LR;
          }
        // SK
        if (i < NSK_DS1)
          {
            x = sk_xxyy_ds1[i] / 100;
            y = sk_xxyy_ds1[i] % 100;
            towerids_ds1[x - 1][y - 1] = RUSDRAW_SK;
          }
        if (i < NSK_DS2)
          {
            x = sk_xxyy_ds2[i] / 100;
            y = sk_xxyy_ds2[i] % 100;
            towerids_ds2[x - 1][y - 1] = RUSDRAW_SK;
          }
	if (i < NSK_DS3)
          {
            x = sk_xxyy_ds3[i] / 100;
            y = sk_xxyy_ds3[i] % 100;
            towerids_ds3[x - 1][y - 1] = RUSDRAW_SK;
          }
      }

    // Make a GPS coordiante look-up table
    for (i = 0; i < SDXYZCLF_RAW_NDET_MAX; i++)
      {
        // GPS coordinates for data set 1
        if (i < SDXYZCLF_RAW_NDET_DS1)
          {
            xxyy = (int) floor(sdxyzclf_raw_ds1[4* i ] + 0.5);
            x = xxyy / 100;
            y = xxyy % 100;
            for (j = 1; j < 4; j++)
              sdxyzclf_ds1[x - 1][y - 1][j - 1] = sdxyzclf_raw_ds1[4* i + j];
          }
        // GPS coordinates for data set 2
        if (i < SDXYZCLF_RAW_NDET_DS2)
          {
            xxyy = (int) floor(sdxyzclf_raw_ds2[4* i ] + 0.5);
            x = xxyy / 100;
            y = xxyy % 100;
            for (j = 1; j < 4; j++)
              sdxyzclf_ds2[x - 1][y - 1][j - 1] = sdxyzclf_raw_ds2[4* i + j];
          }
	// GPS coordinates for data set 3
        if (i < SDXYZCLF_RAW_NDET_DS3)
          {
            xxyy = (int) floor(sdxyzclf_raw_ds3[4* i ] + 0.5);
            x = xxyy / 100;
            y = xxyy % 100;
            for (j = 1; j < 4; j++)
              sdxyzclf_ds3[x - 1][y - 1][j - 1] = sdxyzclf_raw_ds3[4* i + j];
          }
      }
  }
int sdxyzclf_class::get_towerid(int yymmdd, int x, int y)
  {
    if ((x < 1) || (x > SDMON_X_MAX) || (y < 1) || (y > SDMON_Y_MAX))
      return -1;
    
    // data set 1
    if (yymmdd < YYMMDD_DS1_2_DS2)
      return towerids_ds1[x - 1][y - 1];
    // data set 2
    else if (yymmdd >= YYMMDD_DS1_2_DS2 && yymmdd < YYMMDD_DS2_2_DS3)
      return towerids_ds2[x - 1][y - 1];
    // data set 3
    else
      return towerids_ds3[x - 1][y - 1];
  }
bool sdxyzclf_class::tower_part_of_rusdraw_site(int tower_id, int rusdraw_site)
{
  // To check if a given tower ID is a part of the
  // rusdraw_.site code. (tower_id = 0(BR), 1(LR), 2(SK)
  return (rusdraw_bitflag_from_site(rusdraw_site) & (1 << tower_id));
}


int sdxyzclf_class::get_towerid(int yymmdd, int xxyy)
  {
    int x, y;
    x = xxyy / 100;
    y = xxyy % 100;
    return get_towerid(yymmdd, x, y);
  }
bool sdxyzclf_class::get_xyz(int yymmdd, int x, int y, double *xyz)
  {
    if ((x < 1) || (x > SDMON_X_MAX) || (y < 1) || (y > SDMON_Y_MAX))
      {
        fprintf(stderr, "warning: date %06d: X=%d Y=%d is an invalid location\n", 
		yymmdd, x, y);
        return false;
      }
    if (yymmdd < YYMMDD_DS1_2_DS2)
      {
        if ((sdxyzclf_ds1[x - 1][y - 1][0] < (-1.0e3))
            || (sdxyzclf_ds1[x - 1][y - 1][1] < (-1.0e3)) || (sdxyzclf_ds1[x
            - 1][y - 1][2] < (-1.0e3)))
          {
            fprintf(stderr, "warning: date %06d: X=%d Y=%d GPS info is missing \n", 
		    yymmdd, x, y);
            return false;
          }
        memcpy(xyz, sdxyzclf_ds1[x - 1][y - 1], 3* sizeof (double));
      }
    else if(yymmdd >= YYMMDD_DS1_2_DS2 && yymmdd < YYMMDD_DS2_2_DS3)
      {
        if ((sdxyzclf_ds2[x-1][y-1][0] < (-1.0e3)) ||
            (sdxyzclf_ds2[x-1][y-1][1] < (-1.0e3)) ||
            (sdxyzclf_ds2[x-1][y-1][2] < (-1.0e3)) )
          {
	    fprintf(stderr, "warning: date %06d: X=%d Y=%d GPS info is missing \n", 
		    yymmdd, x, y);
            return false;
          }
        memcpy(xyz,sdxyzclf_ds2[x-1][y-1],3*sizeof(double));
      }
    else
      {
        if ((sdxyzclf_ds3[x-1][y-1][0] < (-1.0e3)) ||
            (sdxyzclf_ds3[x-1][y-1][1] < (-1.0e3)) ||
            (sdxyzclf_ds3[x-1][y-1][2] < (-1.0e3)) )
          {
	    fprintf(stderr, "warning: date %06d: X=%d Y=%d GPS info is missing \n", 
		    yymmdd, x, y);
            return false;
          }
        memcpy(xyz,sdxyzclf_ds3[x-1][y-1],3*sizeof(double));
      }
    return true;
  }
bool sdxyzclf_class::get_xyz(int yymmdd, int xxyy, double *xyz)
  {
    int x, y;
    x = xxyy / 100;
    y = xxyy % 100;
    return get_xyz(yymmdd, x, y, xyz);
  }

void sdxyzclf_class::get_closest_sd(int yymmdd, double *xy, int *xxyy,
    double *dr)
  {

    int i, j;
    double pos[2], r[2];
    double rmin, rval;

    rmin = 1.0e10;

    if (yymmdd < YYMMDD_DS1_2_DS2)
      {
        for (i = 0; i < SDXYZCLF_RAW_NDET_DS1; i++)
          {
            for (j = 1; j < 3; j++)
              pos[j - 1] = sdxyzclf_raw_ds1[4* i + j];
            r[0] = 1.2e3 * pos[0] - xy[0];
            r[1] = 1.2e3 * pos[1] - xy[1];
            rval = sqrt(r[0] * r[0] + r[1] * r[1]);
            if (rval < rmin)
              {
                rmin = rval;
                (*xxyy) = (int) floor(sdxyzclf_raw_ds1[4* i ] + 0.5);
                dr[0] = r[0];
                dr[1] = r[1];
              }
          }
      }
    else
      {
        for (i = 0; i < SDXYZCLF_RAW_NDET_DS2; i++)
          {
            for (j = 1; j < 3; j++)
              pos[j - 1] = sdxyzclf_raw_ds2[4* i + j];
            r[0] = 1.2e3 * pos[0] - xy[0];
            r[1] = 1.2e3 * pos[1] - xy[1];
            rval = sqrt(r[0] * r[0] + r[1] * r[1]);
            if (rval < rmin)
              {
                rmin = rval;
                (*xxyy) = (int) floor(sdxyzclf_raw_ds2[4* i ] + 0.5);
                dr[0] = r[0];
                dr[1] = r[1];
              }
          }
      }

  }

vector <sdxyzclf_class::sdpos>& sdxyzclf_class::get_counters(int yymmdd)
{
  sdpos pos;
  Counters.clear();
  for (int x=1; x<= SDMON_X_MAX; x++)
    {
      for (int y=1; y <= SDMON_Y_MAX; y++)
	{
	  pos.xxyy = 100*x+y;
	  pos.towerid=get_towerid(yymmdd,x,y);
	  if(pos.towerid == -1)
	    continue; // ignore the position ID if counter is absent
	  if(!get_xyz(yymmdd,x,y,pos.xyz))
	    continue; // ignore counter if don't have GPS coordinates (also prints a warning message)
	  Counters.push_back(pos);
	}
    }
  return Counters;
}
map<int,int>& sdxyzclf_class::get_counter_status_map(bsdinfo_dst_common *bsdinfo)
{
  counter_status_map.clear();
  // First intialize the map with existing counters as working
  sdpos pos;
  for (int x=1; x<= SDMON_X_MAX; x++)
    {
      for (int y=1; y <= SDMON_Y_MAX; y++)
	{
	  pos.xxyy = 100*x+y;
	  pos.towerid=get_towerid(bsdinfo->yymmdd,x,y);
	  if(pos.towerid == -1)
	    continue; // ignore the position ID if counter is absent
	  if(!get_xyz(bsdinfo->yymmdd,x,y,pos.xyz))
	    continue; // ignore counter if don't have GPS coordinates (also prints a warning message)
	  counter_status_map[pos.xxyy] = 0;
	}
    }
  // Next loop over the bsdinfo bank and label non-working counters
  // in the map
  for (int i=0; i<bsdinfo->nbsds; i++)
    {
      if(bsdinfo->bitf[i])
	counter_status_map[bsdinfo->xxyy[i]] = 1;
    }
  for (int i=0; i<bsdinfo->nsdsout; i++)
    {
      if(bsdinfo->bitfout[i])
	counter_status_map[bsdinfo->xxyyout[i]] = 2;
    }
  return counter_status_map;
}

integer4 sdxyzclf_class::get_event_surroundedness(rusdraw_dst_common  *rusdraw,
						  rusdgeom_dst_common *rusdgeom,
						  bsdinfo_dst_common  *bsdinfo)
{
  // 1. obtain a list of all counters with a flag that indicates whether they work or not
  map<int,int>& status_map = get_counter_status_map(bsdinfo);
  
  // 2. Find the largest signal size counter using only working counters
  double qmax = 0.0;
  int xxyy_largest_q = rusdgeom->xxyy[0];
  for (int isd=0; isd<rusdgeom->nsds; isd++)
    {
      // use only working counters that are part of the event
      if(rusdgeom->igsd[isd] < 2 || status_map[rusdgeom->xxyy[isd]])
	continue;
      if (qmax < rusdgeom->pulsa[isd])
	{
	  qmax = rusdgeom->pulsa[isd];
	  xxyy_largest_q = rusdgeom->xxyy[isd];
	}
    }
  // if there are no working counters that are part of the event the return 0 for
  // the surroundedness flag
  if(fabs(qmax) < 1e-3)
    return 0;
  
  int xx_largest_q = xxyy_largest_q / 100;
  int yy_largest_q = xxyy_largest_q % 100 ;
  
  // 3. define flags that indicate
  // 3.1 that there are working counters to the left, right, below, above the largest signal counter
  bool level_1_surroundedness[4] = {false,false,false,false};
  // 3.2 that there are working neighbor counters to the left, right, below, above the largest signal counter
  bool level_2_surroundedness[4] = {false,false,false,false};
  
  // 4.1 search for working counter to the left of the largest signal counter
  for (int xx = xx_largest_q-1; xx >= 1;  xx--)
    { 
      // don't use if the SD position doesn't exists or doesn't belong to a tower that has triggered on the event
      int tower_id = get_towerid(rusdraw->yymmdd,xx,yy_largest_q);
      if(tower_id < 0)
	continue;
      if(!tower_part_of_rusdraw_site(tower_id,rusdraw->site))
	continue;
      // don't use if the existing SD of a tower that triggered had problems
      if(status_map[xx*100+yy_largest_q])
	continue;
      // we have a working counter to the left of the largest signal counter
      level_1_surroundedness[0] = true;
      // we have a working counter immediately to the left of the largest signal counter
      if(xx == xx_largest_q-1)
	level_2_surroundedness[0] = true;
      // done with the search
      break;
    }
   // 4.2 search for working counter to the right of the largest signal counter
  for (int xx = xx_largest_q+1; xx <= SDMON_X_MAX;  xx++)
    { 
      int tower_id = get_towerid(rusdraw->yymmdd,xx,yy_largest_q);
      if(tower_id < 0)
	continue;
      if(!tower_part_of_rusdraw_site(tower_id,rusdraw->site))
	continue;
      if(status_map[xx*100+yy_largest_q])
	continue;
      level_1_surroundedness[1] = true;
      if(xx == xx_largest_q+1)
	level_2_surroundedness[1] = true;
      break;
    }
  // 4.3 search for working counter below the largest signal counter
  for (int yy = yy_largest_q-1; yy >= 1;  yy--)
    { 
      int tower_id = get_towerid(rusdraw->yymmdd,xx_largest_q,yy);
      if(tower_id < 0)
	continue;
      if(!tower_part_of_rusdraw_site(tower_id,rusdraw->site))
	continue;
      if(status_map[xx_largest_q*100+yy])
	continue;
      level_1_surroundedness[2] = true;
      if(yy == yy_largest_q-1)
	level_2_surroundedness[2] = true;
      break;
    }
  // 4.4 search for working counter below the largest signal counter
  for (int yy = yy_largest_q+1; yy <= SDMON_Y_MAX;  yy++)
    { 
      int tower_id = get_towerid(rusdraw->yymmdd,xx_largest_q,yy);
      if(tower_id < 0)
	continue;
      if(!tower_part_of_rusdraw_site(tower_id,rusdraw->site))
	continue;
      if(status_map[xx_largest_q*100+yy])
	continue;
      level_1_surroundedness[3] = true;
      if(yy == yy_largest_q+1)
	level_2_surroundedness[3] = true;
      break;
    }
  // 5.1 calculate and return the surroundedness flag
  
  // 0 if the largest signal counter is not surrounded by 4 working counters
  integer4 surroundedness = 0;
  
  // 1 if the largest signal counter is surrounded by 4 working counters but
  // not all are immediate neighbors of the largest signal counter
  if(level_1_surroundedness[0] && level_1_surroundedness[1] 
     && level_1_surroundedness[2] && level_1_surroundedness[3])
    surroundedness = 1;
  
  // 2 if the largest signal counter is surrounded by 4 working counters and
  // all 4 counters are the immediate neighbors of the largest signal counter
  if(level_2_surroundedness[0] && level_2_surroundedness[1] 
     && level_2_surroundedness[2] && level_2_surroundedness[3])
    surroundedness = 2;
  
  return surroundedness;
  
}

void sdxyzclf_class::talex00_2_rusdraw(const talex00_dst_common* talex00, rusdraw_dst_common* rusdraw)
{
  // convert talex00 DST bank to rusdraw DST bank for the counters that are
  // part of the main TASD array
  rusdraw->event_num = talex00->event_num;
  rusdraw->event_code = talex00->event_code;
  rusdraw->site = 0; // initialize
  rusdraw->errcode = talex00->errcode;
  rusdraw->yymmdd = talex00->yymmdd;
  rusdraw->hhmmss = talex00->hhmmss;
  rusdraw->usec = talex00->usec;
  rusdraw->monyymmdd = talex00->monyymmdd;
  rusdraw->monhhmmss = talex00->monhhmmss;
  int talex00_tower_ind[3]={BRCT,LRCT,SKCT};
  for (int i = 0; i < 3; i++)
    {
      rusdraw->run_id[i] = talex00->run_id[talex00_tower_ind[i]];
      rusdraw->trig_id[i] = talex00->trig_id[talex00_tower_ind[i]];
    }
  rusdraw->nofwf = 0;
  int tower_bitflag = 0; // to get the correct rusdraw site flag depending on what counters are present
  for (int iwf = 0; iwf < talex00->nofwf; iwf++)
    {
      // include only the waveforms that are part of the main TA SD array
      // (towerid not equal to -1)
      int towerid = get_towerid(talex00->yymmdd, talex00->xxyy[iwf]);
      if(towerid == -1)
	continue;
      tower_bitflag |= (1 << towerid); // combine the bit flags for the main SD towers
      rusdraw->nretry[rusdraw->nofwf] = talex00->nretry[iwf];
      rusdraw->wf_id[rusdraw->nofwf] = talex00->wf_id[iwf];
      rusdraw->trig_code[rusdraw->nofwf] = talex00->trig_code[iwf];
      rusdraw->xxyy[rusdraw->nofwf] = talex00->xxyy[iwf];
      rusdraw->clkcnt[rusdraw->nofwf] = talex00->clkcnt[iwf];
      rusdraw->mclkcnt[rusdraw->nofwf] = talex00->mclkcnt[iwf];
      for (int i = 0; i < 2; i++)
	{
	  rusdraw->fadcti[rusdraw->nofwf][i] = talex00->fadcti[iwf][i];
	  rusdraw->fadcav[rusdraw->nofwf][i] = talex00->fadcav[iwf][i];
	  for(int j = 0; j < rusdraw_nchan_sd; j++)
	    rusdraw->fadc[rusdraw->nofwf][i][j] = talex00->fadc[iwf][i][j];
	  rusdraw->pchmip[rusdraw->nofwf][i] = talex00->pchmip[iwf][i];
	  rusdraw->pchped[rusdraw->nofwf][i] = talex00->pchped[iwf][i];
	  rusdraw->lhpchmip[rusdraw->nofwf][i] = talex00->lhpchmip[iwf][i];
	  rusdraw->lhpchped[rusdraw->nofwf][i] = talex00->lhpchped[iwf][i];
	  rusdraw->rhpchmip[rusdraw->nofwf][i] = talex00->rhpchmip[iwf][i];
	  rusdraw->rhpchped[rusdraw->nofwf][i] = talex00->rhpchped[iwf][i];
	  rusdraw->mftndof[rusdraw->nofwf][i] = talex00->mftndof[iwf][i];
	  rusdraw->mip[rusdraw->nofwf][i] = talex00->mip[iwf][i];
	  rusdraw->mftchi2[rusdraw->nofwf][i] = talex00->mftchi2[iwf][i];
	  for(int j = 0; j < 4; j++)
	    {
	      rusdraw->mftp[rusdraw->nofwf][i][j] = talex00->mftp[iwf][i][j];
	      rusdraw->mftpe[rusdraw->nofwf][i][j] = talex00->mftpe[iwf][i][j];
	    }
	}
      rusdraw->nofwf++;
    }
  rusdraw->site = rusdraw_site_from_bitflag(tower_bitflag);
}
