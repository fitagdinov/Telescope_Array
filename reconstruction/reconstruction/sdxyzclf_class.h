#ifndef _SDXYZCLF_H_
#define _SDXYZCLF_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <event.h>
#include <vector>
#include <map>

// SD origin with respect to CLF, in [1200m] units
#define SD_ORIGIN_X_CLF -12.2435
#define SD_ORIGIN_Y_CLF -16.4406

// SD coordinates in CLF frame in a usable form.
class sdxyzclf_class
{
public:
  sdxyzclf_class();
  bool get_xyz(int yymmdd, int x, int y, double *xyz);
  bool get_xyz(int yymmdd, int xxyy, double *xyz);
  // -1: Counter is not there, 0-BR, 1-LR, 2-SK
  int  get_towerid(int yymmdd, int x, int y);
  int  get_towerid(int yymmdd, int xxyy);
  
  // tell if a given site (0,1,2) is a part of site flag in rusdraw
  bool tower_part_of_rusdraw_site(int tower_id, int rusdraw_site);
  
  // Get closest counter to (2D) xy[2] point
  // yymmdd - date (INPUT)
  // xy[2] - 2D point in CLF frame in XY plane, [meters] (INPUT)
  // xxyy - closest counter position ID (OUTPUT)
  // dr[2] - 2D vector which points from the xy point to the closest counter, [meters] (OUTPUT)
  void get_closest_sd(int yymmdd, double *xy, int *xxyy, double *dr);

  // returns counter_status_map, index by counter position ID.
  // counter_status_map[0101] = 0  would mean counter 0101 is working and there is no problems
  // counter_status_map[0102] = 1 would mean counter 0102 is online but not working correctly
  // counter_status_map[0103] = 2 would mean counter 0102 is offline (data) or not simulated because
  //                              crucial live calibration information is missing (MC)
  // information is loaded from bsdinfo DST bank
  std::map<int,int>& get_counter_status_map(bsdinfo_dst_common *bsdinfo);

  // return the event surroundedness flag
  // 0: largest signal counter, that's part of the event, is not surrounded by 4 working counters
  // 1: largest signal counter, that's part of the event is surrounded by 4 woring counters
  //    (to the left, eight, down, up on the square grid)
  // 2: largest signal counter, that's part of the event is surrounded by 4 woring counters that
  //    are also the immediate neighbors of the largest signal counter
  integer4 get_event_surroundedness(rusdraw_dst_common  *rusdraw,
				    rusdgeom_dst_common *rusdgeom,
				    bsdinfo_dst_common  *bsdinfo);
  
  // convert talex00 DST bank to rusdraw DST bank for the counters that are
  // part of the main TASD array
  void talex00_2_rusdraw(const talex00_dst_common* talex00, rusdraw_dst_common* rusdraw);

  
  class sdpos
  {
  public:
    // position of counters in CLF frame 
    int xxyy;
    int towerid;
    double xyz[3];
    sdpos() 
    {
      xxyy    = 0;
      towerid = 0;
      xyz[0]  = 0;
      xyz[1]  = 0;
      xyz[2]  = 0;
    }
    ~sdpos() { ; }
  };
  
  // get all counters that are available on a given date
  std::vector<sdpos>& get_counters(int yymmdd);

private:
  int towerids_ds1[SDMON_X_MAX][SDMON_Y_MAX];
  int towerids_ds2[SDMON_X_MAX][SDMON_Y_MAX];
  int towerids_ds3[SDMON_X_MAX][SDMON_Y_MAX];
  double sdxyzclf_ds1[SDMON_X_MAX][SDMON_Y_MAX][3];
  double sdxyzclf_ds2[SDMON_X_MAX][SDMON_Y_MAX][3];
  double sdxyzclf_ds3[SDMON_X_MAX][SDMON_Y_MAX][3];
  std::vector<sdpos> Counters; /* vector that holds the counter information */
  std::map<int,int> counter_status_map;
};
#endif
