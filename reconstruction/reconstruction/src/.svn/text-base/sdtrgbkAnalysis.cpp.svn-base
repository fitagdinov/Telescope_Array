#include "sdtrgbkAnalysis.h"
#include "sdmc_bsd_bitf.h"
#include "sdxyzclf_class.h"
#include <string.h>

static sdxyzclf_class sdxyzclf;

void sdtrgbkAnalysis::load_sds(tasdevent_dst_common *p)
{
  int xxyy = -1;
  char tstamp[0x100];
  sprintf(tstamp,"%06d:%06d.%06d",p->date,p->time,p->usec);
  for (int iwf = 0; iwf < p->num_trgwf; iwf++)
    {
      if ((p->sub[iwf].lid == xxyy) && (n_sdinfo_buffer == 0))
        {
          printWarn("%s Invalid xxyy: %04d", tstamp, p->sub[iwf].lid);
          n_bad_sds++;
          continue;
        }
      if (p->sub[iwf].lid != xxyy)
        {
	  if(sdxyzclf.get_towerid(p->date,p->sub[iwf].lid) == -1)
	    {
	      if(opt.verbosity >= 1)
		printWarn("%s %04d not used in the analysis",tstamp,p->sub[iwf].lid);
	      n_bad_sds++;
	      continue;
	    }
	  if(is_in_bsd_list(p->sub[iwf].lid))
	    {
	      if(opt.verbosity >= 2)
		fprintf(stdout,"NOTICE: %s %04d excluded; listed in bsdinfo\n",tstamp,p->sub[iwf].lid);
	      n_bad_sds++;
	      continue;
	    }
          if (n_sdinfo_buffer >= NSDMAX)
            {
              printWarn("%s number of SDs exceeds the maximum: %d", tstamp, NSDMAX);
              n_bad_sds++;
              continue;
            }
          if (!sdinfo_buffer[n_sdinfo_buffer].init_sd(p, n_sdinfo_buffer, iwf))
            {
              n_bad_sds++;
              continue;
            }
          xxyy = p->sub[iwf].lid;
          n_sdinfo_buffer++;
        }
      sdinfo_buffer[n_sdinfo_buffer - 1].add_wf(p, iwf);
    }
  yymmdd = p->date;
  hhmmss = p->time;
  usec = p->usec;
}

void sdtrgbkAnalysis::load_sds(tasdcalibev_dst_common *p)
{
  int xxyy = -1;
  char tstamp[0x100];
  sprintf(tstamp,"%06d:%06d.%06d",p->date,p->time,p->usec);
  for (int iwf = 0; iwf < p->numTrgwf; iwf++)
    {
      if ((p->sub[iwf].lid == xxyy) && (n_sdinfo_buffer == 0))
        {
          printWarn("Invalid xxyy: %04d", p->sub[iwf].lid);
          n_bad_sds++;
          continue;
        }
      if (p->sub[iwf].lid != xxyy)
        {
	  if(sdxyzclf.get_towerid(p->date,p->sub[iwf].lid) == -1)
	    {
	      if(opt.verbosity >= 1)
		printWarn("%s %04d not used in the analysis",tstamp,p->sub[iwf].lid);
		  n_bad_sds++;
	      continue;
	    }
	  if(is_in_bsd_list(p->sub[iwf].lid))
	    {
	      if(opt.verbosity >= 2)
		fprintf(stdout,"NOTICE: %s %04d excluded; listed in bsdinfo\n",tstamp,p->sub[iwf].lid);
	      n_bad_sds++;
	      continue;
	    }
          if (n_sdinfo_buffer >= NSDMAX)
            {
              printWarn("%s number of SDs exceeds the maximum: %d", tstamp, NSDMAX);
              n_bad_sds++;
              continue;
            }
          if (!sdinfo_buffer[n_sdinfo_buffer].init_sd(p, n_sdinfo_buffer, iwf))
            {
              n_bad_sds++;
              continue;
            }
          xxyy = p->sub[iwf].lid;
          n_sdinfo_buffer++;
        }
      sdinfo_buffer[n_sdinfo_buffer - 1].add_wf(p, iwf);
    }
  yymmdd = p->date;
  hhmmss = p->time;
  usec = p->usec;
}

void sdtrgbkAnalysis::load_sds(rusdraw_dst_common *p)
{
  int xxyy = -1;
  char tstamp[0x100];
  sprintf(tstamp,"%06d:%06d.%06d",p->yymmdd,p->hhmmss,p->usec);
  for (int iwf = 0; iwf < p->nofwf; iwf++)
    {
      if(sdxyzclf.get_towerid(p->yymmdd,p->xxyy[iwf]) == -1)
	{
	  if(opt.verbosity >= 1)
	    printWarn("%s %04d not used in the analysis",tstamp,p->xxyy[iwf]);
	  n_bad_sds++;
	  continue;
	}
      if ((p->xxyy[iwf] == xxyy) && (n_sdinfo_buffer == 0))
        {
          printWarn("%s Invalid xxyy: %04d", tstamp, p->xxyy[iwf]);
          n_bad_sds++;
          continue;
        }
      if (p->xxyy[iwf] != xxyy)
        {
	  if(is_in_bsd_list(p->xxyy[iwf]))
	    {
	      if(opt.verbosity >= 2)
		fprintf(stdout,"NOTICE: %s %04d excluded; listed in bsdinfo\n",tstamp, p->xxyy[iwf]);
	      n_bad_sds++;
	      continue;
	    }
          if (n_sdinfo_buffer >= NSDMAX)
            {
              printWarn("%s number of SDs exceeds the maximum: %d", tstamp, NSDMAX);
              n_bad_sds++;
              continue;
            }
          if (!sdinfo_buffer[n_sdinfo_buffer].init_sd(p, n_sdinfo_buffer, iwf))
            {
              n_bad_sds++;
              continue;
            }
          xxyy = p->xxyy[iwf];
          n_sdinfo_buffer++;
        }
      sdinfo_buffer[n_sdinfo_buffer - 1].add_wf(p, iwf);
    }
  yymmdd = p->yymmdd;
  hhmmss = p->hhmmss;
  usec = p->usec;
}



void sdtrgbkAnalysis::load_bad_sds(bsdinfo_dst_common *bsdinfo)
{
  if(opt.ignore_bsdinfo)
    return;
  for (int i=0; i<bsdinfo->nbsds; i++)
    {
      integer4 not_working_according_to_sdmc_checks;
      sdmc_calib_check_bitf(bsdinfo->bitf[i],&not_working_according_to_sdmc_checks,0);
      if(not_working_according_to_sdmc_checks)
	bsd_list[bsdinfo->xxyy[i]] = bsdinfo->bitf[i];
    }
}

bool sdtrgbkAnalysis::is_in_bsd_list(int xxyy)
{
  return (bsd_list.find(xxyy) != bsd_list.end());
}

void sdtrgbkAnalysis::pick_good_ped()
{
  int isd;
  sdinfo_class *p;
  sd_good_ped.clear();
  char tstamp[0x100];
  sprintf(tstamp,"%06d:%06d.%06d",yymmdd,hhmmss,usec);
  for (isd = 0; isd < n_sdinfo_buffer; isd++)
    {
      // Exclude SDs with bad pedestals
      if ((sdinfo_buffer[isd].ped[0] < 1.0) || (sdinfo_buffer[isd].ped[1] < 1.0))
        {
	  printWarn("%s BAD PEDESTAL: XXYY = %04d ped[0] = %d ped[1] = %d", 
		    tstamp, sdinfo_buffer[isd].xxyy, sdinfo_buffer[isd].ped[0], sdinfo_buffer[isd].ped[1]);
          n_bad_sds++;
          continue;
        }
      p = &sdinfo_buffer[isd];
      sd_good_ped.push_back(p);
    }
}

void sdtrgbkAnalysis::pick_cont_sd()
{
  char spcxy[SDMON_X_MAX][SDMON_Y_MAX];
  char sptmcxy[SDMON_X_MAX][SDMON_Y_MAX];
  vector<sdinfo_class *>::iterator isd, jsd, ksd;
  memset(spcxy, 0, sizeof(spcxy));
  memset(sptmcxy, 0, sizeof(sptmcxy));
  sd_spat_cont.clear();
  sd_spat_isol.clear();
  sd_pot_spat_tim_cont.clear();
  if (sd_good_ped.size() == 0)
    return;
  for (isd = sd_good_ped.begin(); isd != sd_good_ped.end(); isd++)
    {
      for (jsd = isd; jsd != sd_good_ped.end(); jsd++)
        {
          if (jsd == isd)
            continue;
          for (ksd = jsd; ksd != sd_good_ped.end(); ksd++)
            {
              if (ksd == jsd)
                continue;
              if (space_pat_recog((*isd)->xxyy, (*jsd)->xxyy, (*ksd)->xxyy))
                {

                  spcxy[((*isd)->xxyy) / 100 - 1][((*isd)->xxyy) % 100 - 1] = 1;
                  spcxy[((*jsd)->xxyy) / 100 - 1][((*jsd)->xxyy) % 100 - 1] = 1;
                  spcxy[((*ksd)->xxyy) / 100 - 1][((*ksd)->xxyy) % 100 - 1] = 1;
                  if (in_time((*isd)->tlim, (*jsd)->tlim, (*ksd)->tlim, L2TWND))
                    {
                      sptmcxy[((*isd)->xxyy) / 100 - 1][((*isd)->xxyy) % 100 - 1] = 1;
                      sptmcxy[((*jsd)->xxyy) / 100 - 1][((*jsd)->xxyy) % 100 - 1] = 1;
                      sptmcxy[((*ksd)->xxyy) / 100 - 1][((*ksd)->xxyy) % 100 - 1] = 1;
                    }
                }
            }
        }

      // SDs that participate in spatial trigger patterns
      if (spcxy[((*isd)->xxyy) / 100 - 1][((*isd)->xxyy) % 100 - 1])
        sd_spat_cont.push_back((*isd));

      // SDs that do not participate in spatial trigger patterns
      else
        sd_spat_isol.push_back((*isd));

      // potentially space-time contiguous SDs
      if (sptmcxy[((*isd)->xxyy) / 100 - 1][((*isd)->xxyy) % 100 - 1])
        sd_pot_spat_tim_cont.push_back((*isd));
    }
}

bool sdtrgbkAnalysis::find_level2_trigger(int DeltaPed)
{

  int isig, jsig, ksig; // for looping over signals in the triplet of SDs

  // If there is no 3 counters that are potentially contiguous in space and time
  // (their time bounds are contiguous and they meet the trigger pattern)
  // return false.
  if (sd_pot_spat_tim_cont.size() < 3)
    {
      printWarn("Number of potentially space-time contiguous SDs is %d < 3", sd_pot_spat_tim_cont.size());
      return false;
    }

  // Out of potentially space-time contiguous SDs, pick those that have level-1 trigger signals
  l1sd.clear();
  for (vector<sdinfo_class *>::iterator isd = sd_pot_spat_tim_cont.begin(); isd != sd_pot_spat_tim_cont.end(); isd++)
    {
      if ((*isd)->find_l1_sig_correct(DeltaPed) == 0)
        continue;
      // if ((*isd)->find_l1_sig(DeltaPed) == 0)
      //   continue;
      l1sd.push_back((*isd));
    }

  // if the number of level-1 trigger SDs is less than 3 for this pedestal configuration,
  // return false
  if (l1sd.size() < 3)
    return false;

  // Find level-2 trigger
  l2sd.clear();
  for (vector<sdinfo_class *>::iterator isd = l1sd.begin(); isd != l1sd.end(); isd++)
    {
      for (vector<sdinfo_class *>::iterator jsd = isd; jsd != l1sd.end(); jsd++)
        {
          if (jsd == isd)
            continue;
          for (vector<sdinfo_class *>::iterator ksd = jsd; ksd != l1sd.end(); ksd++)
            {
              if (ksd == jsd)
                continue;

              // Triplet of SDs that is spatially contiguous
              if ((space_pattern = space_pat_recog((*isd)->xxyy, (*jsd)->xxyy, (*ksd)->xxyy)))
                {
                  // Loop over all level-1 trigger signals in these 3 SDs,
                  // find if there are signals in the SDs that are in time
                  for (isig = 0; isig < (*isd)->nl1; isig++)
                    {
                      for (jsig = 0; jsig < (*jsd)->nl1; jsig++)
                        {
                          for (ksig = 0; ksig < (*ksd)->nl1; ksig++)
                            {
                              if (in_time((*isd)->secf[isig], (*jsd)->secf[jsig], (*ksd)->secf[ksig], L2TWND))
                                {
                                  // save the information on signals which cause the level-2 trigger
                                  (*isd)->il2sig = isig;
                                  (*jsd)->il2sig = jsig;
                                  (*ksd)->il2sig = ksig;
                                  // save those level-1 SDs (and their signals) that casued
                                  // the level-2 trigger
                                  l2sd.push_back((*isd));
                                  l2sd.push_back((*jsd));
                                  l2sd.push_back((*ksd));
                                  return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
  return false;
}

bool sdtrgbkAnalysis::find_level2_trigger_lower_ped()
{

  int d_ped = 0;
  has_triggered = false;

  // No point in trying to find the trigger
  // if the number of potentially space-time contiguous
  // SDs is less than 3
  if (sd_pot_spat_tim_cont.size() < 3)
    return false;
  while (d_ped >= (-NL1cnt))
    {
      if ((has_triggered = find_level2_trigger(d_ped)))
        break;
      d_ped--;
    }
  return has_triggered;
}

int sdtrgbkAnalysis::find_level2_trigger_raise_ped()
{
  int d_ped = 0;

  // this routine applies only to events which
  // have triggered without decreasing the pedestals
  if (has_triggered == false)
    return 0;
  if (l1sd.size() < 3)
    {
      printErr("find_level2_trigger_raise_ped: assumed that the event has triggered but the number of level-1 SDs < 3");
      return 0;
    }
  // If the event had to be triggered
  // with lowered pedestals then
  // there is no reason to try to increase the pedestal
  // and try to re-trigger the event
  if (l1sd[0]->d_ped < 0)
    return 0;

  while (d_ped < NL1cnt)
    {
      if (find_level2_trigger(d_ped + 1))
        d_ped++;
      else
        break;
    }
  // important, so that the
  // level-2 trigger information is retained for the last value of d_ped
  // when event still triggers
  if (d_ped > 0)
    find_level2_trigger(d_ped);
  return d_ped;
}

bool sdtrgbkAnalysis::in_time(double t1, double t2, double t3, double time_window)
{
  if ((fabs(t2 - t1) > time_window) || (fabs(t3 - t1) > time_window) || (fabs(t3 - t2) > time_window))
    return false;
  return true;
}

bool sdtrgbkAnalysis::in_time(double t1[2], double t2[2], double t3[2], double time_window)
{
  if (((t1[0]-t2[1])>time_window) || ((t2[0]-t1[1])>time_window))
    return false; // if the first two SDs are further apart than the time window then return false
  // time boundary within which the 3rd signal needs to be
  double t[2] = 
    {
      t1[0]>t2[0] ? t1[0] : t2[0], // lower limit = latest start time of the two counters
      t1[1]<t2[1] ? t1[1] : t2[1]  // upper limit = earliest stop time of the two counters
    };
  // if 3rd counter signal ends earlier than time_window from where the latest signal of other two counters starts
  if(t[0] - t3[1] > time_window)
    return false;
  // if 3rd counter signal starts later than time_window from where the earliest signal of other two counters ends
  if(t3[0] - t[1] > time_window)
    return false;
  return true;
}

int sdtrgbkAnalysis::space_pat_recog(int xxyy1, int xxyy2, int xxyy3)
{
  int i, j, xy[3][2];
  int d2; // square distances b/w a pair of counters
  int d2max; // maximum square distance b/w a pair of counters in the triplet
  int found_grid;
  xy[0][0] = xxyy1 / 100;
  xy[0][1] = xxyy1 % 100;
  xy[1][0] = xxyy2 / 100;
  xy[1][1] = xxyy2 % 100;
  xy[2][0] = xxyy3 / 100;
  xy[2][1] = xxyy3 % 100;
  // there should be at least one pair of SDs that's on the grid; find it.
  d2max = 0;
  found_grid = 0;
  for (i = 0; i < 3; i++)
    {
      for (j = 0; j < 3; j++)
        {
          if (j == i)
            continue;
          // if there is a pair of SDs whose square separation
          // distance is larger than 4,
          // space trigger pattern is not met for the whole triplet.
          if ((d2 = (xy[i][0] - xy[j][0]) * (xy[i][0] - xy[j][0]) + (xy[i][1] - xy[j][1]) * (xy[i][1] - xy[j][1])) > 4)
            return 0;
          if (d2 > d2max)
            d2max = d2;
          if (d2 == 1)
            found_grid = 1;
        }
    }
  return found_grid * d2max;
}


// assumes that the SD information is already loaded
void sdtrgbkAnalysis::analyze_event()
{
 
  vector<sdinfo_class *>::iterator isd;
  static int ievent = 0;

  pick_good_ped(); // pick SDs with good pedestals

  pick_cont_sd(); // pick SDs that are contiguous in space and pick SDs that might be contiguous in time

  // find a triplet of SDs that causes the level-2 (event) trigger
  // if can't find on the first trial, it will keep lowering the pedestals until
  // level-2 trigger is found.
  find_level2_trigger_lower_ped();

  // Put trigger information ( including lowered pedestals info, if necessary) into the
  // sdtrgbk DST bank
  put2sdtrgbk_triginfo();

  // Print out the trigger information
  if (opt.verbosity >= 2)
    {
      fprintf(stdout, "\n\n -----------------   EVENT %6d  %06d:%06d.%06d   ------------------------\n", ievent,
          yymmdd, hhmmss, usec);
      fflush(stdout);
      fprintf(stdout, "NUMBER OF SDS: %d\n", GetNSD());
      fprintf(stdout, "NUMBER OF GOOD PED SDS: %d\n", getNgoodPedSD());

      fprintf(stdout, "ISOLATED SDS (%d): ", (int) sd_spat_isol.size());
      for (isd = sd_spat_isol.begin(); isd != sd_spat_isol.end(); isd++)
        fprintf(stdout, "%04d(%d) ", (*isd)->xxyy, (*isd)->nwf);
      fprintf(stdout, "\n");

      fprintf(stdout, "SPATIALLY CONT. SDS(%d): ", (int) sd_spat_cont.size());
      for (isd = sd_spat_cont.begin(); isd != sd_spat_cont.end(); isd++)
        fprintf(stdout, "%04d(%d) ", (*isd)->xxyy, (*isd)->nwf);
      fprintf(stdout, "\n");

      fprintf(stdout, "POTENTIALLY SPACE-TIME CONT. SDS(%d): ", (int) sd_pot_spat_tim_cont.size());
      for (isd = sd_pot_spat_tim_cont.begin(); isd != sd_pot_spat_tim_cont.end(); isd++)
        fprintf(stdout, "%04d(%d) ", (*isd)->xxyy, (*isd)->nwf);
      fprintf(stdout, "\n");

      fprintf(stdout, "LEVEL-1 SDS (%d): ", (int) l1sd.size());
      for (isd = l1sd.begin(); isd != l1sd.end(); isd++)
        fprintf(stdout, "%04d(%d) ", (*isd)->xxyy, (*isd)->nwf);
      fprintf(stdout, "\n");

      fprintf(stdout, "LEVEL-2 TRIGGER FOUND: %s\n",  (has_triggered ? "YES" : "NO"));

      if (has_triggered)
        {
          fprintf(stdout, "LEVEL-2 SD:  DEC_PED=%d | ", (-l2sd[0]->d_ped));
          for (isd = l2sd.begin(); isd != l2sd.end(); isd++)
            fprintf(stdout, "XXYY=%04d SECF=%.6f Qu=%d Ql=%d |  ", (*isd)->xxyy, (*isd)->secf[(*isd)->il2sig],
                (*isd)->q[(*isd)->il2sig][0], (*isd)->q[(*isd)->il2sig][1]);
          fprintf(stdout, "\n");
        }

      fflush(stdout);
    }

  if (has_triggered && sdtrgbk_.igevent == 2)
    {
      sdtrgbk_.inc_ped = (integer2) find_level2_trigger_raise_ped();
      // increase the goodness of the event
      // if it can trigger with raised pedestals
      if (sdtrgbk_.inc_ped > 0)
        sdtrgbk_.igevent = 3;

      // SDs that participated in the event trigger with raised pedestals
      for (isd = l2sd.begin(); isd != l2sd.end(); isd++)
        {
          if (sdtrgbk_.ig[(*isd)->sdindex] == 5)
            sdtrgbk_.ig[(*isd)->sdindex] = 7;
          else
            sdtrgbk_.ig[(*isd)->sdindex] = 6;
        }
    }
  else
    sdtrgbk_.inc_ped = 0;

  if (opt.verbosity >= 2)
    {
      fprintf(stdout, "LEVEL-2-INC-PED TRIGGER FOUND %s\n", (sdtrgbk_.inc_ped > 0 ? "YES" : "NO" ));
      if (sdtrgbk_.inc_ped > 0)
        {
          fprintf(stdout, "LEVEL-2-INC-PED SD:  INC_PED=%d | ", l2sd[0]->d_ped);
          for (isd = l2sd.begin(); isd != l2sd.end(); isd++)
            fprintf(stdout, "XXYY=%04d SECF=%.6f Qu=%d Ql=%d |  ", (*isd)->xxyy, (*isd)->secf[(*isd)->il2sig],
                (*isd)->q[(*isd)->il2sig][0], (*isd)->q[(*isd)->il2sig][1]);
          fprintf(stdout, "\n");
        }
    }

  ievent++;

}

void sdtrgbkAnalysis::put2sdtrgbk_triginfo()
{

  int i;
  memset(&sdtrgbk_, 0, sizeof(sdtrgbk_));

  if (has_triggered)
    {
      sdtrgbk_.igevent = 1;

      switch (space_pattern)
        {
        case 2:
          sdtrgbk_.trigp = 0;
          break;
        case 4:
          sdtrgbk_.trigp = 1;
          break;
        default:
          sdtrgbk_.trigp = -1;
          break;
        }
    }
  else
    {
      // Event did not trigger even with lowered pedestals
      sdtrgbk_.igevent = 0;

      // Set the trigger pattern to 3
      // (if couldn't find 3 SDs out of
      // level-1 SDs in potentially space-time contiguous
      // set that are in time)
      sdtrgbk_.trigp = 3;

      // Set the trigger pattern to 2
      // if couldn't find 3 level-1 signals out of
      // potentially space-time contiguous SDs
      if (l1sd.size() < 3)
        sdtrgbk_.trigp = 2;

      // set trigger pattern to 1 if couldn't find
      // 3 potentially space-time contiguous SDs
      // out of spatially contiguous SDs
      if (sd_pot_spat_tim_cont.size() < 3)
        sdtrgbk_.trigp = 1;

      // set trigger pattern to 0 if couldn't
      // find 3 space contiguous SDs
      if (sd_spat_cont.size() < 3)
        sdtrgbk_.trigp = 0;
    }

  // raw SD data bank used
  if (n_sdinfo_buffer == 0)
    sdtrgbk_.raw_bankid = RUSDRAW_BANKID;
  else
    sdtrgbk_.raw_bankid = sdinfo_buffer[0].raw_bankid;

  sdtrgbk_.nsd = (integer2) n_sdinfo_buffer;
  sdtrgbk_.n_bad_ped = (integer2) (n_sdinfo_buffer - sd_good_ped.size());
  sdtrgbk_.n_spat_cont = (integer2) sd_spat_cont.size();
  sdtrgbk_.n_isol = (integer2) sd_spat_isol.size();
  sdtrgbk_.n_pot_st_cont = (integer2) sd_pot_spat_tim_cont.size();
  sdtrgbk_.n_l1_tg = (integer2) l1sd.size();
  sdtrgbk_.dec_ped = 0;
  if (l1sd.size() > 0)
    sdtrgbk_.dec_ped = (integer2) (-l1sd[0]->d_ped); // save the absolute value of pedestal decrease

  if (has_triggered)
    {
      // Increase the goodness of the event if it has triggered
      // without lowering the pedestal
      if (sdtrgbk_.dec_ped == 0)
        sdtrgbk_.igevent = 2;

      if (l2sd.size() < 3)
        {
          printErr("FATAL: event has triggered but # of SDs in the level-2 SD vector is %d", l2sd.size());
          exit(2);
        }
      for (i = 0; i < 3; i++)
        {
          sdtrgbk_.il2sd[i] = (integer2) l2sd[i]->sdindex;
          sdtrgbk_.il2sd_sig[i] = (integer2) l2sd[i]->il2sig;
        }
    }

  // Put in the information on all counters
  for (int isd = 0; isd < n_sdinfo_buffer; isd++)
    {
      sdtrgbk_.xxyy[isd] = (integer2) sdinfo_buffer[isd].xxyy;
      sdtrgbk_.wfindex_cal[isd] = (integer2) sdinfo_buffer[isd].wfindex_cal;
      sdtrgbk_.nl1[isd] = (integer2) sdinfo_buffer[isd].nl1;
      sdtrgbk_.tlim[isd][0] = sdinfo_buffer[isd].tlim[0];
      sdtrgbk_.tlim[isd][1] = sdinfo_buffer[isd].tlim[1];
      for (i = 0; i < sdtrgbk_.nl1[isd]; i++)
        {
          sdtrgbk_.secf[isd][i] = sdinfo_buffer[isd].secf[i];
          sdtrgbk_.ich[isd][i] = (integer2) sdinfo_buffer[isd].ich[i];
          sdtrgbk_.l1sig_wfindex[isd][i] = (integer2) sdinfo_buffer[isd].wfindex[sdinfo_buffer[isd].iwfsd[i]];
          sdtrgbk_.q[isd][i][0] = (integer2) sdinfo_buffer[isd].q[i][0];
          sdtrgbk_.q[isd][i][1] = (integer2) sdinfo_buffer[isd].q[i][1];
        }
      sdtrgbk_.ig[isd] = 0; // initialize SD goodness flag
    }

  vector<sdinfo_class*>::iterator isd;

  // SDs that have good pedestals
  for (isd = sd_good_ped.begin(); isd != sd_good_ped.end(); isd++)
    sdtrgbk_.ig[(*isd)->sdindex] = 1;

  // SDs that participate in spatial trigger patterns
  for (isd = sd_spat_cont.begin(); isd != sd_spat_cont.end(); isd++)
    sdtrgbk_.ig[(*isd)->sdindex] = 2;

  // SDs that are potentially space-time contiguous
  for (isd = sd_pot_spat_tim_cont.begin(); isd != sd_pot_spat_tim_cont.end(); isd++)
    sdtrgbk_.ig[(*isd)->sdindex] = 3;

  // SDs that are potentially space-time contiguous and have level-1 signals in them
  for (isd = l1sd.begin(); isd != l1sd.end(); isd++)
    sdtrgbk_.ig[(*isd)->sdindex] = 4;

  // SDs that participated in the event trigger ( with or w/o
  // lowering the pedestals were as necessary to trigger the event)
  for (isd = l2sd.begin(); isd != l2sd.end(); isd++)
    sdtrgbk_.ig[(*isd)->sdindex] = 5;

}

void sdtrgbkAnalysis::analyzeEvent(tasdevent_dst_common *tasdeventp, bsdinfo_dst_common *bsdinfo)
{
  init();
  load_bad_sds(bsdinfo);
  load_sds(tasdeventp); // load sd and waveform information from tasdevent bank
  analyze_event();
}
void sdtrgbkAnalysis::analyzeEvent(tasdcalibev_dst_common *tasdcalibevp, bsdinfo_dst_common *bsdinfo)
{
  init();
  load_bad_sds(bsdinfo);
  load_sds(tasdcalibevp); // load sd and waveform information from tasdcalibev bank
  analyze_event();
}
void sdtrgbkAnalysis::analyzeEvent(rusdraw_dst_common *rusdrawp, bsdinfo_dst_common *bsdinfo)
{
  init(); // clear all buffers from previous events
  load_bad_sds(bsdinfo);
  load_sds(rusdrawp); // load sd and waveform information from rusdraw bank
  analyze_event(); // trigger backup analysis
}


void sdtrgbkAnalysis::printErr(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "\nERROR (sdtrgbkAnalysis): %s\n", mess);
}

void sdtrgbkAnalysis::printWarn(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "WARNING (sdtrgbkAnalysis): %s\n", mess);
}
