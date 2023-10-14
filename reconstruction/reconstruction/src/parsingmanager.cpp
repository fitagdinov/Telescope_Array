#include "parsingmanager.h"

parsingManager::parsingManager(const listOfOpt& passed_opt, rusdpass0io *rusdpass0io_pointer): opt(passed_opt)
{
  int itower, iievent, iimon;
  p0io=rusdpass0io_pointer;
  if (p0io == 0)
    {
      fprintf(stderr, "parsingManager: I/O class was not initialized\n");
      exit(2);
    }
  
  // intialize the indexer and pass the date so that it uses
  // a correct tower-detector layout
  sdi = new sdindex_class(opt.yymmdd);

  // Initialize the parsers for each tower
  for (itower=0; itower<3; itower++)
    {
      raw[itower] = new towerParser(opt,itower,p0io);
      need_data[itower] = true;
      have_data[itower] = true;
      read_data[itower] = true;
      nevents_tower_total[itower] = 0;
      nmon_tower_total[itower] = 0;
    }

  // Initialize buffers
  for (itower=0; itower < 3; itower ++)
    {
      // Buffers to hold read out events
      n_readout_events[itower] = 0;
      readout_events[itower] = new rusdraw_dst_common[NRUSDRAW_TOWER];

      for (iievent=0; iievent<NRUSDRAW_TOWER; iievent++)
	readout_event_index[itower][iievent] = iievent;

      // Buffers to hold read out moniotoring cycles
      n_readout_mon[itower] = 0;
      readout_mon[itower] = new sdmon_dst_common[NSDMON_TOWER];
      for (iimon=0; iimon<NSDMON_TOWER; iimon++)
	readout_mon_index[itower][iimon] = iimon;

    }

  // Buffer to hold time-matched events
  n_tmatched_events = 0;
  tmatched_events = new rusdraw_dst_common[NRUSDRAW_TMATCHED];
  for (iievent=0; iievent<NRUSDRAW_TMATCHED; iievent++)
    tmatched_event_index[iievent] = iievent;

  // Buffer to hold time-matched monitoring cycles
  n_tmatched_mon = 0;
  tmatched_mon = new sdmon_dst_common[NSDMON_TMATCHED];
  for (iimon=0; iimon<NSDMON_TMATCHED; iimon++)
    tmatched_mon_index[iimon] = iimon;

  // needed for 1mip fitter.
  hmip[0] = new TH1F("hmip0","mip fit (lower)",SDMON_NMONCHAN,-0.5,511.5);
  hmip[1] = new TH1F("hmip1","mip fit (upper)",SDMON_NMONCHAN,-0.5,511.5);
  ffit = 0;

  // Initialize the current monitoring cycle second
  cur_moncycle_second = SDGEN::time_in_sec_j2000(opt.yymmdd, 0);
  min_moncycle_second = cur_moncycle_second;
  max_moncycle_second = cur_moncycle_second+86400;
  curmon_num = 0;
  curevt_num = 0;

  have_good_mon = false; // Will be true once we have a good monitoring cycle in the buffer

  nevents_tmatched_total = 0; // total number of time-matched events
  nmon_tmatched_total = 0; // total number of time-matched mon. cycles
  
  n_duplicated_removed = 0; // number of duplicated events removed initialized
  
  written_event_times.resize(0); // initialize the vector of written events
  
  n_bad_calib = 0; // initialize the number of badly calibrated events
  
  fSuccess = 1; // default success is 1, unless changes in the course of running...
}
void parsingManager::Start()
{
  int itower;
  int readout_val;
  int yymmdd;
  while (true)
    {
      // Keep reading out the monitoring cycles for each tower and get out 
      // events when they appear
      for (itower = 0; itower < 3; itower ++)
	{
	  do
	    {
	      // Stop reading the tower if don't need its data or its data has ended              
	      if (!read_data[itower])
		break;

	      readout_val=raw[itower]->Parse();
	      // Depending on what the parser has read out:
	      switch (readout_val)
		{

		  //////////////////////////////////////////////////////////
		  ////////////// READ OUT AN EVENT.  SAVE IF OCCURED ///////
		  ////////////// ON DESIRED DATE ///////////////////////////
		  //////////////////////////////////////////////////////////
                case READOUT_EVENT:
                  {
                    yymmdd = raw[itower]->get_curEventDate();
                    if (yymmdd == opt.yymmdd)
                      {
                        need_data[itower] = save_event(itower);
                        nevents_tower_total[itower] ++;
                      }
                    else
                      {
                        raw[itower]->cleanEvent();
                        break;
                      }
                    if (yymmdd <= 0)
                      {
                        printErr(
				 "Event tower = %d came with wrong date: %06d\n",
				 itower, yymmdd);
                      }
                    break;
                  }

                  ///////////////////////////////////////////////////////////
                  ///// READ OUT A MONITORING CYCLE.  SAVE IF OCCURED ///////
                  ///// ON DESIRED DATE /////////////////////////////////////
                  ///////////////////////////////////////////////////////////
                case READOUT_MON:
                  {
                    yymmdd = raw[itower]->get_curMonDate();
                    if (yymmdd == opt.yymmdd)
                      {
                        need_data[itower] = save_mon(itower);
                        nmon_tower_total[itower] ++;
                      }
                    else
                      {
                        raw[itower]->cleanMonCycle();
                        break;
                      }
                    if (yymmdd <= 0)
                      {
                        printErr(
				 "Monitoring cycle = %d came with wrong date: %06d\n",
				 itower, yymmdd);
                      }
                    break;
                  }

                  // INPUT STREAM FOR A GIVEN TOWER HAS ENDED
                case READOUT_ENDED:
                  {
		    if(opt.verbose>=2)
		      {
			fprintf(stdout, "TOWER %d ENDED\n", itower);
			fflush(stdout);
		      }
                    have_data[itower] = false;
                    break;
                  }

                  // SERIOUS PARSING PROBLEM AND THE READOUT PROCESS NEEDS TO BE TERMINATED
                case READOUT_FAILURE:
                  {
                    fSuccess = 0;
                    break;
                  }
		  
                default:
                  {
                    break;
                  }
		}

	      // Stop reading the tower in case of failure
	      if (fSuccess == 0)
		break;

	      read_data[itower] = (need_data[itower] && have_data[itower]);

	    } while (readout_val != READOUT_MON);

	  // Stop reading any towers if there is a readout failure in either one of them
	  if (fSuccess == 0)
	    break;

	} // for (itower = 0; itower < 3; itower ++)

      // Abort the entire parsing process if there is a failure
      if (fSuccess == 0)
	break;

      // Process data once certain buffers are full or there is no more data for certain towers
      // If can't process the data, (data finished or have errors) then stop the read out
      if (!(read_data[0] || read_data[1] || read_data[2]))
	{
	  if (!process_data())
	    {
	      fSuccess=0;
	      break;
	    }
	  for (itower=0; itower<3; itower++)
	    read_data[itower] = (need_data[itower] && have_data[itower]);

	}

      // Stop the entire readout process once all data files are finished
      if (!(have_data[0] || have_data[1] || have_data[2]))
	{
	  break;
	}

      // Quit if there is a failure somewhere
      if (fSuccess == 0)
	break;

    } // main while (true) loop
    
  // Check and make sure that each tower was readout successfully
  if (fSuccess == 1)
    {
      for (itower=0; itower < 3; itower++)
	fSuccess *= (raw[itower]->fSuccess);
    }
    
}

void parsingManager::sort_tower_events()
{
  int itower, i, j;
  int sec1, sec2;
  int saved_index;
  rusdraw_dst_common *evt1, *evt2;
  for (itower=0; itower < 3; itower ++)
    {
      for (i=0; i<n_readout_events[itower]; i++)
	{
	  evt1 = obtain_event(itower, i);
	  sec1 = SDGEN::time_in_sec_j2000(evt1->yymmdd, evt1->hhmmss);
	  for (j=i+1; j<n_readout_events[itower]; j++)
	    {
	      evt2 = obtain_event(itower, j);
	      sec2 = SDGEN::time_in_sec_j2000(evt2->yymmdd, evt2->hhmmss);
	      // Flip the order
	      if (sec2 < sec1)
		{
		  saved_index = readout_event_index[itower][j];
		  readout_event_index[itower][j]
		    = readout_event_index[itower][i];
		  readout_event_index[itower][i] = saved_index;
		  i--; // re-do the current i-for-loop iteration
		  break; // break out of the current j-for-loop
		}
	      // If events have same second, then check their micro-second
	      if (sec2==sec1)
		{
		  if (evt2->usec < evt1->usec)
		    {
		      saved_index = readout_event_index[itower][j];
		      readout_event_index[itower][j]
			= readout_event_index[itower][i];
		      readout_event_index[itower][i] = saved_index;
		      i--;
		      break;
		    }
		}
	    } // j-for-loop
	} // i-for-loop
    } // itower-loop
}

void parsingManager::sort_tmatched_events()
{
  int i, j;
  int sec1, sec2;
  int saved_index, saved_event_num;
  rusdraw_dst_common *evt1, *evt2;
  for (i=0; i<n_tmatched_events; i++)
    {
      evt1 = obtain_event(i);
      sec1 = SDGEN::time_in_sec_j2000(evt1->yymmdd, evt1->hhmmss);
      for (j=i+1; j<n_tmatched_events; j++)
	{
	  evt2 = obtain_event(j);
	  sec2 = SDGEN::time_in_sec_j2000(evt2->yymmdd, evt2->hhmmss);
	  if (sec2 < sec1)
	    {
	      saved_index = tmatched_event_index[j];
	      tmatched_event_index[j] = tmatched_event_index[i];
	      tmatched_event_index[i] = saved_index;
	      saved_event_num = evt2->event_num;
	      evt2->event_num = evt1->event_num;
	      evt1->event_num = saved_event_num;
	      i--;
	      break;
	    }
	  if (sec2==sec1)
	    {
	      if (evt2->usec < evt1->usec)
		{
		  saved_index = tmatched_event_index[j];
		  tmatched_event_index[j] = tmatched_event_index[i];
		  tmatched_event_index[i] = saved_index;
		  saved_event_num = evt2->event_num;
		  evt2->event_num = evt1->event_num;
		  evt1->event_num = saved_event_num;
		  i--;
		  break;
		}
	    }
	} // j-for-loop
    } // i-for-loop
}

bool parsingManager::save_event(int itower)
{
  int iievent;
  iievent=readout_event_index[itower][n_readout_events[itower]];
  n_readout_events[itower]++;
  raw[itower]->get_curEvent(&readout_events[itower][iievent]);
  if (n_readout_events[itower] < NRUSDRAW_TOWER)
    return true;
  return false;
}
// save current event for calibrating and writing out
bool parsingManager::save_event()
{
  int iievent;
  if (n_tmatched_events >= NRUSDRAW_TMATCHED)
    {
      printErr("Fatal Error: Buffer for time matched events is full");
      return false;
    }
  iievent=tmatched_event_index[n_tmatched_events];
  memcpy(&tmatched_events[iievent], &curEvent, sizeof(rusdraw_dst_common));
  n_tmatched_events ++;
  curevt_num++; // inrement the overal event counter
  return true;
}
void parsingManager::remove_event(int itower, int ievent)
{
  int i, saved_index;
  saved_index=readout_event_index[itower][ievent];
  n_readout_events[itower]--;
  for (i=ievent; i<n_readout_events[itower]; i++)
    {
      readout_event_index[itower][i]=readout_event_index[itower][i+1];
    }
  // Add the saved event buffer index at the end of the index array so that
  // it can be used later.
  readout_event_index[itower][n_readout_events[itower]]=saved_index;
}
void parsingManager::remove_event(int ievent)
{
  int i, saved_index;
  saved_index=tmatched_event_index[ievent];
  n_tmatched_events--;
  for (i=ievent; i<n_tmatched_events; i++)
    tmatched_event_index[i]=tmatched_event_index[i+1];
  tmatched_event_index[n_tmatched_events] = saved_index;
}
rusdraw_dst_common* parsingManager::obtain_event(int itower, int ievent)
{
  int iievent;
  if ((ievent < 0) || (ievent >= n_readout_events[itower]))
    return 0;
  iievent=readout_event_index[itower][ievent];
  return &readout_events[itower][iievent];
}
rusdraw_dst_common* parsingManager::obtain_event(int ievent)
{
  int iievent;
  if ((ievent < 0) || (ievent >= n_tmatched_events))
    return 0;
  iievent=tmatched_event_index[ievent];
  return &tmatched_events[iievent];
}
bool parsingManager::save_mon(int itower)
{
  int iimon;
  iimon=readout_mon_index[itower][n_readout_mon[itower]];
  n_readout_mon[itower]++;
  raw[itower]->get_curMon(&readout_mon[itower][iimon]);
  if (n_readout_mon[itower] < NSDMON_TOWER)
    return true;
  return false;
}
bool parsingManager::save_mon()
{
  int iimon;
  if (n_tmatched_mon >= NSDMON_TMATCHED)
    {
      printErr("Fatal Error: Buffer for time matched monitoring cycles is full");
      return false;
    }
  iimon=tmatched_mon_index[n_tmatched_mon];
  memcpy(&tmatched_mon[iimon], &curMon, sizeof(sdmon_dst_common));
  n_tmatched_mon ++;
  curmon_num ++; // increase the overall monitoring cycle counter
  return true;
}
void parsingManager::remove_mon(int itower, int imon)
{
  int i, saved_index;
  saved_index=readout_mon_index[itower][imon];
  n_readout_mon[itower]--;
  for (i=imon; i<n_readout_mon[itower]; i++)
    readout_mon_index[itower][i]=readout_mon_index[itower][i+1];
  readout_mon_index[itower][n_readout_mon[itower]] = saved_index;
}
void parsingManager::remove_mon(int imon)
{
  int i, saved_index;
  saved_index=tmatched_mon_index[imon];
  n_tmatched_mon--;
  for (i=imon; i<n_tmatched_mon; i++)
    tmatched_mon_index[i]=tmatched_mon_index[i+1];
  tmatched_mon_index[n_tmatched_mon] = saved_index;
}
sdmon_dst_common* parsingManager::obtain_mon(int itower, int imon)
{
  int iimon;
  if ((imon < 0) || (imon >= n_readout_mon[itower]))
    return 0;
  iimon=readout_mon_index[itower][imon];
  return &readout_mon[itower][iimon];
}
sdmon_dst_common* parsingManager::obtain_mon(int imon)
{
  int iimon;
  if ((imon < 0) || (imon >= n_tmatched_mon))
    return 0;
  iimon=tmatched_mon_index[imon];
  return &tmatched_mon[iimon];
}

void parsingManager::calibrate_event(rusdraw_dst_common *event,
				     sdmon_dst_common *mon)
{
  int iwf, ind;
  for (iwf=0; iwf<event->nofwf; iwf++)
    {
      ind=sdi->getInd(event->xxyy[iwf]);
      if (ind<0)
	{
	  printErr(
		   "Can't get calibration for XXYY=%04d,site=%d,date=%06d time=%06d.%06d",
		   event->xxyy[iwf], event->site, event->yymmdd, event->hhmmss,
		   event->usec);
	  continue;
	}
      event->monyymmdd=mon->yymmddb;
      event->monhhmmss=mon->hhmmssb;
      memcpy(event->pchmip[iwf], mon->pchmip[ind], 2*sizeof(integer4));
      memcpy(event->pchped[iwf], mon->pchped[ind], 2*sizeof(integer4));
      memcpy(event->lhpchmip[iwf], mon->lhpchmip[ind], 2*sizeof(integer4));
      memcpy(event->lhpchped[iwf], mon->lhpchped[ind], 2*sizeof(integer4));
      memcpy(event->rhpchmip[iwf], mon->rhpchmip[ind], 2*sizeof(integer4));
      memcpy(event->rhpchped[iwf], mon->rhpchped[ind], 2*sizeof(integer4));
      memcpy(event->mftndof[iwf], mon->mftndof[ind], 2*sizeof(integer4));
      memcpy(event->mip[iwf], mon->mip[ind], 2*sizeof(real8));
      memcpy(event->mftchi2[iwf], mon->mftchi2[ind], 2*sizeof(real8));
      memcpy(event->mftp[iwf], mon->mftp[ind], 2*4*sizeof(real8));
      memcpy(event->mftpe[iwf], mon->mftpe[ind], 2*4*sizeof(real8));
    }
}
bool parsingManager::calibrate_events()
{

  int ievent, evtsec;
  int imon, monsec1, monsec2, imon_closest, mindt;
  rusdraw_dst_common *event;
  sdmon_dst_common *mon;
  bool bad_calib = false;

  // Initialize the calibration flags
  for (ievent=0; ievent<n_tmatched_events; ievent++)
    is_event_calibrated[ievent] = false;

  // Go over each time matched monitoring cycle and look for events which occured in their time ranges
  for (imon=0; imon<n_tmatched_mon; imon++)
    {
      if ((mon = obtain_mon(imon))== 0)
	{
	  printErr("Internal inconsistency in calibrating events: couldn't get mon. cycle");
	  return false;
	}
      // Don't use monitoring cycles which had readout issues
      if (mon->errcode != 0)
	continue;

      // Seconds at the beginning and the end of the monitoring cycle
      monsec1=SDGEN::time_in_sec_j2000(mon->yymmddb, mon->hhmmssb); // second at the beginning of the mon. cycle
      monsec2=SDGEN::time_in_sec_j2000(mon->yymmdde, mon->hhmmsse); // second at the end of the mon. cycle

      // Go over the time matched events and see which ones occured in the time range of this monitoring cycle
      for (ievent=0; ievent < n_tmatched_events; ievent++)
	{
	  if ((event=obtain_event(ievent))==0)
	    {
	      printErr("Internal inconsistency in calibrating events: couldn't get event");
	      return false;
	    }

	  // Don't calibrate events which were seen by a site that's not present
	  // in a given monitoring cycle
	  if(!tower_id_com(event->site,mon->site))
	    break;
	  
	  evtsec=SDGEN::time_in_sec_j2000(event->yymmdd, event->hhmmss);

	  // Stop when events are not in time window of the current monitoring cycle
	  if (evtsec >= monsec2)
	    break;

	  // Calibrate the events using current monitoring cycle
	  if (evtsec >= monsec1)
	    {
	      calibrate_event(event, mon); // calibrated the event
	      is_event_calibrated[ievent] = true; // set the flag saying that the event is calibrated
	    }
	}
    }

  // Loop over events which didn't calibrate ( occured in a corrupted mon. cyle or at the end of the day when
  // and mon. cycles have already been written out)
  for (ievent=0; ievent<n_tmatched_events; ievent++)
    {
      // If event was successfully calibrated then break out of the loop
      if (is_event_calibrated[ievent])
	continue;

      if ((event=obtain_event(ievent))==0)
	{
	  printErr("Internal inconsistency in calibrating events: couldn't get event");
	  return false;
	}
      evtsec=SDGEN::time_in_sec_j2000(event->yymmdd, event->hhmmss);

      // Find a suitable monitoring cycle (closest in time to the event)
      imon_closest=-1;
      mindt=86400;
      for (imon=0; imon<n_tmatched_mon; imon++)
	{
	  if ((mon = obtain_mon(imon))== 0)
	    {
	      printErr("Internal inconsistency in calibrating events: couldn't get mon. cycle");
	      return false;
	    }
	  // Don't use the corrupted monitoring cycle
	  if (mon->errcode != 0)
	    continue;
	  // Don't calibrate events which were seen by a site that's not present
	  // in a given monitoring cycle
	  if(!tower_id_com(event->site,mon->site))
	    continue;
	  monsec1=SDGEN::time_in_sec_j2000(mon->yymmddb, mon->hhmmssb);
	  monsec2=SDGEN::time_in_sec_j2000(mon->yymmdde, mon->hhmmsse);

	  // Determine which monitoring cycle is closest in time to the event under consideration
	  if (abs(monsec1-evtsec) < mindt || abs(monsec2-evtsec) < mindt)
	    {
	      mindt=abs(monsec1-evtsec);
	      imon_closest = imon;
	    }
	}

      // Have a problem if don't have a good monitoring cycle anywhere to calibrate the event, report it
      if ((imon_closest == -1) && (!have_good_mon))
	{
	  printErr("could not find a good mon cycle for event date=%06d time=%06d.%06d",
		   event->yymmdd, event->hhmmss, event->usec);
	  imon_closest=-1;
	  mindt=86400;
	  for (imon=0; imon<n_tmatched_mon; imon++)
	    {
	      if ((mon = obtain_mon(imon))== 0)
		{
		  printErr("Internal inconsistency in calibrating events: couldn't get mon. cycle");
		  return false;
		}
	      // Don't calibrate events which were seen by a site that's not present
	      // in a given monitoring cycle
	      if(!tower_id_com(event->site,mon->site))
		continue;
	      monsec1=SDGEN::time_in_sec_j2000(mon->yymmddb, mon->hhmmssb);
	      monsec2=SDGEN::time_in_sec_j2000(mon->yymmdde, mon->hhmmsse);
	      
	      // Determine which monitoring cycle is closest in time to the event under consideration
	      if (abs(monsec1-evtsec) < mindt || abs(monsec2-evtsec) < mindt)
		{
		  mindt=abs(monsec1-evtsec);
		  imon_closest = imon;
		}
	    }
	  if(imon_closest == -1)
	    {
	      printErr("Fatal Error: can't find any mon cycle for event site=%d date=%06d time=%06d.%06d",
		       event->site, event->yymmdd, event->hhmmss, event->usec);
	      if(opt.verbose >= 1)
		{
		  fprintf(stderr,"Currently available mon cycles are:\n");
		  for (imon=0; imon<n_tmatched_mon; imon++)
		    {
		      if ((mon = obtain_mon(imon))== 0)
			{
			  printErr("Internal inconsistency in calibrating events: couldn't get mon. cycle");
			  return false;
			}
		      fprintf(stderr, "mon: %d errcode=%d site=%d FROM %06d %06d TO %06d %06d\n",
			      mon->event_num,mon->errcode,mon->site,mon->yymmddb,mon->hhmmssb,
			      mon->yymmdde,mon->hhmmsse);
		    }
		}
	      return false;
	    }
	  bad_calib = true; // using a closest monitoring cycle with non-zero error code
	}
      
      // If didn't find a good mon. cycle in the time matched buffer but have a good mon. cycle in the special buffer,
      // then use that good monitoring cycle
      if ((imon_closest == -1) && have_good_mon)
	{
	  calibrate_event(event, &last_good_mon);
	  is_event_calibrated[ievent] = true;
	  continue;
	}

      // Check which one is better to use: last good monitoring cycle in the special buffer or 
      // the closest in time monitoring cycles out of good monitoring cycles found in current buffer
      if (have_good_mon)
	{
	  monsec1=SDGEN::time_in_sec_j2000(last_good_mon.yymmddb,
					   last_good_mon.hhmmssb);
	  monsec2=SDGEN::time_in_sec_j2000(last_good_mon.yymmdde,
					   last_good_mon.hhmmsse);
	  if (abs(monsec1-evtsec) < mindt || abs(monsec2-evtsec) < mindt)
	    {
	      calibrate_event(event, &last_good_mon);
	      is_event_calibrated[ievent] = true;

	      continue;
	    }
	}

      // If the last good monitoring cycle is either absent or is not the best choice ( not closer in time)
      // than the closest mon. cycle in the buffer, then use closest mon. cycle in the buffer
      if ((mon = obtain_mon(imon_closest))== 0)
	{
	  printErr("Internal inconsistency in calibrating events: couldn't get mon. cycle");
	  return false;
	}
      // if forced to use a bad monitoring cycle, then increase the event error code by a large number
      if(bad_calib)
	{
	  event->errcode += 10 * mon->errcode; // increase the event error code by a lot
	  n_bad_calib ++;   // increase the number of badly calibrated events
	}
      calibrate_event(event, mon);
      is_event_calibrated[ievent] = true;
    }

  // Check and make sure that all events have been calibrated
  for (ievent=0; ievent<n_tmatched_events; ievent++)
    {
      if (!is_event_calibrated[ievent])
	{
	  printErr("Internal inconsistency in event calibration, event should be calibrated but it is not");
	  return false;
	}
    }
  return true;
}

bool parsingManager::remove_duplicated_events()
{
  int ievent, jevent, ievent_remove;
  rusdraw_dst_common *event1, *event2;
  double jt1, jt2;
  for (ievent=0; ievent< n_tmatched_events; ievent++)
    {
      if (!(event1=obtain_event(ievent)))
	{
	  printErr("Internal error in removing duplicated events: failed to get event1");
	  return false;
	}
      jt1 = (double) SDGEN::time_in_sec_j2000(event1->yymmdd,event1->hhmmss);
      jt1 += ((double)event1->usec)/1e6;
      for (jevent=0; jevent < n_tmatched_events; jevent++)
	{
	  if(jevent == ievent)
	    continue;
	  if (!(event2=obtain_event(jevent)))
	    {
	      printErr("Internal error in removing duplicated events; failed to get event2");
	      return false;
	    }
	  jt2  = (double)SDGEN::time_in_sec_j2000(event2->yymmdd,event2->hhmmss);
	  jt2 += ((double)event2->usec)/1e6;
	  if(fabs(jt2-jt1) < ((double)(opt.dup_usec))/1e6)
	    {
	      // decide which event to remove
	      ievent_remove = -1; // initialize
	      // if one of the events has a smaller error code value, then
	      // remove that event
	      if (event1->errcode < event2->errcode)
		ievent_remove = jevent;
	      if(event2->errcode < event1->errcode)
		ievent_remove = ievent;
	      // if both events have the same error flag value
	      // then save the one that has more waveforms recorded
	      if(ievent_remove == -1)
		{
		  if(event1->nofwf > event2->nofwf)
		    ievent_remove = jevent;
		  if(event2->nofwf > event1->nofwf)
		    ievent_remove = ievent;
		}
	      // if both events have the same error flag value
	      // and the same number of waveforms, then remove the later event
	      if(ievent_remove == -1)
		{
		  if(jt1 < jt2)
		    ievent_remove = jevent;
		  if(jt2 < jt1)
		    ievent_remove = ievent;
		}
	      // if the events are also recorded
	      // on the same exact time, then remove
	      // the one that was recorded later
	      if (ievent_remove == -1)
		{
		  if(ievent < jevent)
		    ievent_remove = jevent;
		  if(jevent < ievent)
		    ievent_remove = ievent;
		}
	      // if still undecided on which event to remove, there's an
	      // internal error
	      if (ievent_remove == -1)
		{
		  printErr("remove_duplicated_events: failed to decide which out of 2 duplicated event to remove");
		  return false;
		}
	      rusdraw_dst_common *event = obtain_event(ievent_remove);
	      if (!event)
		{
		  printErr("Internal error in removing duplicated events; failed to get event labeled for removal");
		  return false;
		}
	      fprintf(stderr, "removing duplicated event: %d errcode=%d site=%d date=%06d time=%06d.%06d nofwf=%d secsm=%d\n",
		      event->event_num,event->errcode,event->site,event->yymmdd,event->hhmmss,event->usec,
		      event->nofwf,SDGEN::time_in_sec_j2000(event->yymmdd,event->hhmmss)-min_moncycle_second);
	      remove_event(ievent_remove); // removing the event from the list of time matched events
	      n_duplicated_removed ++;
	      // so that the ievent for loop starts from the beginning:
	      // when this goes to the next ievent for loop iteration, ievent
	      // will be increased by 1 which starts the for loop at ievent=0.
	      ievent = -1;
	      // break out from the jevent for-loop.
	      break;
	    }
	}
    }
  // also remove any events which repeat those that have been already written out
  rusdraw_dst_common *event = 0;
  double jt = 0;
  vector<double>::iterator jtw; // to iterate over written out event times
  for (ievent=0; ievent< n_tmatched_events; ievent++)
    {
      if (!(event=obtain_event(ievent)))
	{
	  printErr("Internal error in removing duplicated events: failed to get event while checking against written times");
	  return false;
	}
      jt = (double) SDGEN::time_in_sec_j2000(event->yymmdd,event->hhmmss);
      jt += ((double)event->usec)/1e6;
      for (jtw = written_event_times.begin(); jtw !=  written_event_times.end(); jtw++)
	{
	  if(fabs((*jtw)-jt) < ((double)(DUP_USEC))/1e6)
	    {
	      fprintf(stderr, "removing duplicated event: %d errcode=%d site=%d date=%06d time=%06d.%06d nofwf=%d secsm=%d ",
		      event->event_num,event->errcode,event->site,event->yymmdd,event->hhmmss,event->usec,
		      event->nofwf,SDGEN::time_in_sec_j2000(event->yymmdd,event->hhmmss)-min_moncycle_second);
	      fprintf(stderr, "(already written)\n");
	      // removing the event from the list of time matched events
	      // because it has already been written out
	      remove_event(ievent);
	      n_duplicated_removed ++;
	      ievent--; // needs to be re-iterated after the event removal
	      break;
	    }
	}
    }
  return true;
}

bool parsingManager::write_out_events()
{
  int ievent;
  rusdraw_dst_common *event;
  double jt = 0; // full time (second since Jan 1, 2000) of written out events
  for (ievent=0; ievent< n_tmatched_events; ievent++)
    {
      if ((event=obtain_event(ievent))==0)
	{
	  printErr("Internal error in writing out events");
	  return false;
	}
      if(opt.wevt)
	{
	  if (!p0io->writeEvent(event))
	    return false;
	  if(opt.verbose>=2)
	    {
	      fprintf(stdout, "event: %d errcode=%d site=%d date=%06d time=%06d.%06d nofwf=%d secsm=%d\n",
		      event->event_num,event->errcode,event->site,event->yymmdd,event->hhmmss,event->usec,
		      event->nofwf,SDGEN::time_in_sec_j2000(event->yymmdd,event->hhmmss)-min_moncycle_second);
	      fflush(stdout);
	    }
	}
      jt = (double)SDGEN::time_in_sec_j2000(event->yymmdd,event->hhmmss);
      jt += ((double)event->usec)/1e6;
      written_event_times.push_back(jt); // save the time of the written events
    }
  n_tmatched_events = 0;
  return true;
}

bool parsingManager::write_out_mon()
{
  int imon;
  int last_good_imon; // to remember the latest good monitoring cycle
  sdmon_dst_common *mon;
  last_good_imon = -1;
  for (imon=0; imon < n_tmatched_mon; imon++)
    {
      if ((mon = obtain_mon(imon))==0)
	{
	  printErr("Internal error in writing out mon. cycles");
	  return false;
	}
      if(opt.wmon)
	{
	  if (!p0io->writeMon(mon))
	    return false;
	  if(opt.verbose>=2)
	    {
	      fprintf(stdout, "mon: %d errcode=%d site=%d FROM %06d %06d TO %06d %06d\n",
		      mon->event_num,mon->errcode,mon->site,mon->yymmddb,mon->hhmmssb,
		      mon->yymmdde,mon->hhmmsse);
	      fflush(stdout);
	    }
	}
      if ((mon->errcode == 0) && (mon->site == SDMON_BRLRSK))
	last_good_imon = imon;
    }
  // If there is a latest good monitoring cycle, then save it
  if (last_good_imon >= 0)
    {
      if ((mon = obtain_mon(last_good_imon))==0)
	{
	  printErr("Internal error in getting mon. cycle");
	  return false;
	}
      memcpy(&last_good_mon, mon, sizeof(sdmon_dst_common));
      have_good_mon = true;
    }
  n_tmatched_mon = 0; // Clear out the buffer for the time matched monitoring cycles
  return true;
}

bool parsingManager::process_data()
{
  int itower;
  int jtower;
  bool break_flag;
  // true if mon. cycles for a given tower are absent from the beginning of processing
  bool is_absent[3];

  bool have_moncycle;
  int monsec[3] = 
    {0, 0, 0}; // store current monitoring cycle second
  sdmon_dst_common *mon[3] =
    { 0, 0, 0 };

  int ievent, jevent;
  int evtsec[3]; // store current event second
  int evtsecmi[3]; // minimum event second in the buffer for which can do time matching for given tower
  int evtsecmi_all; // minumum event second in the buffer for which will do time matching
  int evtsecma[3]; // maximum event second for which can do time matching for a given tower
  int evtsecma_all; // maximum event second in the buffer for which will do time matchng
  rusdraw_dst_common *evt[3] =
    { 0, 0, 0 };

  // Check for towers that are not present either because detectors were off
  // or because all monitoring cycles were written out
  for (itower=0; itower<3; itower++)
    {
      is_absent[itower] = false;
      if (n_readout_mon[itower] == 0)
	is_absent[itower] = true;
    }

  ///////////////////// TIME MATCHING THE MONITORING CYCLES ////////////////////////////
  while (cur_moncycle_second < max_moncycle_second)
    {

      if ( (n_readout_mon[0]==0) && (n_readout_mon[1]==0)
	   && (n_readout_mon[2] == 0))
	break;

      // Get the information on the first monitoring cycles in the 
      // current buffers for towers that are present.
      for (itower=0; itower<3; itower++)
	{
	  if (n_readout_mon[itower]==0)
	    continue;
	  if ((mon[itower] = obtain_mon(itower, 0)) != 0)
	    {
	      monsec[itower]=SDGEN::time_in_sec_j2000(mon[itower]->yymmddb,
						      mon[itower]->hhmmssb);
	    }
	  else
	    {
	      printErr("Fatal Error: Internal inconsistency in time matching monitoring cycles");
	      return false;
	    }
	}

      // Check if need more data to proceed: if some tower is present but
      // all of its monitoring cycles have been time-matched, then wait
      // for more data before time-matching the monitoring cycles of 
      // the remaining towers.
      break_flag = false;
      for (itower=0; itower<3; itower++)
	{
	  if ((!is_absent[itower]) && (n_readout_mon[itower] == 0))
	    {
	      break_flag = true;
	      break;
	    }
	}
      if (break_flag)
	break;

      // Clean the combined monitoring cycle buffer
      raw[0]->cleanMonCycle(&curMon);

      // Combine all monitoring cycles which occured on the given second
      have_moncycle = false;
      for (itower=0; itower<3; itower++)
	{
	  if (is_absent[itower])
	    continue;
	  if (monsec[itower] == cur_moncycle_second)
	    {
	      if (!addMonCycle(&curMon, mon[itower]))
		return false;
	      remove_mon(itower, 0); // remove the 1st monitoring cycle in the buffer
	      have_moncycle = true;
	    }
	}

      if (have_moncycle)
	{
	  compMonCycle(&curMon);
	  if (!save_mon())
	    return false;
	  nmon_tmatched_total ++;
	}

      // Increment the monitoring cycle second
      cur_moncycle_second += 600;

    }

  /////////////////// TIME MATCHING EVENTS ///////////////////////////////////
  sort_tower_events(); // Make sure events are in time order, sometimes they are not

  // Determine smallest and largest second for which we should consider time matching
  for (itower=0; itower < 3; itower ++)
    {
      if (n_readout_events[itower] == 0)
	continue;
      evt[itower] = obtain_event(itower, 0);
      evtsecmi[itower] = SDGEN::time_in_sec_j2000(evt[itower]->yymmdd,
						  evt[itower]->hhmmss); // earliest event for given tower
      evt[itower] = obtain_event(itower, (n_readout_events[itower]-1));
      evtsecma[itower] = evtsec[itower] = 
	SDGEN::time_in_sec_j2000(evt[itower]->yymmdd, evt[itower]->hhmmss); // latest event for given tower

      // For safety, for largest second in time matching we use actual largest second minus 16 seconds, because
      // more events may occur in that time window. If, however, the data for given tower has ended, we don't expect
      // any more events and can safely use tower's actual largest second in time matching.
      if (have_data[itower])
	evtsecma[itower] -= 16;
    }

  // Adjust largest event second for time matching.  If the data stream hasn't yet ended, then 
  // subtract off 16 seconds from the latest second for time matching: may still get events in that 
  // time window in the next read. Find the earliest and latest seconds which will be used in time matching.

  evtsecmi_all = max_moncycle_second;
  evtsecma_all = max_moncycle_second-1;

  for (itower=0; itower<3; itower++)
    {
      if (n_readout_events[itower] == 0)
	continue;

      // Determine what shold be the latest second in time matching for all towers.
      // Find the smallest latest second over all towers ( give chance for more events to come in for some towers,
      // if necessary)
      if ((evtsecma_all > evtsecma[itower]) && have_data[itower])
	evtsecma_all = evtsecma[itower];

      // Determine what should be the earliest second for time matching for all towers.
      // Again, use the smallest earliest second over all towers, include earliest events which don't
      // have partners for time matching.
      if (evtsecmi_all > evtsecmi[itower])
	evtsecmi_all = evtsecmi[itower];
    }

  // Actual time-matching.

  if(opt.verbose>=2)
    {
      fprintf(stdout, "************ READY TO TIME MATCH *******************\n");
      fflush(stdout);
    }
  for (itower=0; itower < 3; itower++)
    {
      for (ievent=0; ievent<n_readout_events[itower]; ievent++)
	{
	  evt[itower] = obtain_event(itower, ievent);
	  evtsec[itower] = SDGEN::time_in_sec_j2000(evt[itower]->yymmdd,
						    evt[itower]->hhmmss);
	  if(opt.verbose>=2)
	    {
	      fprintf(stdout, "site %d date=%06d time=%06d.%06d %d %d\n",itower,
		      evt[itower]->yymmdd,evt[itower]->hhmmss,evt[itower]->usec,evt[itower]->nofwf,
		      evtsec[itower]-min_moncycle_second);
	      fflush(stdout);
	    }
	}
    }
  if(opt.verbose>=2)
    {
      fprintf(stdout, "evtsecmi_all = %d evtsecma_all = %d\n",evtsecmi_all-min_moncycle_second,
	      evtsecma_all-min_moncycle_second);
      fprintf(stdout, "nBR=%d nLR=%d nSK=%d  .... \n",n_readout_events[0],
	      n_readout_events[1],n_readout_events[2]);
      fprintf(stdout, "****************************************************\n");
      fflush(stdout);
    }
  for (itower=0; itower<3; itower++)
    {
      while (n_readout_events[itower] > 0)
	{
	  // If there are events but can't obtain the pointer, then we have a failure
	  if ((evt[itower] = obtain_event(itower, 0))==0)
	    {
	      printErr("Internal inconsistency in time matching events: couldn't get expected event from the buffer");
	      return false;
	    }
	  // Make sure that the event second is in correct bounds. 
	  evtsec[itower] = SDGEN::time_in_sec_j2000(evt[itower]->yymmdd,
						    evt[itower]->hhmmss);

	  // At this point, events should not have second smaller than the one determined above
	  if (evtsec[itower] < evtsecmi_all)
	    {
	      printErr("Internal inconsistency in time matching events: got event second larger than expected");
	      return false;
	    }

	  // If event second exceeds the maximum second allowed in time-matching, then don't time-match events from this tower
	  if (evtsec[itower] > evtsecma_all)
	    break;

	  raw[0]->cleanEvent(&curEvent); // clean up the combined event
	  // add 1st event from a given tower to the combined event buffer
	  if (!addEvent(&curEvent, evt[itower]))
	    curEvent.errcode ++;
	  remove_event(itower, 0); // take out the event from tower's buffer, it's no longer needed there


	  // Look for events from other towers and find those which time-match the earliest itower event.
	  for (jtower=0; jtower<3; jtower++)
	    {
	      // Don't do time-matching on the same tower, of course
	      if (jtower == itower)
		continue;
	      for (jevent=0; jevent<n_readout_events[jtower]; jevent++)
		{
		  if ((evt[jtower] = obtain_event(jtower, jevent))==0)
		    {
		      printErr("Internal inconsistency in time matching events");
		      return false;
		    }
		  evtsec[jtower] = SDGEN::time_in_sec_j2000(evt[jtower]->yymmdd,
							    evt[jtower]->hhmmss);
		  // Don't need to go through events list for this tower if 
		  // its events are occuring later than the event for which we are trying to find
		  // time-matching companions
		  if (evtsec[jtower] > evtsec[itower])
		    break;

		  // Same - second events.  Consider time-matching them to event under consideration.
		  if (evtsec[jtower] == evtsec[itower])
		    {
		      if (abs(evt[jtower]->usec - curEvent.usec)
			  < opt.tmatch_usec)
			{
			  if (curEvent.usec > evt[jtower]->usec)
			    curEvent.usec = evt[jtower]->usec;
			  if (!addEvent(&curEvent, evt[jtower]))
			    curEvent.errcode ++;
			  remove_event(jtower, jevent); // jevent of jtower is no longer needed in the buffer
			}
		    }
		} // jevent-for-loop
	    } // jtower-for-loop
	  // By now, have checked if other towers have any events that are time-matching with the earliest itower event and
	  // included them into combined event. The event is time-matched.  If there was no itower event to begin with, this 
	  // part of the loop would not be executed.

	  if (!save_event()) // Save the event into time matched buffer
	    return false;
	  nevents_tmatched_total ++;

	} // while(n_readout_events[itower] > 0) loop which goes over all candidates for time-matching for itower
    } // itower-for-loop

  // Making sure the time-matched events are sorted in time
  sort_tmatched_events();
  
  // Remove the duplicated trigger events, if any
  if(!remove_duplicated_events())
    return false;
  
  // Calibrate the events
  if (!calibrate_events())
    return false;
  
  //  write events into DST files
  if (!write_out_events())
    return false;
    
  // Write out the monitoring cycles
  if (!write_out_mon())
    return false;

  // Finish data processing.  Set the appropriate flags, etc

  for (itower=0; itower<3; itower++)
    {
      if (n_readout_events[itower] < NRUSDRAW_TOWER && n_readout_mon[itower]
	  <NSDMON_TOWER)
	need_data[itower] = true;
    }

  if ((!need_data[0]) && (!need_data[1]) && (!need_data[2]))
    {
      if ((n_readout_events[0] == NRUSDRAW_TOWER) && (n_readout_events[1]
						      == NRUSDRAW_TOWER) && (n_readout_events[1] == NRUSDRAW_TOWER))
	{
	  printErr("Fatal Error: Event buffers are full and events have not been written out");
	}
      if ((n_readout_mon[0] == NSDMON_TOWER) && (n_readout_mon[1]
						 == NSDMON_TOWER) && (n_readout_mon[1] == NSDMON_TOWER))
	{
	  printErr("Fatal Error: Monitoring buffers are full and mon. cycles have not been written out");
	}

      return false;
    }
  fflush(stdout);
  return true;
}

bool parsingManager::addEvent(rusdraw_dst_common *comevent,
			      rusdraw_dst_common *event)
{

  int iwfmi, iwfma, iwf, j, k, itower;

  itower=event->site;
  if (!chk_tower_id(itower))
    return false;

  if (!sdi->addTowerID(&comevent->site, itower))
    {
      printErr("addEvent: Cant't combine tower IDs!");
      return false;
    }

  if (comevent->event_num == -1)
    {
      comevent->event_num = curevt_num;
      comevent->event_code = 1;
      comevent->errcode = event->errcode;
      comevent->yymmdd = event->yymmdd;
      comevent->hhmmss = event->hhmmss;
      comevent->usec = event->usec;
      comevent->monyymmdd = event->monyymmdd;
      comevent->monhhmmss = event->monhhmmss;
      comevent->nofwf = 0;
    }
  else
    {
      comevent->errcode += event->errcode;
    }

  comevent->run_id[itower] = event->run_id[itower];
  comevent->trig_id[itower] = event->trig_id[itower];

  iwfmi=comevent->nofwf;
  comevent->nofwf += event->nofwf;
  iwfma=iwfmi+event->nofwf-1;
  if (iwfma >= RUSDRAWMWF)
    {
      printErr("addEvent: Too many waveforms: current %d maximum %d\n", iwfma
	       +1, RUSDRAWMWF);
      comevent->nofwf = RUSDRAWMWF;
      return false;
    }
  for (iwf=iwfmi; iwf<=iwfma; iwf++)
    {
      comevent->nretry[iwf] = event->nretry[iwf-iwfmi];
      comevent->wf_id[iwf] = event->wf_id[iwf-iwfmi];
      comevent->trig_code[iwf] = event->trig_code[iwf-iwfmi];
      comevent->xxyy[iwf] = event->xxyy[iwf-iwfmi];
      comevent->clkcnt[iwf] = event->clkcnt[iwf-iwfmi];
      comevent->mclkcnt[iwf] = event->mclkcnt[iwf-iwfmi];
      for (j=0; j<2; j++)
	{
	  comevent->fadcti[iwf][j] = event->fadcti[iwf-iwfmi][j];
	  comevent->fadcav[iwf][j] = event->fadcav[iwf-iwfmi][j];
	  for (k=0; k<rusdraw_nchan_sd; k++)
	    {
	      comevent->fadc[iwf][j][k] = event->fadc[iwf-iwfmi][j][k];
	    }
	  comevent->pchmip[iwf][j] = event->pchmip[iwf-iwfmi][j];
	  comevent->pchped[iwf][j] = event->pchped[iwf-iwfmi][j];
	  comevent->lhpchmip[iwf][j] = event->lhpchmip[iwf-iwfmi][j];
	  comevent->lhpchped[iwf][j] = event->lhpchped[iwf-iwfmi][j];
	  comevent->rhpchmip[iwf][j] = event->rhpchmip[iwf-iwfmi][j];
	  comevent->rhpchped[iwf][j] = event->rhpchped[iwf-iwfmi][j];
	  comevent->mftndof[iwf][j] = event->mftndof[iwf-iwfmi][j];
	  comevent->mip[iwf][j] = event->mip[iwf-iwfmi][j];
	  comevent->mftchi2[iwf][j] = event->mftchi2[iwf-iwfmi][j];
	  for (k=0; k<4; k++)
	    {
	      comevent->mftp[iwf][j][k] = event->mftp[iwf-iwfmi][j][k];
	      comevent->mftpe[iwf][j][k] = event->mftpe[iwf-iwfmi][j][k];
	    }
	}
    }

  return true;
}

bool parsingManager::addMonCycle(sdmon_dst_common *comcycle,
				 sdmon_dst_common *mon)
{
  int itower;
  // minimum and maximum value of SD index depending on what tower SDs we
  // are adding to the combined monitoring cycle
  int indmi, indma;
  int ind, j, k; // dummy


  itower=mon->site;
  if (!chk_tower_id(itower))
    return false;

  switch (itower)
    {

    case SDMON_BR:
      {
        indmi=0;
        indma=sdi->getNsds(SDMON_BR) - 1;
        break;
      }
    case SDMON_LR:
      {
        indmi=sdi->getNsds(SDMON_BR);
        indma=indmi+sdi->getNsds(SDMON_LR) - 1;
        break;
      }
    case SDMON_SK:
      {
        indmi=sdi->getNsds(SDMON_BR)+sdi->getNsds(SDMON_LR);
        indma=sdi->getNsds() - 1;
        break;
      }
    default:
      {
        printErr("addMonCycle: internal inconsistency");
        return false;
        break;
      }
    }

  if (!sdi->addTowerID(&comcycle->site, itower))
    {
      printErr("addMonCycle: Can't combine tower IDs");
      return false;
    }

  // If monitoring cycle is clean
  if (comcycle->event_num == SDMON_CL_VAL)
    {
      comcycle->event_num = curmon_num;
      comcycle->errcode = mon->errcode;
      comcycle->yymmddb = mon->yymmddb;
      comcycle->hhmmssb = mon->hhmmssb;
      comcycle->yymmdde = mon->yymmdde;
      comcycle->hhmmsse = mon->hhmmsse;
      comcycle->lind = sdi->getMaxInd();
    }
  else
    {
      comcycle->errcode += mon->errcode;
    }
  comcycle->nsds[itower] = mon->nsds[itower];
  comcycle->run_id[itower] = mon->run_id[itower];
  for (ind=indmi; ind<=indma; ind++)
    {
      comcycle->xxyy[ind] = mon->xxyy[ind-indmi]; /* detector location */
      for (j=0; j<2; j++)
	{
	  for (k=0; k<SDMON_NMONCHAN; k++)
	    {
	      comcycle->hmip[ind][j][k] = mon->hmip[ind-indmi][j][k];
	      if (k < (SDMON_NMONCHAN / 2))
		{
		  comcycle->hped[ind][j][k] = mon->hped[ind-indmi][j][k];
		}
	      if (k<(SDMON_NMONCHAN/4))
		{
		  comcycle->hpht[ind][j][k] = mon->hpht[ind-indmi][j][k];
		  comcycle->hpcg[ind][j][k] = mon->hpcg[ind-indmi][j][k];
		}
	    }
	  comcycle->pchmip[ind][j] = mon->pchmip[ind-indmi][j];
	  comcycle->pchped[ind][j] = mon->pchped[ind-indmi][j];
	  comcycle->lhpchmip[ind][j] = mon->lhpchmip[ind-indmi][j];
	  comcycle->lhpchped[ind][j] = mon->lhpchped[ind-indmi][j];
	  comcycle->rhpchmip[ind][j] = mon->rhpchmip[ind-indmi][j];
	  comcycle->rhpchped[ind][j] = mon->rhpchped[ind-indmi][j];
	}
      for (j=0; j<10; j++)
	{
	  comcycle->ccadcbvt[ind][j] = mon->ccadcbvt[ind-indmi][j];
	  comcycle->blankvl1[ind][j] = mon->blankvl1[ind-indmi][j];
	  comcycle->ccadcbct[ind][j] = mon->ccadcbct[ind-indmi][j];
	  comcycle->blankvl2[ind][j] = mon->blankvl2[ind-indmi][j];
	  comcycle->ccadcrvt[ind][j] = mon->ccadcrvt[ind-indmi][j];
	  comcycle->ccadcbtm[ind][j] = mon->ccadcbtm[ind-indmi][j];
	  comcycle->ccadcsvt[ind][j] = mon->ccadcsvt[ind-indmi][j];
	  comcycle->ccadctmp[ind][j] = mon->ccadctmp[ind-indmi][j];
	  comcycle->mbadcgnd[ind][j] = mon->mbadcgnd[ind-indmi][j];
	  comcycle->mbadcsdt[ind][j] = mon->mbadcsdt[ind-indmi][j];
	  comcycle->mbadc5vt[ind][j] = mon->mbadc5vt[ind-indmi][j];
	  comcycle->mbadcsdh[ind][j] = mon->mbadcsdh[ind-indmi][j];
	  comcycle->mbadc33v[ind][j] = mon->mbadc33v[ind-indmi][j];
	  comcycle->mbadcbdt[ind][j] = mon->mbadcbdt[ind-indmi][j];
	  comcycle->mbadc18v[ind][j] = mon->mbadc18v[ind-indmi][j];
	  comcycle->mbadc12v[ind][j] = mon->mbadc12v[ind-indmi][j];
	  comcycle->crminlv2[ind][j] = mon->crminlv2[ind-indmi][j];
	  comcycle->crminlv1[ind][j] = mon->crminlv1[ind-indmi][j];

	}
      comcycle->gpsyymmdd[ind] = mon->gpsyymmdd[ind-indmi];
      comcycle->gpshhmmss[ind] = mon->gpshhmmss[ind-indmi];
      comcycle->gpsflag[ind] = mon->gpsflag[ind-indmi];
      comcycle->curtgrate[ind] = mon->curtgrate[ind-indmi];
      comcycle->num_sat[ind] = mon->num_sat[ind-indmi];
      for (j=0; j<2; j++)
	{
	  comcycle->mftndof[ind][j] = mon->mftndof[ind-indmi][j];
	  comcycle->mip[ind][j] = mon->mip[ind-indmi][j];
	  comcycle->mftchi2[ind][j] = mon->mftchi2[ind-indmi][j];
	  for (k=0; k<4; k++)
	    {
	      comcycle->mftp[ind][j][k] = mon->mftp[ind-indmi][j][k];
	      comcycle->mftpe[ind][j][k] = mon->mftpe[ind-indmi][j][k];
	    }
	}
    }

  return true;
}

void parsingManager::printStats(FILE *fp)
{
  int itower;
  char tname[3][3]=
    { "BR", "LR", "SK" };

  // Print each tower's stats
  for (itower=0; itower<3; itower++)
    {
      // Print other parsing information from each tower
      raw[itower]->printStats(fp);

      // Print total number of events from each tower
      fprintf(fp, "DATE: %06d NEVENTS-%s: %d\n", opt.yymmdd, tname[itower],
	      nevents_tower_total[itower]);

      // Print total number of mon. cycles from each tower
      fprintf(fp, "DATE: %06d NMON-%s: %d\n", opt.yymmdd, tname[itower],
	      nmon_tower_total[itower]);
    }
  fprintf(fp, "\n");

  // Print date, total numbers of events and total number of monitoring cycles
  fprintf(fp, "DATE: %06d NEVENTS-ALL: %d NEVENTS-WRITTEN: %d NMON-ALL: %d\n", opt.yymmdd,
	  nevents_tmatched_total,nevents_tmatched_total-n_duplicated_removed, nmon_tmatched_total);
  
  // Print date and the number of duplicated trigger events removed, if any
  if(n_duplicated_removed)
    {
      fprintf(fp, "DATE: %06d DUPLICATED_EVENTS_REMOVED: %d\n", opt.yymmdd,
	      n_duplicated_removed);
    }
  
  // bad calibration case
  if(n_bad_calib)
    {
      fprintf(fp, "DATE: %06d N_BAD_CALIB: %d\n", opt.yymmdd,n_bad_calib);
      // if not enough good events due to bad calibration, then its not a successfull day
      if(nevents_tmatched_total-n_duplicated_removed - n_bad_calib < NMIN_GOOD_EVENTS)
	fSuccess = 0;
    }
  
  // Print the overall success flag
  fprintf(fp, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, fSuccess);

}
void parsingManager::printStats()
{
  printStats(stdout);
  fflush(stdout);
}

bool parsingManager::chk_tower_id(int itower)
{
  if ( (itower<0) || (itower>2))
    {
      printErr("%d is an invalid tower id value", itower);
      return false;
    }
  return true;
}

void parsingManager::printErr(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "parsingManager: %s\n", mess);
}

static real8 mipfun(real8 *x, real8 *par)
{
  // Fit parameters:
  // par[0]=Gauss Mean
  // par[1]=Gauss Sigma
  // par[2]=Linear Coefficient
  // par[3]=Scalling Factor (integral)
  return par[3]*(1+par[2]*(x[0]-par[0])) *(TMath::Gaus(x[0], par[0], par[1],
						       true));
}

void parsingManager::mipFIT(integer4 ind, sdmon_dst_common *mon)
{
  // par[0]=Gauss Mean
  // par[1]=Gauss Sigma
  // par[2]=Linear Coefficient
  // par[3]=Scalling Factor (integral)
  real8 fr[2];
  real8 sv[4], pllo[4], plhi[4];
  real8 delta;
  integer4 j, k;

  // go over upper and lower
  for (k=0; k<2; k++)
    {

      // Fit range
      delta = (real8)(mon->pchmip[ind][k]-mon->lhpchmip[ind][k]);
      fr[0] = (real8)mon->pchmip[ind][k]-delta;
      fr[1] = (real8)mon->pchmip[ind][k]+0.7*delta;

      // Starting values
      sv[0] =(real8)mon->pchmip[ind][k]; // estimate for mean of the gaussian
      sv[1] = delta; // estimate for sigma of the gaussian  
      sv[2] = 0.1; // The linear coeficient estimate
      sv[3] = hmip[k]->Integral(1, 598); // the last two channels contain garbage

      // Lower limits
      pllo[0] = (real8)mon->pchmip[ind][k]-2.0*delta;
      pllo[1] = 0.5*sv[1];
      pllo[2] = 2e-2;
      pllo[3] = 0.5*sv[3];

      // Upper Limits
      plhi[0] = (real8)mon->pchmip[ind][k]+2.0*delta;
      plhi[1] = 2.0*sv[1];
      plhi[2] = 10.0;
      plhi[3] = 1.5*sv[3];

      // Initialize the fit function
      ffit = new TF1("mipfun",mipfun,fr[0],fr[1],4);
      ffit->SetParameters(sv);
      for (j=0; j<4; j++)
	ffit->SetParLimits(j, pllo[j], plhi[j]);

      // Fit
      if((hmip[k]->Integral() < 1) || (hmip[k]->Fit(ffit, "RB0Q") == -1))
	{
	  if(opt.verbose >= 3)
	    {
	      printErr(
		       "mipFIT: xxyy=%04d layer=%d yymmdd=%06d hhmmss=%06d npts=%d",
		       sdi->getXXYYfromInd(ind), k, mon->yymmddb, mon->hhmmssb,
		       hmip[k]->Integral());
	    }
	  mon->mftchi2[ind][k] = 1.0e3;
	  mon->mftndof[ind][k] = 0;
	  for (j=0; j<4; j++)
	    {
	      mon->mftp[ind][k][j]  = 0.0;
	      mon->mftpe[ind][k][j] = 0.0;
	    }
	}
      else
	{
	  // Get chi2 and ndof
	  mon->mftchi2[ind][k] = ffit->GetChisquare();
	  mon->mftndof[ind][k] = ffit->GetNDF();
	  // Get fit parameters and their errors
	  for (j=0; j<4; j++)
	    {
	      mon->mftp[ind][k][j] = ffit->GetParameter(j);
	      mon->mftpe[ind][k][j] = ffit->GetParError(j);
	    }
	}
      // Discard the fit function
      delete ffit;
    }
}

// To calculate the peak values of the histograms and find their half-peak channels.
void parsingManager::compMonCycle(sdmon_dst_common *mon)
{
  int i, j, k, l;

  // First index is for upper/lower
  // 2nd index: 0 for the values, 1 for the channel numbers (in C-notation, just like the firmware)
  int pch_mip[2][2], pch_ped[2][2], lhpch_mip[2][2], lhpch_ped[2][2],
    rhpch_mip[2][2], rhpch_ped[2][2], jl_mip, jl_ped, ju_mip, ju_ped;

  unsigned char dflag[2]; // this flag is 255 if found all half-peak channels

  // Loop over the INDs
  for (i=0; i<=mon->lind; i++)
    {
      if (mon->xxyy[i]<=0)
	continue; // Don't bother with the counters which didn't checkin for some reasons

      // Initialize the peak channels and peak values and clean the 1mip histograms
      for (k=0; k<2; k++)
	{
	  for (l=0; l<2; l++)
	    {
	      pch_mip[k][l] = 0;
	      pch_ped[k][l] = 0;
	    }
	  hmip[k]->Reset(); // clean the 1MIP histograms used in fitting
	}

      // Find the peak channels for mip & ped.  Avoid using
      // the last channels as they may contain special error flags recorded by DAQ. 
      for (j=0; j<(SDMON_NMONCHAN-1); j++)
	{
	  // Loop over upper/lower
	  for (k=0; k<2; k++)
	    {
	      if (mon->hmip[i][k][j] >= 0)
		hmip[k]->SetBinContent(j+1, mon->hmip[i][k][j]);
	      if (mon->hmip[i][k][j] > pch_mip[k][0])
		{
		  pch_mip[k][0] = mon->hmip[i][k][j]; // the peak value
		  pch_mip[k][1] = j; // the peak channel
		}
	      // peak channel for ped, keeping in mind that the number of ped chanels is half as large as the number
	      // of channels for mip.
	      if ( (j < (SDMON_NMONCHAN /2 - 1)) && (mon->hped[i][k][j]
						     > pch_ped[k][0]))
		{
		  pch_ped[k][0] = mon->hped[i][k][j];
		  pch_ped[k][1] = j;
		}

	    }
	}

      // Record the peak channels to DST bank
      for (k=0; k<2; k++)
	{
	  mon->pchmip[i][k] = pch_mip[k][1];
	  mon->pchped[i][k] = pch_ped[k][1];
	}

      // Initialize the half-peak channels, first with the peak channels
      for (k=0; k<2; k++)
	{

	  // When mip and ped half-peak channels are found, then these flags will be set to 15.
	  dflag[k] = (unsigned char)0;

	  for (l=0; l<2; l++)
	    {
	      lhpch_mip[k][l] = pch_mip[k][l];
	      rhpch_mip[k][l] = pch_mip[k][l];
	      lhpch_ped[k][l] = pch_ped[k][l];
	      rhpch_ped[k][l] = pch_ped[k][l];
	    }
	}

      // Find the half-peak channels for mip and ped.
      for (j=0; j<(SDMON_NMONCHAN-1); j++)
	{

	  // No need to continue if found the half-peak channels for mip and ped.
	  if (dflag[0] == (unsigned char)15 && dflag[1] == (unsigned char)15)
	    break;

	  for (k=0; k<2; k++)
	    {
	      jl_mip = pch_mip[k][1]-j;
	      ju_mip = pch_mip[k][1]+j;
	      jl_ped = pch_ped[k][1]-j;
	      ju_ped = pch_ped[k][1]+j;

	      // left half-peak for mip
	      if ((lhpch_mip[k][0] > (pch_mip[k][0] / 2)) && (jl_mip >=0 )
		  && (mon->hmip[i][k][jl_mip] > 0))
		{
		  lhpch_mip[k][0] = mon->hmip[i][k][jl_mip];
		  lhpch_mip[k][1] = jl_mip;
		}
	      else
		{
		  // set the bit flag that says left-half peak for mip is found
		  dflag[k] |= (unsigned char)1;
		}

	      // right half-peak for mip
	      if ((rhpch_mip[k][0] > (pch_mip[k][0] / 2)) && (ju_mip
							      < (SDMON_NMONCHAN-1)) && (mon->hmip[i][k][ju_mip] > 0))
		{
		  rhpch_mip[k][0] = mon->hmip[i][k][ju_mip];
		  rhpch_mip[k][1] = ju_mip;
		}
	      else
		{
		  // indicated that the right-half peak for mip is found
		  dflag[k] |= (unsigned char)2;
		}

	      // left half-peak for ped
	      if ((lhpch_ped[k][0] > (pch_ped[k][0] / 2)) && (jl_ped >=0 )
		  && (mon->hped[i][k][jl_ped] > 0))
		{
		  lhpch_ped[k][0] = mon->hped[i][k][jl_ped];
		  lhpch_ped[k][1] = jl_ped;
		}
	      else
		{
		  dflag[k] |= (unsigned char)8;
		}

	      // right half-peak for ped
	      if ((rhpch_ped[k][0] > (pch_ped[k][0] / 2)) && (ju_ped
							      < (SDMON_NMONCHAN/2 - 1)) && (mon->hped[i][k][jl_ped] > 0))
		{
		  rhpch_ped[k][0] = mon->hped[i][k][ju_ped];
		  rhpch_ped[k][1] = ju_ped;
		}
	      else
		{
		  dflag[k] |= (unsigned char)16;
		}
	    }
	}

      // Record the half peak channels to DST bank, and fit the 1mip histograms      
      for (k=0; k<2; k++)
	{
	  mon->lhpchmip[i][k] = lhpch_mip[k][1];
	  mon->lhpchped[i][k] = lhpch_ped[k][1];
	  mon->rhpchmip[i][k] = rhpch_mip[k][1];
	  mon->rhpchped[i][k] = rhpch_ped[k][1];
	}

      // Fit 1MIP histograms for upper and lower
      mipFIT(i, mon);

      // Evaluate best estimate for 1MIP peak, subtract the pedestal
      for (k=0; k<2; k++)
	{
	  mon->mip[i][k]= mon->mftp[i][k][0]+ 0.5/mon->mftp[i][k][2]
	    * (sqrt(1.0+4.0 *mon->mftp[i][k][2]*mon->mftp[i][k][2]
                    * mon->mftp[i][k][1]*mon->mftp[i][k][1])-1.0)- 1.5
	    *(real8)mon->pchped[i][k];
	}

    } // for(i=0;i<=mon->lind ... 
}


bool parsingManager::tower_id_com(int towid_evt, int towid_mon ) 
{
  // check if event tower ID is compatible with montitoring cycle
  if(towid_mon == RUSDRAW_BRLRSK)
    return true;
  
  switch(towid_evt)
    {
      
    case RUSDRAW_BR:
      if(towid_mon==RUSDRAW_BR || towid_mon==RUSDRAW_BRLR || towid_mon == RUSDRAW_BRSK)
	return true;
      break;

    case RUSDRAW_LR:
      if(towid_mon==RUSDRAW_LR || towid_mon==RUSDRAW_BRLR || towid_mon == RUSDRAW_LRSK)
	return true;
      break;
      
    case RUSDRAW_SK:
      if(towid_mon==RUSDRAW_SK || towid_mon==RUSDRAW_BRSK || towid_mon == RUSDRAW_LRSK)
	return true;
      break;

    case RUSDRAW_BRLR:
      if(towid_mon==RUSDRAW_BRLR)
	return true;
      break;
      
    case RUSDRAW_BRSK:
      if(towid_mon==RUSDRAW_BRSK)
	return true;
      break;
      
    case RUSDRAW_LRSK:
      if(towid_mon==RUSDRAW_LRSK)
	return true;
      break;
      
    case RUSDRAW_BRLRSK:
      if(towid_mon==RUSDRAW_BRLRSK)
	return true;
      break;
    
    default:
      printErr("tower_id_com: event tower ID not acceptible: %d",towid_evt);
      break;
      
    }
  
  return false;
}
