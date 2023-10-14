#include "tower_parser.h"

#define TLINE        1 // PPS time header
#define eLINE        3 // Indicates where (in time) a given trigger occured
#define ELINE        4 // Event header
#define WLINE        5 // Waveforms header
#define wLINE        6 // A wavform header (for 1 waveform)
#define LLINE        7 // These lines contain the monitoring information
#define EENDLINE     8 // Event end line
// returns the time difference in minutes, used for finding
// whether a monitoring cycle is exactly 10 minutes long.
static double deltaTinMin(int yymmdd1, int hhmmss1, int yymmdd2, int hhmmss2)
{
  double res;
  res = (double)(SDGEN::timeAftMNinSec(hhmmss2) - SDGEN::timeAftMNinSec(hhmmss1)) / 60.0;
  // Assumes that the only possible exception is when the run starts before
  // the midnight and ends after the midnight.  In that case, we need to add
  // 24 hrs to the difference.
  if (yymmdd1!=yymmdd2)
    res += (double)24*60;
  return res;
}

static int floor_to_int_600(int hhmmss)
{
   // second since midnight rounded to smaller multiple of 600
  int sec_sm = 600 * (SDGEN::timeAftMNinSec(hhmmss)/600);
  int hh = sec_sm / 3600;
  int mm = (sec_sm - 3600 * hh) / 60;
  int ss = sec_sm - 3600 * hh - 60 * mm;
  // adjusted hhmmss so that the second since midnight
  // is rounded to the smaller multiple of 600
  return 10000*hh + 100*mm + ss;
}

towerParser::towerParser(const listOfOpt& passed_opt, int itower, rusdpass0io* rusdpass0io_pointer): opt(passed_opt)
{
  // Set the tower ID
  tower_id = itower;
  if ( (tower_id<0) || (tower_id>2))
    {
      fprintf(stderr, "towerParser: %d is an invalid tower id value\n",
	      tower_id);
      exit(2);
    }
  // Set the pointer to I/O handler
  p0io = rusdpass0io_pointer;
  if (p0io == 0)
    {
      fprintf(stderr, "towerParser: I/O class was not initialized\n");
      exit(2);
    }
  
  // Initialize the indexing class and give the the current date
  // so that it uses correct tower-detector layouts
  sdi = new sdindex_class(opt.yymmdd);
  maxIND = sdi->getMaxInd(tower_id); // maximum index for a current tower
  cleanEvent(&eventBuf); // clean the event buffer
  cleanMonCycle(&monBuf); // clean the monitoring buffer

  firstMonCycle = true; // Haven't called Parse() method yet, so initialize this variable
  firstTline = true;
  // Initialize the GPS timing structure and index
  for (igpstime=0; igpstime<MAXNGPST; igpstime++)
    {
      gpstbuf[igpstime].yymmdd = 0;
      gpstbuf[igpstime].hhmmss = 0;
      gpstbuf[igpstime].secnum = 0;
    }
  igpstime=0; // Reset the internal GPS time index
  nTlines = 0;
  nLlines = 0;
  nElines = 0;
  nEendLines = 0;
  nWlines = 0;
  nWlines = 0;
  nwLines = 0;
  yymmdd_cur = 0;
  hhmmss_cur = 0;
  yymmdd_exp = 0;
  hhmmss_exp = 0;
  secnum_pps_cur = 0;
  secnum_pps_last = -1;
  secnumlast = -1;
  secnumcur = 0;
  wfCount[0] = 0;
  wfCount[1] = 0;
  total_readout_problems = 0;
  event_readout_problems = 0;
  secnum_mismatches = 0;
  mon_readout_problems = 0;
  pps_1sec_problems = 0;
  on_time = 86400; // If there is time missing for the date, will subtract it off
  onEvent = false; // Ready to read a new event
  onMon = false; // Ready to read a new monitoring cycle

  time_recovery_mode = false;
  
  fSuccess = 1;
}

towerParser::~towerParser()
{

}

void towerParser::get_curEvent(rusdraw_dst_common *event)
{
  memcpy(event, &eventBuf, sizeof(rusdraw_dst_common));
}
int towerParser::get_curEventDate()
{
  return eventBuf.yymmdd;
}
void towerParser::get_curMon(sdmon_dst_common *mon)
{
  memcpy(mon, &monBuf, sizeof(sdmon_dst_common));
}
int towerParser::get_curMonDate()
{
  return monBuf.yymmddb;
}
int towerParser::Parse()
{
  char inBuf[ASCII_LINE_LEN]; // read buffer
  int iline; // to identify the lines

  int usec_evt; // micro-second for the event
  int secnum_evt; // GPS second number for the event
  int secnum_test; // for checking the second number at the beginning of the monitoring cycle

  int yr, mo, da, hr, mi, sec;
  int jte, jto; // expected and obtained seconds since midnight of jan 1, 2000
  int ijt, iyymmdd, ihhmmss, isecnum; // Dummy variables for making off-time corrections to trigger time buffer
  int ind; // IND corresponding to XXYY


  int xxyy_got_mon, xxyy_got_evt, det_x, det_y; // Counter position ID that's read being out
  int tgtblnum; // Number of trigger tables
  int mclkcnt; // Max. Clock Count monitoring
  int flagL[3]; // Some flags which we don't know yet / don't need for analysis
  int minnum; // Minute number
  int monCh[4]; // Channels of the monitoring histograms
  int ichan; // To loop over monCh array
  int wfNum; // Number of waveforms as read out
  int wnth; // waveform ID number as read out
  int wnretry; // number of retries in getting the waveform
  int wnline; // number of lines in wf readout as declared by the firmware, should be 132
  int wchkline; // count how many lines were actually put out in the ascii file
  int FADCchannel; // current fadc channel reported by the firmware
  int FADCi; // current fadc channel obtained by counting
  bool FADC_corrupted; // true if the fadc trace is corrupted.

  int hhmmss_adj; // Adjusted mon. cyce start time
  int delta_sec; // Number of seconds missed b/w adjusted mon. cycle time and actual mon. cycle time
  
  unsigned int repno           = 0; // repetition number (T-LINE)
  unsigned int triggerTimeHost = 0; // needed for E-lines (32 bit)
  unsigned int wfmBitf         = 0; // bit-wise operations specifically on waveform data
  int off_time                 = 0; // time for which the subarray(s) were not working

  // If about to read a new monitoring cycle, prepare it's time information
  if (!onMon)
    {
      cleanMonCycle(&monBuf);
      monBuf.errcode = 0;
      monBuf.site = tower_id;
      monBuf.run_id[tower_id] = p0io->GetReadFileRunID(tower_id);
      monBuf.nsds[tower_id] = maxIND+1; // number of SDs
      monBuf.yymmddb = yymmdd_cur;
      monBuf.hhmmssb = hhmmss_cur;
      secnum_test = SDGEN::timeAftMNinSec(hhmmss_cur) % 600;
      if (secnum_test != 0)
	{
	  fixMonCycleStart(hhmmss_cur, &hhmmss_adj, &delta_sec);
	  printErr(
		   "Monitoring cycle start time %06d mod 600 sec != 0; adjusted to %06d",
		   hhmmss_cur, hhmmss_adj);
	  monBuf.hhmmssb = hhmmss_adj;
	  monBuf.errcode += delta_sec;
	}
      onMon = true;
    }

  while (p0io->get_line(tower_id, inBuf))
    {

      iline = whatLine(inBuf); // identify the line


      /* Parsing the lines. The line recorgnizer ensures that the lines
         are valid by performing consistency checks.  No need to worry
         about that here. */
      switch (iline)
	{

	  /********** Parsing the L-lines (below) *******************/

	  // L 1525 15888b 0 118 0 037 2faeefd 1d 1e 1b 1a
	  // L xxyy = 1525 flag0=15888b flag1=0 secnum=118 flag2=0 tgtblnum=037 mclkcnt=2faeefd, channels=....

        case LLINE:
          {
            nLlines++;

            // Do not parse monitoring information while in time-reovery mode:
            // current monitoring cycle will be labeled as corrupted anyways
            if (time_recovery_mode)
              break;
            if (sscanf(inBuf, "L %d %x %x %d %x %d %x %x %x %x %x",
		       &xxyy_got_mon, &flagL[0], &flagL[1], &secnumcur, &flagL[2],
		       &tgtblnum, &mclkcnt, &monCh[0], &monCh[1], &monCh[2], &monCh[3])
                != 11)
              {
                printErr("wrong L-line pattern");
                if (yymmdd_cur == opt.yymmdd)
                  {
                    mon_readout_problems++;
                    total_readout_problems++;
                  }
                continue;
              }

            det_x = xxyy_got_mon / 100;
            det_y = xxyy_got_mon % 100;

            if ((det_x < 1) || (det_y < 1) || (det_x > SDMON_X_MAX) || (det_y
									>SDMON_Y_MAX))
              {
                printErr("L-line has absurd xxyy value: %d", xxyy_got_mon);
                if (yymmdd_cur==opt.yymmdd)
                  {
                    mon_readout_problems++;
                    total_readout_problems++;
                  }
                continue;
              }

            // If there is a problems at getting the information, we just continue.  The corresponding
            // monitoring information then will be equal to -1, this is how we know that there was
            // a problem with the DAQ.  Don't just label the whole monitoring cycle as bad.

            if ((secnumcur != secnum_pps_cur))
              {
                // a minor problem
                if (opt.verbose>=3)
                  {
                    printErr("secnum_mismatch: %04d expected %d actual %d",
			     xxyy_got_mon, secnum_pps_cur, secnumcur);
                  }
                if (yymmdd_cur==opt.yymmdd)
                  secnum_mismatches++; // just count how many secnum mismatches there were in the run.
                if ((secnumcur < 0) || (secnumcur> 599))
                  continue;
              }

            /////////////////////////////////////////////////////////////////////////////////
            /*=============================================================================*/
            /* PARSING THE MONITORING INFORMATION, LAST 4 DIGITS OF L-LINES (BELOW)        */
            /*=============================================================================*/
            /////////////////////////////////////////////////////////////////////////////////

            ind = sdi->getInd(tower_id, xxyy_got_mon);

            if ((ind < 0) || (ind> maxIND))
              {
		if(opt.verbose >=1)
		  printErr("XXYY = %04d not understood", xxyy_got_mon);
                if (yymmdd_cur == opt.yymmdd)
                  {
                    mon_readout_problems++;
                    total_readout_problems++;
                  }
                continue;
              }

            // Keep record of the largest valid index found in the monitoring cycle
            if (monBuf.lind < ind)
              monBuf.lind = ind;

            // Record the detector position
            monBuf.xxyy[ind] = xxyy_got_mon;

            if ((secnumcur >=0) && (secnumcur < 600))
              {
                // Number of trigger tables
                monBuf.tgtblnum[ind][secnumcur] = tgtblnum;

                // Maximum clock count monitoring
                monBuf.mclkcnt[ind][secnumcur] = mclkcnt;
              }

            // 1MIP histogram (UPPER)
            if ((secnumcur >= 0) && (secnumcur < 128))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hmip[ind][1][4 * secnumcur + ichan] = monCh[ichan];
                  }
              }

            // 1MIP histogram (LOWER)
            if ((secnumcur >= 128) && (secnumcur < 256))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hmip[ind][0][4 * (secnumcur - 128) + ichan] = monCh[ichan];
                  }
              }

            // Pedestals (UPPER)
            if ((secnumcur >= 256) && (secnumcur < 320))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hped[ind][1][4 * (secnumcur - 256) + ichan] = monCh[ichan];
                  }
              }

            // Pedestals (LOWER)
            if ((secnumcur >= 320) && (secnumcur < 384))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hped[ind][0][4 * (secnumcur - 320) + ichan] = monCh[ichan];
                  }
              }

            // Pulse height histogram (UPPER)
            if ((secnumcur >= 384) && (secnumcur < 416))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hpht[ind][1][4 * (secnumcur - 384) + ichan] = monCh[ichan];
                  }
              }

            // Pulse height histogram (LOWER)
            if ((secnumcur >= 416) && (secnumcur < 448))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hpht[ind][0][4 * (secnumcur - 416) + ichan] = monCh[ichan];
                  }
              }

            // Pulse charge histogram (UPPER)
            if ((secnumcur >= 448) && (secnumcur < 480))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hpcg[ind][1][4 * (secnumcur - 448) + ichan] = monCh[ichan];
                  }
              }

            // Pulse charge histogram (LOWER)
            if ((secnumcur >= 480) && (secnumcur < 512))
              {
                for (ichan = 0; ichan < 4; ichan++)
                  {
                    monBuf.hpcg[ind][0][4 * (secnumcur - 480) + ichan] = monCh[ichan];
                  }
              }

            // CC status variables, recorded every minute for 10 minutes, so there should be 10 minute numbers.
            if ((secnumcur >= 512) && (secnumcur < 532))
              {

                /*
		  Every secnum allows 4 status variables (last 4 digits of the L-lines).
		  This is what's recorded for secnum = 512-531 (last 4 digits of the L-line):
		  1) CCADCvalueBattVoltage(minnum),Blank(minnum),CCADCValueBattCurrent(minnum),Blank(minnum)
		  2) CCADCValueRefVoltage(minnum),CCADCValueBattTemp(minnum),CCADCValueSolarVoltage(minnum),CCADCValueCCTemp
		  One list of variables follows the other  untill these variables are recorded for
		  minnum = 0..9.  This all happens in secnum = 512..531.

		  First, we calculate what is the minnum.  For that, just take the integer part of
		  (secnum - 512) / 2 because there are two lists of variables one after the other. To figure out whether
		  the 4 digits are for the 1st or for the 2nd list of variables, take the remainder of
		  (secnum-512) / 2, if it's 0, then the L-line pertains to the 1st list of variables, if it's 1, then
		  the L-line pertains to the 2nd list of variables.

		*/
                minnum = (secnumcur - 512) / 2;
                // If it's 1st list of variables
                if (((secnumcur-512) % 2) == 0)
                  {
                    monBuf.ccadcbvt[ind][minnum] = monCh[0]; /* CC ADC value Batt Voltage */
                    monBuf.blankvl1[ind][minnum] = monCh[1]; /* 1st blank value in b/w, in case later it will have something */
                    monBuf.ccadcbct[ind][minnum] = monCh[2]; /* CC ADC Value Batt Current */
                    monBuf.blankvl2[ind][minnum] = monCh[3]; /* 2nd blank value in b/w, in case later it will have something */
                  }
                else
                  {
                    monBuf.ccadcrvt[ind][minnum] = monCh[0]; /* CC ADC Value Ref Voltage */
                    monBuf.ccadcbtm[ind][minnum] = monCh[1]; /* CC ADC Value Batt Temp */
                    monBuf.ccadcsvt[ind][minnum] = monCh[2]; /* CC ADC Value SolarVoltage */
                    monBuf.ccadctmp[ind][minnum] = monCh[3]; /* CC ADC Value CC Temp */
                  }

              }

            // MB status variables, recorded every minute for 10 minutes
            if ((secnumcur >= 532) && (secnumcur < 552))
              {

                /*
		  This is what's recorded for secnum = 532-551 (last 4 digits of the L-line):
		  1)MBADCvalueGND(minnum),MBADCvalueSDTemp(minnum),MBADCvalue5.0V(minnum),MBADCvalueSDHum(minnum)
		  2)MBADCvalue3.3V(minnum),MBADCvalueBDTemp(minnum),MBADCvalue1.8V(minnum),MBADCvalue1.2V(minnum)
		*/
                minnum = (secnumcur - 532) / 2;
                // If it's 1st list of variables
                if (((secnumcur-532) % 2) == 0)
                  {
                    monBuf.mbadcgnd[ind][minnum] = monCh[0]; /* Main board ADC value "GND" */
                    monBuf.mbadcsdt[ind][minnum] = monCh[1]; /* Main board ADC value SDTemp */
                    monBuf.mbadc5vt[ind][minnum] = monCh[2]; /* Main board ADC value 5.0V */
                    monBuf.mbadcsdh[ind][minnum] = monCh[3]; /* Main board ADC value SDHum */
                  }
                else
                  {
                    monBuf.mbadc33v[ind][minnum] = monCh[0]; /* Main board ADC value 3.3V */
                    monBuf.mbadcbdt[ind][minnum] = monCh[1]; /* Main board ADC value BDTemp */
                    monBuf.mbadc18v[ind][minnum] = monCh[2]; /* Main boad ADC value 1.8V */
                    monBuf.mbadc12v[ind][minnum] = monCh[3]; /* Main boad ADC value 1.2V */
                  }

              }

            // Rate Monitor
            if ((secnumcur >= 552) && (secnumcur < 557))
              {

                /*
		  This is what's recorded for secnum = 551-556 (last 4 digits of the L-line):
		  1minCrLv2gt3mip(minnum),1minCrLv2gt0.3mip(minnum),1minCrLv2gt3mip(minnum+1),
		  1minCrLv2gt0.3mip(minnum+1)
		*/

                // Minute number that corresponds to the 1st pair of status variables
                // The other pair of variables will be at (minnum+1).
                minnum = 2 * (secnumcur-552);

                monBuf.crminlv2[ind][minnum] = monCh[0]; /* 1min count rate Lv2(>3mip) */
                monBuf.crminlv1[ind][minnum] = monCh[1]; /* 1min count rate Lv1(>0.3mip) */
                monBuf.crminlv2[ind][minnum+1] = monCh[2]; /* 1min count rate Lv2(>3mip) */
                monBuf.crminlv1[ind][minnum+1] = monCh[3]; /* 1min count rate Lv1(>0.3mip) */

              }

            // GPS monitor
            if (secnumcur==557)
              {
                monBuf.gpsyymmdd[ind] = monCh[0]; /* Date(YMD) */
                monBuf.gpshhmmss[ind] = monCh[1]; /* Time(HMS) */
                monBuf.gpsflag[ind] = monCh[2]; /* GPSFLAG */
                monBuf.curtgrate[ind] = monCh[3]; /* CURRENT TRIGGER Rate */
              }
            if (secnumcur==558)
              {
                monBuf.num_sat[ind] = monCh[1]; /* number of satellites seen by the SD */
              }
            /////////////////////////////////////////////////////////////////////////////////
            /*=============================================================================*/
            /* PARSING THE MONITORING INFORMATION, LAST 4 DIGITS OF L-LINES (ABOVE)        */
            /*=============================================================================*/
            /////////////////////////////////////////////////////////////////////////////////


            secnumlast = secnumcur;

            break;
          } // case LLINE

          /********** Parsing the L-lines (above) *******************/

          // TIME HEADER
        case TLINE:
          {
            nTlines++;

            // Don't store last time if still in time recovering mode;
            // wait until know for sure that the timing has been recovered.
            if (!time_recovery_mode)
              {
                yymmdd_exp = yymmdd_cur;
                hhmmss_exp = hhmmss_cur;
              }
	    
	    // Time that one expects by increasing previous time by
	    // one second. yymmdd_exp, hhmmss_exp will now contain
	    // current yymmdd, hhmmss obtained from previously known
	    // yymmdd, hhmmss by increasing time by 1 second.
	    SDGEN::parseAABBCC(yymmdd_exp, &yr, &mo, &da);
	    yr += 2000;
	    SDGEN::parseAABBCC(hhmmss_exp, &hr, &mi, &sec);
	    SDGEN::change_second(&yr, &mo, &da, &hr, &mi, &sec, 1);
            yr -= 2000;
	    SDGEN::toAABBCC(yr, mo, da,  &yymmdd_exp);
	    SDGEN::toAABBCC(hr, mi, sec, &hhmmss_exp);
	    
	    // Get the current date and time
            sscanf(inBuf, "#T %8x %6d %6d", &repno, &yymmdd_cur, &hhmmss_cur);

	    // Compare the current date and time with what we expect
            if ((hhmmss_exp != hhmmss_cur) || (yymmdd_exp != yymmdd_cur))
              {
                // If it's not the 1st T-line in the readout
                if (nTlines > 1)
                  {
		    
		    // get the second since Jan 1, 2000 for the expected time
                    jte = SDGEN::time_in_sec_j2000(yymmdd_exp,hhmmss_exp);
		    
		    // get the second since Jan 1, 2000 for the observed time
		    jto = SDGEN::time_in_sec_j2000(yymmdd_cur,hhmmss_cur);
		    
                    // If only 1 PPS second is missing
                    if ((jte-jto) == 1)
                      {
                        // a minor problem
                        if (opt.verbose>=3)
                          {
                            printErr(
				     "GPS: expected %06d %06d actual %06d %06d; using expected",
				     yymmdd_exp, hhmmss_exp, yymmdd_cur, hhmmss_cur);
                          }
                        
			// in case of a problem, assign expected yymmdd, hhmmss to pps timing.
                        yymmdd_cur = yymmdd_exp;
                        hhmmss_cur = hhmmss_exp;
                        
			// if this is the date of interest, count the missing
			// pps problems
			if (yymmdd_cur == opt.yymmdd)
                            pps_1sec_problems++;
			
			// if we previously have been in the time recovery mode
			// then notify that the time is OK now
                        if (time_recovery_mode)
                          {
                            time_recovery_mode = false;
                            printErr(
				     "Notice: GPS recovered: expected %06d %06d actual %06d %06d",
				     yymmdd_exp, hhmmss_exp, yymmdd_cur, hhmmss_cur);
                            time_recovery_mode = false;
                          }
                      }
                    // Otherwise, we may have detector off-time here, which
                    // should be reported for the given date
                    else if ((jte-jto) < 0)
                      {
			// if we have been in the time recovery mode then
			// this is expected.
			if (time_recovery_mode)
                          {
                            time_recovery_mode = false;
                            printErr(
				     "Notice: GPS recovered: expected %06d %06d actual %06d %06d",
				     yymmdd_exp,hhmmss_exp,yymmdd_cur,hhmmss_cur);
                            time_recovery_mode = false;
                          }

			// put in the offending GPS as a 1st element into
			// the recovery structure
			tline_recovery[0].yymmdd = yymmdd_cur; 
			tline_recovery[0].hhmmss = hhmmss_cur;
			tline_recovery[0].calc_the_rest();
			n_tline_recovery = 1; // number of elements in the recovery structure now
			

			// If the expect time is less than the observed time,
			// then the situation may be dangerous: it's either the
			// detector off-time OR a wrong GPS time stamp has been
			// written. Must carefully check what's going to happen next
			// in the firmware output.
			
			// save the current position in the firmware output file
			p0io->save_current_pos(tower_id);

			// continue reading the firmware output file, looking for
			// T-lines.
			while(p0io->get_line(tower_id,inBuf))
			  {
			    // look only at the T-lines
			    if(whatLine(inBuf) != TLINE)
			      continue;
			    sscanf(inBuf, "#T %8x %6d %6d", &repno, 
				   &tline_recovery[n_tline_recovery].yymmdd, 
				   &tline_recovery[n_tline_recovery].hhmmss);
			    tline_recovery[n_tline_recovery].calc_the_rest();
			    n_tline_recovery++;
			    // stop reading the file if enough time
			    // recovery information has been gathered
			    if(n_tline_recovery == N_TLINE_RECOVERY)
			      break;
			  }
			
			// put the I/O immediately back into the
			// original position in the firmware output
			// file for continued parsing
			p0io->goto_saved_pos(tower_id);
			
			// examine the recovery stucture and determine if
			// its really the detector off-time or just an error
			// in the current GPS time stamp.
			
			// if the time follows the normal sequence afterwards, this
			// is the detector off-time
			if(tline_recovery[1].j2000sec-1==tline_recovery[0].j2000sec)
			  {
			    off_time = getOffTime(jte, jto);
			    on_time -= off_time;
			    monBuf.errcode += off_time;
	
			    // Also, adjust the roll-over buffer which
			    // keeps second numbers for assigning times to
			    // events. jto will be added to GPS trigger
			    // time buffer on the exist from the large
			    // correcting if-statement
			    for (ijt = jte; ijt < jto; ijt++)
			      {
				iyymmdd = SDGEN::j2000sec2yymmdd(ijt);
				ihhmmss = 10000*((ijt%86400)/3600)+ 100*((ijt%3600)/60)+ (ijt%60);
				isecnum = (ijt%600);
				addGpsTime(iyymmdd, ihhmmss, isecnum);
			      }
			  }
			// we have an error in the GPS time stamp and should correct it
			else
			  {
			    printErr("GPS: wrong time: expected %06d %06d actual %06d %06d; using expected",
				     yymmdd_exp, hhmmss_exp, yymmdd_cur, hhmmss_cur);
			    yymmdd_cur = yymmdd_exp;
			    hhmmss_cur = hhmmss_exp;
			    // count the pps problems
			    if (yymmdd_cur == opt.yymmdd)
			      pps_1sec_problems++;
			  }
                      }
                    // If current time is smaller than the previous time by more
                    // than 1 second, we have a serious problem and one should check
                    // the data.
                    else
                      {
                        if (!time_recovery_mode)
                          {
                            printErr(
				     "GPS problem: expected %06d %06d actual %06d %06d; attempt to recover",
				     yymmdd_exp, hhmmss_exp, yymmdd_cur, hhmmss_cur);
                            // If we are currently reading an event while this blackout then make sure to
                            // give this event a large error code, in case it ends up being reported.
                            eventBuf.errcode += 10;
                            monBuf.errcode += 10;
                            time_recovery_mode = true;
                          }
                        else
                          {
                            printErr(
				     "GPS problem persists: expected %06d %06d actual %06d %06d; attempt to recover",
				     yymmdd_exp, hhmmss_exp, yymmdd_cur, hhmmss_cur);
                          }
                      }
                  } // if (nTlines > 1 ...
              }
            else
              {
                // If time equals expected time then we say that it's recovered
                if (time_recovery_mode)
                  {
                    printErr(
			     "Notice: GPS recovered: expected %06d %06d actual %06d %06d",
			     yymmdd_exp, hhmmss_exp, yymmdd_cur, hhmmss_cur);
                    time_recovery_mode = false;
                  }
              }
            if (!time_recovery_mode)
              {
                secnum_pps_last = secnum_pps_cur;
                secnum_pps_cur = SDGEN::timeAftMNinSec(hhmmss_cur) % 600;
                addGpsTime(yymmdd_cur, hhmmss_cur, secnum_pps_cur);
              }
            // Don't finish the current monitoring cycle until the GPS timing is recovered.
            else
	      break;
	    
            // This is where the current monitoring cycle ends and a new one starts.
            if ((secnum_pps_cur - secnum_pps_last) < 0)

              {
                ///////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////
                ///// RETURN THE MONITORING CYCLE//////////////////////////
                ///////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////

                // If this is not the first monitoring cycle, everythihng is normal:
                // monitoring cycle should be exactly 10min long.
                if (!firstMonCycle && nTlines > 1)
                  {
                    // Add the latest T-line time for the end of the current monitoring cycle.
                    monBuf.yymmdde = yymmdd_cur;
                    monBuf.hhmmsse = hhmmss_cur;

                    if ((int)deltaTinMin(monBuf.yymmddb, monBuf.hhmmssb,
					 monBuf.yymmdde, monBuf.hhmmsse) != 10)
                      {
                        printErr("Monitoring cycle did not end 10 min after its start");
                        monBuf.errcode++;
                        if (yymmdd_cur==opt.yymmdd)
                          {
                            mon_readout_problems++;
                          }
                      }
		    // enforce that the monitoring cycle always starts
		    // at the multiple of 600 sec since midnight
		    monBuf.hhmmssb = floor_to_int_600(monBuf.hhmmssb);
                    onMon = false;
                    return READOUT_MON;
                  }
                // If this is the 1st call to Parse routine, then the 1st monitoting cycle is not good
                // (it will be some left overs from the previous monitoring cycle in the previous date
                // which we presumably have already aquired by running the pass0 program on earlier date)
                else
                  {
                    cleanMonCycle(&monBuf);
                    monBuf.site = tower_id;
                    monBuf.run_id[tower_id] = p0io->GetReadFileRunID(tower_id);
                    monBuf.nsds[tower_id] = maxIND+1; // number of SDs
                    monBuf.yymmddb = yymmdd_cur; // Date at the beginning of the new monitoring cycle
                    monBuf.hhmmssb = hhmmss_cur;
                    onMon = true; // Need to reading a new monitoring cycle
                    firstMonCycle = false; // Proceed as everything is not a first call to Parse routine
                  }
              }

            break;

          } // case TLINE


          // EVENT HEADER
        case ELINE:
          {
            cleanEvent();
            nElines++;
            sscanf(inBuf, "E %8x %8x", &eventBuf.trig_id[tower_id], &triggerTimeHost);
            eventBuf.site = tower_id;
            eventBuf.run_id[tower_id] = p0io->GetReadFileRunID(tower_id);
            usec_evt=(int)(triggerTimeHost&0xfffff); // 20 LSB are triggered microsecond
	    // top 12 MSB: the top-most 2 MSB represent the hybtrid
	    // trigger flags if it is a hybrid trigger then the ohter
	    // 10 bits represent the GPS second number (0-599 range),
	    // which is what we want
	    secnum_evt=(int)((triggerTimeHost>>20)&0x3ff);
            
	    // If event occured during time recovery mode, then make sure it's labeled as corrupted if
            // eventually it gets reported.
            if (time_recovery_mode)
              eventBuf.errcode += 10;
            if (!getGpsTime(secnum_evt, &eventBuf.yymmdd, &eventBuf.hhmmss))
              {
                // Print a message and increase event readout problems only if the date is relevant
                if (yymmdd_cur == opt.yymmdd)
                  {
                    printErr(
			     "trig_id = %d time was not found in trigger table, secnum = %d",
			     eventBuf.trig_id[tower_id], secnum_evt);
                    eventBuf.errcode += 10; // set the error code to a non-zero value
                    if (yymmdd_cur==opt.yymmdd)
		      event_readout_problems++;
                  }
                // If could not find the trigger time, then record the most recent time.
                eventBuf.yymmdd = yymmdd_cur;
                eventBuf.hhmmss = hhmmss_cur;
              }
            else
	      eventBuf.errcode = 0;
            eventBuf.usec = usec_evt;
            eventBuf.event_code = 1; // 1 for data, 0 for MC
            onEvent = true; // indication that we are in the event
            wfCount[1] = 0; // intialize the over-event waveform counter
            // This information will be filled in once a good monitoring cycle is found, which begins
            // right after the event was collected
            eventBuf.monyymmdd = -1;
            eventBuf.monhhmmss = -1;

            break;
          } // case ELINE


          /* WAVEFORM HEADER */
        case WLINE:
          {
            nWlines++;

            sscanf(inBuf, "W %4d %d", &xxyy_got_evt, &wfNum);
            if (wfCount[1] == RUSDRAWMWF)
              {
                printErr("Can't add XXYY=%04d counter: event has too many waveforms",xxyy_got_evt);
                eventBuf.errcode++; // increment the event error code
                if (yymmdd_cur==opt.yymmdd)
		  event_readout_problems++;
                break;
              }
            eventBuf.xxyy[wfCount[1]] = xxyy_got_evt;
            det_x = xxyy_got_evt / 100;
            det_y = xxyy_got_evt % 100;
            if (det_x> SDMON_X_MAX || det_y> SDMON_Y_MAX || det_x < 1 || det_y
                < 1)
              {
                printErr("Event has absurd xxyy value: %04d",
			 eventBuf.xxyy[wfCount[1]]);
                eventBuf.errcode ++;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
              }
            wfCount[0] = 0; /* initialize the waveform counter for
			       checking the consistency of W-lines */

            if (!onEvent)
              {
                printErr("W line came before the E-line");
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
              }

            break;

          } // case WLINE


          // WAVEFORM DATA
        case wLINE:
          {
            nwLines++;
            if (!onEvent)
              {
                printErr("w line came before the E-line");
                FADC_corrupted = true;
		if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }

            if (wfCount[1] == RUSDRAWMWF)
              {
                printErr("Can't add XXYY=%04d waveform information: event has too many waveforms",
			 xxyy_got_evt);
                eventBuf.errcode++; // increment the event error code
                if (yymmdd_cur==opt.yymmdd)
		  event_readout_problems++;
                break;
              }

            sscanf(inBuf, "w %d %d %d", &wnth, &wnretry, &wnline);

            // Record the waveform id in the trigger
            eventBuf.wf_id[wfCount[1]] = wnth;

            // Record the number of retries to get the waveform
            eventBuf.nretry[wfCount[1]] = wnretry;

            // Don't bother with a waveform that doesn't have 132 lines of information
            if (wnline != 132)
              break;

            /*
	      +-----------
	      | wave form
	      +-----------
	      Line 0    : time-stamp+triggerflg
	      MSB 3bit: trigger flg LSB 29bit time-stamp

	      Line 1    : CH0sum (MSB16bit) CH1sum(LSB16bit)

	      Line 2-129: MSB 8    bit fadc bin#
	      LSB12-23 bit CH0 fadc value
	      LSB 0-11 bit CH1 fadc value

	      Line 130  : Max clock cout of the detector
	      --> arount 50Mcouut
	      to get trigger time with
	      20nsec resolution, Please calculate
	      (LSB29bit of Line0)/(MAX clock)
	      Line 131  : Bank pointer value for the wave form

	    */

            /* Read the waveform information and the FADC trace */
            wchkline = 0;
	    FADC_corrupted = false;
            if (!p0io->get_line(tower_id, inBuf)) // obtain the next string
              {
                printErr("Corrupted waveform: XXYY=%04d could not read waveform clock count",
			 xxyy_got_evt);
		FADC_corrupted = true;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }
            if ((!isxdigit(inBuf[0])) || (!isxdigit(inBuf[7])))
              {
                printErr("Corrupted waveform: XXYY=%04d waveform clock count is not a hex number",
			 xxyy_got_evt);
		FADC_corrupted = true;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }

            wchkline++;

            /*
	      42f37e5d
	      Line 0    : time-stamp+triggerflg
	      MSB 3bit: trigger flg, LSB 29bit time-stamp
	    */

            sscanf(inBuf, "%8x", &wfmBitf);

            // Record the level 1 trigger code.
            eventBuf.trig_code[wfCount[1]] = (int)(wfmBitf>>29);
	    
            // Record the clock count at the beginning of the waveform,
            // waveform number is wfCount[1].
            eventBuf.clkcnt[wfCount[1]] = (int)(wfmBitf & 0x1fffffff);
	    
            // Line 1    : CH0sum (MSB16bit) CH1sum(LSB16bit)
	    
            if (!p0io->get_line(tower_id, inBuf))
              {
                printErr("Corrupted waveform: XXYY=%04d could not read FADC integral",
			 xxyy_got_evt);
		FADC_corrupted = true;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }
            if ((!isxdigit(inBuf[0])) || (!isxdigit(inBuf[7])))
              {
                printErr("Corrupted waveform: XXYY=%04d FADC integral is not a hex number",
			 xxyy_got_evt);
		FADC_corrupted = true;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }

            wchkline++;

            sscanf(inBuf, "%8x", &wfmBitf);

            // Record the FADC trace integrals, divided by 8, as read in from the raw data file
            eventBuf.fadcti[wfCount[1]][0] = (int) (wfmBitf >> 16); // 16 MSB
            eventBuf.fadcti[wfCount[1]][1] = (int) (wfmBitf & 0xffff); // 16 LSB


            /*
	      Reading FADC trace, 128 lines.
	      Line 2-129: MSB 8 bit fadc bin#
	      LSB12-23 bit CH0 fadc value
	      LSB 0-11 bit CH1 fadc value     */

            // Don't flip the logical order, or else the readLine
            // will get screwed up - first check how many FADC channels are read,
            // and then try to read more if needed.
            FADCi = 0;
            while ((FADCi < 128) && p0io->get_line(tower_id, inBuf))
              {
                if ((!isxdigit(inBuf[0])) || (!isxdigit(inBuf[7])))
                  {
                    printErr("Corrupted waveform: XXYY=%04d FADC channel information is not a hex number",
			     xxyy_got_evt);
		    FADC_corrupted = true;
                    if (yymmdd_cur==opt.yymmdd)
                      {
                        total_readout_problems++;
                        event_readout_problems++;
                      }
                    break;
                  }

                wchkline++;

                sscanf(inBuf, "%8x", &wfmBitf);
                
		// Readout channel of the FADC trace.
                FADCchannel = (int) (wfmBitf >> 24); // 8 MSB
                
		if (FADCchannel != FADCi)
                  {
		    // sometimes, the 0'th fadc slice gets dropped and
		    // some error code is written instead
		    if(FADCi == 0)
		      {
			if(opt.verbose >=1)
			  printErr("Corrupted waveform: XXYY=%04d FADC channel %d information missing",
				   xxyy_got_evt, FADCi);
			FADC_corrupted = true;
			// look ahead and check if the fadc trace becomes
			// normal afterwards
			p0io->save_current_pos(tower_id);
			p0io->get_line(tower_id,inBuf);
			sscanf(inBuf, "%8x",&wfmBitf);
			p0io->goto_saved_pos(tower_id);
			// if it doesn't become normal, print an error
			if ((!isxdigit(inBuf[0])) || (!isxdigit(inBuf[7])) || (((int)(wfmBitf>>24))!=1))
			  {
			    printErr("Corrupted waveform: XXYY=%04d FADC trace not recoverable, corrupted at the beginning",
				     xxyy_got_evt);
			    FADC_corrupted = true;
			    if (yymmdd_cur==opt.yymmdd)
			      {
				total_readout_problems++;
				event_readout_problems++;
			      }
			    break;
			  }
			// if the FADC trace becomes normal
			// afterwards, set the current (corrupted)
			// FADC time slices to zero (later, wfmBitf is
			// used in determining the FADC counts in each
			// layer and if it is zero, they will be set
			// to zero)
			FADCchannel = 0;
			wfmBitf     = 0;
		      }// if(FADCi==0 ...
		    
		    // if the corruption happened somewhere in the middle
		    // of the FADC trace and the actual number differs from the expected
		    // by one
		    else if((FADCi > 0) && (FADCchannel-FADCi == 1))
		      {
			if(opt.verbose >=1)
			  printErr("Corrupted waveform: XXYY=%04d FADC channels %d and %d are corrupted",
				   xxyy_got_evt,FADCi-1,FADCi);
			FADC_corrupted = true;
			eventBuf.fadc[wfCount[1]][0][FADCi-1] = 0;
			eventBuf.fadc[wfCount[1]][1][FADCi-1] = 0;
			eventBuf.fadc[wfCount[1]][0][FADCi]   = 0;
			eventBuf.fadc[wfCount[1]][1][FADCi]   = 0;
			FADCi++;
		      }
		    
		    // if the corruption happened in the middle and
		    // FADCchannel is different from FADCi by more
		    // than one, it's likely that the current line was
		    // corrupted.
		    else if ((FADCi>0) && (FADCchannel-FADCi != 1))
		      {
			// attempt to recover if this happens before the end of the FADC trace
			if(FADCi < 127)
			  {
			    // see what the next line says
			    p0io->save_current_pos(tower_id);
			    p0io->get_line(tower_id,inBuf);
			    p0io->goto_saved_pos(tower_id);
			    if ((!isxdigit(inBuf[0])) || (!isxdigit(inBuf[7])))
			      {
				printErr("Corrupted waveform: XXYY=%04d FADC mismatch not recoverable",
					 xxyy_got_evt);
				FADC_corrupted = true;
				if (yymmdd_cur==opt.yymmdd)
				  {
				    total_readout_problems++;
				    event_readout_problems++;
				  }
				break;
			      }
			    if(opt.verbose >= 1)
			      printErr("Corrupted waveform: XXYY=%04d FADC channel %d corrupted",
				       xxyy_got_evt,FADCi);
			    FADC_corrupted = true;
			    // this zeroes out the current waveform information
			    FADCchannel = FADCi;
			    wfmBitf = 0;
			  }
		      }
		    
		    // if there were any mismatches, increase the error values
		    eventBuf.errcode++;
		    if (yymmdd_cur==opt.yymmdd)
		      {
			total_readout_problems++;
			event_readout_problems++;
		      }
		    
                  } // if(FADCchannel != FADCi ...
		
                // This records the FADC trace for each waveform for upper and lower.

                // FADC counts are integers maximum up to
                // 0xfff, so it's safe to convert them to
                // usual integers
                // LSB12-23 (Lower counter FADC counts) :
                eventBuf.fadc[wfCount[1]][0][FADCi] = (int) ((wfmBitf >> 12) & 0xfff);
                // LSB0-11  (Upper  counter FADC counts) :
                eventBuf.fadc[wfCount[1]][1][FADCi] = (int) (wfmBitf & 0xfff);
		
                FADCi++;
		
              }
	    
	    // if the FADC readout wasn't 128 lines long (corruption, etc) then
	    // need to increment the number of problems and make sure that the counter
	    // max. clock count values stay reasonable (even though the waveform
	    // will not be used in the analysis, it will be less confusing if the
	    // max. clock count of the counter is set to some specific value)
            if (FADCi != 128)
              {
		// To make sure that we are not reporting the same
		// error twice: report FADC trace error only when we
		// were not able to finish the readout due to lack of
		// lines ( end of file, etc)
		if(!FADC_corrupted)
		  {
		    printErr("Corrupted waveform: XXYY=%04d FADC trace is not 128 lines long",
			     xxyy_got_evt);
		    FADC_corrupted = true;
		  }
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
		eventBuf.mclkcnt[wfCount[1]] = (int)floor(5e7+0.5);
                eventBuf.errcode ++;
                break;
              }

            /* Line 130  : Max clock cout of the detector
	       --> arount 50Mconut
	       to get trigger time with
	       20nsec resolution, Please calculate
	       (LSB29bit of Line0)/(MAX clock) */

            if (!p0io->get_line(tower_id, inBuf))
              {
                printErr("Corrupted waveform: XXYY=%04d could not read MaxClockCount",
			 xxyy_got_evt);
		eventBuf.mclkcnt[wfCount[1]] = (int)floor(5e7+0.5);
                eventBuf.errcode++;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }
            wchkline++;

            if ((!isxdigit(inBuf[0])) || (!isxdigit(inBuf[7])))
              {
                printErr("Corrupted waveform: %04d MaxClockCount is not a hex number",
			 xxyy_got_evt);
                eventBuf.errcode++;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
		eventBuf.mclkcnt[wfCount[1]] = (int)floor(5e7+0.5);
                break;
              }

            // converting to int is fine, the number isn't too large,
            // max clock count is usually around 50MHz, which is good.
            sscanf(inBuf, "%8x", &eventBuf.mclkcnt[wfCount[1]]);

            /* Line 131  : Bank pointer value for the wave form (16 MSB),
	       CH0 and CH0 FADC averages 16 LSB (first 8 bits of 16 LSB is
	       for CH0 fadc average and 2nd 8 bits of 16 LSB is for CH1)
	    */

            if (!p0io->get_line(tower_id, inBuf))
              {
                printErr("Corrupted waveform: XXYY=%04d could not read fadc average and bank pointer",
			 xxyy_got_evt);
                eventBuf.errcode++;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }
            wchkline++;

            if (!isxdigit(inBuf[0]) || !isxdigit(inBuf[7]))
              {
                printErr(
			 "Corrupted waveform: XXYY=%04d fadc average and bank pointer values are not hex numbers",
			 xxyy_got_evt);
                eventBuf.errcode++;
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }
            sscanf(inBuf, "%8x", &wfmBitf);

            // waveform bank pointer value
            //      eventBuf.wf_bankp[[wfCount[1]]]    = (int)(wfmBitf>>16);

            // mean of fadc trace for lower
            eventBuf.fadcav[wfCount[1]][0] = (int)((wfmBitf>>8)&0xff);

            // mean of fadc trace for upper
            eventBuf.fadcav[wfCount[1]][1] = (int)(wfmBitf&0xff);

            wfCount[0]++; // Count wf since the last W-line
            wfCount[1]++; // count the wf since the last E-line

            /* Keep the previous xxyy, of the
	       detector in case there is another waveform in this counter.
	       If it's not the case, then this xxyy value will get
	       overwritten by a new W-line or not recorded
	       at all if we are at the end of the event */
            if (wfCount[1]<RUSDRAWMWF)
              {
                eventBuf.xxyy[wfCount[1]] = eventBuf.xxyy[wfCount[1]-1];
              }
            break;

          } // case wLINE


          // EVENT END LINE
        case EENDLINE:
          {
            nEendLines++;

            if (!onEvent)
              {
                printErr("Reached event end line but didn't see the event start line");
                if (yymmdd_cur==opt.yymmdd)
                  {
                    total_readout_problems++;
                    event_readout_problems++;
                  }
                break;
              }

            // record the number of the waveforms for this event
            eventBuf.nofwf = wfCount[1];

            /////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////
            ///////////////// RETURN THE EVENT ///////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////
            wfCount[1] = 0; // reset the waveform counter
            onEvent = false; // Finished parsing event
            return READOUT_EVENT; // Return and say that an event has been read out
            break;

          } // case EENDLINE


          // EVERYTHING ELSE
        default:
          {
            break;
          }

	} // switch (iline)

    } // while(p0io->get_line(tower_id,inBuf).... )


  // If the time never recovered itself until the end of the readout
  if (time_recovery_mode)
    {
      printErr("readout ended and GPS time did not recover");
      fSuccess = 0;
      if (yymmdd_exp == opt.yymmdd)
	{
	  jte=SDGEN::time_in_sec_j2000(yymmdd_exp, hhmmss_exp);
	  jto=SDGEN::time_in_sec_j2000(yymmdd_exp, 0)+86400;
	  off_time = getOffTime(jte, jto);
	  on_time -= off_time;
	}
    }

  // Data files are finished.  Return 0.
  return READOUT_ENDED;
}

void towerParser::addGpsTime(int yymmdd, int hhmmss, int secnum)
{
  // When the number of GPS times exceeds MAXNGPST, overwrite
  // the numbers from the beginning
  gpstbuf[igpstime % MAXNGPST].yymmdd = yymmdd;
  gpstbuf[igpstime % MAXNGPST].hhmmss = hhmmss;
  gpstbuf[igpstime % MAXNGPST].secnum = secnum;
  igpstime++; // keep incrementing the timing information index
}
bool towerParser::getGpsTime(int secnum, int *yymmdd, int *hhmmss)
{
  int isec;
  for (isec=0; isec<MAXNGPST; isec++)
    {
      if (secnum == gpstbuf[isec].secnum)
	{
	  (*yymmdd) = gpstbuf[isec].yymmdd;
	  (*hhmmss) = gpstbuf[isec].hhmmss;
	  return true;
	}
    }
  (*yymmdd) = 0;
  (*hhmmss) = 0;
  return false;
}

void towerParser::cleanEvent(rusdraw_dst_common *event)
{
  int i;
  memset(event, 0, sizeof(rusdraw_dst_common));
  event->event_num = -1;
  event->errcode = 0;
  event->site = -1;
  for (i=0; i<3; i++)
    {
      event->run_id [i] = -1;
      event->trig_id[i] = -1;
    }
}
void towerParser::cleanEvent()
{
  cleanEvent(&eventBuf);
}
void towerParser::cleanMonCycle(sdmon_dst_common *mon)
{
  int i, j, k;
  mon->event_num = SDMON_CL_VAL;
  mon->site=SDMON_CL_VAL;
  for (i=0; i<3; i++)
    {
      mon->run_id [i] = SDMON_CL_VAL;
      mon->nsds [i] = SDMON_CL_VAL;
    }
  mon->errcode = 0;
  mon->yymmddb = SDMON_CL_VAL;
  mon->hhmmssb = SDMON_CL_VAL;
  mon->yymmdde = SDMON_CL_VAL;
  mon->hhmmsse = SDMON_CL_VAL;
  mon->lind = 0;
  // If a detector that's in the middle of the array is
  // not working, then we have to have zeros for its monitoring information.
  for (i=0; i<SDMON_MAXSDS; i++)
    {
      mon->xxyy[i] = SDMON_CL_VAL;
      for (k=0; k<2; k++)
	{
	  for (j=0; j<SDMON_NMONCHAN; j++)
	    {
	      mon->hmip[i][k][j] = SDMON_CL_VAL;
	      if (j<(SDMON_NMONCHAN/2))
		{
		  mon->hped[i][k][j] = SDMON_CL_VAL;
		}
	      if (j<(SDMON_NMONCHAN/4))
		{
		  mon->hpht[i][k][j] = SDMON_CL_VAL;
		  mon->hpcg[i][k][j] = SDMON_CL_VAL;
		}
	    }
	  mon->pchmip[i][k] = SDMON_CL_VAL;
	  mon->pchped[i][k] = SDMON_CL_VAL;
	  mon->lhpchmip[i][k] = SDMON_CL_VAL;
	  mon->lhpchped[i][k] = SDMON_CL_VAL;
	  mon->rhpchmip[i][k] = SDMON_CL_VAL;
	  mon->rhpchped[i][k] = SDMON_CL_VAL;
	}
      for (j=0; j<600; j++)
	{
	  mon->tgtblnum[i][j] = SDMON_CL_VAL;
	  mon->mclkcnt [i][j] = SDMON_CL_VAL;
	  if (j<10)
	    {
	      // CC
	      mon->ccadcbvt[i][j] = SDMON_CL_VAL;
	      mon->blankvl1[i][j] = SDMON_CL_VAL;
	      mon->ccadcbct[i][j] = SDMON_CL_VAL;
	      mon->blankvl2[i][j] = SDMON_CL_VAL;
	      mon->ccadcrvt[i][j] = SDMON_CL_VAL;
	      mon->ccadcbtm[i][j] = SDMON_CL_VAL;
	      mon->ccadcsvt[i][j] = SDMON_CL_VAL;
	      mon->ccadctmp[i][j] = SDMON_CL_VAL;
	      // MB
	      mon->mbadcgnd[i][j] = SDMON_CL_VAL;
	      mon->mbadcsdt[i][j] = SDMON_CL_VAL;
	      mon->mbadc5vt[i][j] = SDMON_CL_VAL;
	      mon->mbadcsdh[i][j] = SDMON_CL_VAL;
	      mon->mbadc33v[i][j] = SDMON_CL_VAL;
	      mon->mbadcbdt[i][j] = SDMON_CL_VAL;
	      mon->mbadc18v[i][j] = SDMON_CL_VAL;
	      mon->mbadc12v[i][j] = SDMON_CL_VAL;
	      // RM
	      mon->crminlv2[i][j] = SDMON_CL_VAL;
	      mon->crminlv1[i][j] = SDMON_CL_VAL;
	    }
	}
      // GPS
      mon->gpsyymmdd[i] = SDMON_CL_VAL;
      mon->gpshhmmss[i] = SDMON_CL_VAL;
      mon->gpsflag[i] = SDMON_CL_VAL;
      mon->curtgrate[i] = SDMON_CL_VAL;
      mon->num_sat[i] = SDMON_CL_VAL;
    }
}

void towerParser::cleanMonCycle()
{
  cleanMonCycle(&monBuf);
}

int towerParser::whatLine(char *aline)
{

  if (strlen(aline) == 0)
    return 0;

  switch (aline[0])

    {
      // L 1525 15888b 0 118 0 037 2faeefd 1d 1e 1b 1a
    case 'L':
      {
        if (isdigit(aline[2]) && isdigit(aline[5]) && isxdigit(aline[7]))
          return LLINE;
        break;
      }

      // several useful possibilities when a line starts with the # sign
    case '#':
      {
        /* REPETITION HEADER, provides the absolute time ...
	   [REPETITION] [YYMMDD] [HHMMSS] [SUBSEC] [SECNUM]
	   #T 00005850 080313 051025 4340 599*/
        /* rep. number is in HEX, everything else is in decimal */
        if ((aline[1] == 'T') && isxdigit(aline[3]) && isdigit(aline[12])
            && isdigit(aline[19]))
          return TLINE;

        /* Trigger timing information:
         * #e [bank pointer] [TRIGTIME@HOST]
         */
        /* All digits are assumed to be in hex format */
        if ((aline[1] == 'e') && isxdigit(aline[3]) && isxdigit(aline[12]))
          return eLINE;

        /* There is one such label after every E-line.
	   Assuming that this means that the event ends here.
	   ### DONE
	   shows end of correction of wave form
	   associated with the trigger.
	   ### DONE 858303 17a9957b <-- Don't know these numbers */

        if (strncmp(aline, "### DONE", 8) == 0)
          return EENDLINE;

        break;
      }
      /*
	E [Event#] [TRIGTIME@HOST]
	"Event#" "event number"
	"TRIGTIME@HOST" trigger timing measured by host
      */

    case 'E':
      {
        if (isxdigit(aline[2]) && isxdigit(aline[9]) && isxdigit(aline[11])
            && isxdigit(aline[18]))
          return ELINE;
        break;
      }

      /* Waveform header:
	 [detector XY] [number of waveforms]
	 W 1608 1 */

    case 'W':
      {
        if (isdigit(aline[2]) && isdigit(aline[5]) && isdigit(aline[7]))
          return WLINE;
        break;
      }
      /*
	Indicates the presence of the waveform data (FADC traces).
	There will be 2 more lines of numeric information, we'll
	have to check them latter in the main loop.
	w [nth] [nretry] [nline]
	"nth"    N-th wave form for the trigger from this detector
	"nline"  number of line to out put (usually 132line)
	w 0 0 132
      */
    case 'w':
      {
        if ( // everything is decimal
	    isdigit(aline[2]) && isdigit(aline[4]) && isdigit(aline[6]))
          return wLINE;
        break;
      }
    default:
      return 0;

    }

  return 0; // If didn't match anything, still return 0

}

void towerParser::printStats(FILE *fp)
{
  char tname[3][3]=
    { "BR", "LR", "SK" };
  // Print total number of lines of each type for the given tower
  fprintf(fp, "\n\n******** READING STATS-%s: ********\n", tname[tower_id]);
  fprintf(fp, "#T - LINES-%s: %d\n", tname[tower_id], nTlines);
  fprintf(fp, "L - LINES-%s: %d\n", tname[tower_id], nLlines);
  fprintf(fp, "E - LINES-%s: %d\n", tname[tower_id], nElines);
  fprintf(fp, "### DONE - LINES-%s: %d\n", tname[tower_id], nEendLines);
  fprintf(fp, "W - LINES-%s: %d\n", tname[tower_id], nWlines);
  fprintf(fp, "w - LINES-%s: %d\n", tname[tower_id], nwLines);

  // Report the parsing problems, if any
  fprintf(fp, "\n\n******** PROBLEMS-%s: ********\n", tname[tower_id]);
  if (event_readout_problems> 0)
    {
      fprintf(fp, "DATE: %06d EVENT READOUT PROBLEMS-%s: %d\n", opt.yymmdd,
	      tname[tower_id], event_readout_problems);
    }
  if (mon_readout_problems> 0)
    {
      fprintf(fp, "DATE: %06d MONITORING CYCLE READOUT PROBLEMS-%s: %d\n",
	      opt.yymmdd, tname[tower_id], mon_readout_problems);
    }
  if (total_readout_problems> 0)
    {
      fprintf(fp, "DATE: %06d TOTAL READOUT PROBLEMS-%s: %d\n", opt.yymmdd,
	      tname[tower_id], total_readout_problems);
    }
  if (secnum_mismatches> 0)
    {
      fprintf(fp, "DATE: %06d SECNUM MISMATCHES-%s: %d\n", opt.yymmdd,
	      tname[tower_id], secnum_mismatches);
    }
  if (pps_1sec_problems > 0)
    {
      fprintf(fp, "DATE: %06d PPS 1 SEC CORRECTIONS-%s: %d\n", opt.yymmdd,
	      tname[tower_id], pps_1sec_problems);
    }
  fprintf(fp, "\n\n******** OTHER-%s: ********\n", tname[tower_id]);
  fprintf(fp, "DATE: %06d ONTIME-%s: %d\n", opt.yymmdd, tname[tower_id],
	  on_time);
  fprintf(fp, "DATE: %06d SUCCESS-%s: %d\n", opt.yymmdd, tname[tower_id],
	  fSuccess);
}
void towerParser::printStats()
{
  printStats(stdout);
  fflush(stdout);
}

int towerParser::getOffTime(int jte, int jto)
{
  int iday, idaymax;
  int jtlo, jtm, jtm1, jtm2;
  int yymmdd;
  int res, offtime;
  // If obtained time is less than expected then still return 0.  Such situation is better handled separately
  // in function which calls this function.
  // This assumes that the obtained time has also been corrected by 1 second if 1 second increase
  // didn't happen.
  if (jto <= jte)
    return 0;
  offtime=0; // Initialize variable for the offtime
  jtm1 = jte - (jte % 86400); // time of the midnight for the expected date in seconds
  jtm2 = jto - (jto % 86400); // time of the midnight for the obtained date in seconds
  idaymax = (jtm2 - jtm1) / 86400; // maximum day index
  for (iday=0; iday<=idaymax; iday++)
    {
      jtm = jtm1 + iday*86400; // time of the midnight of the date considered
      jtlo = (jtm < jte ? jte : jtm);
      if ((jto - jtm) <= 86400)
	res = jto-jtlo;
      else
	res=jtm+86400-jtlo;
      yymmdd = SDGEN::j2000sec2yymmdd(jtm);
      if (yymmdd == opt.yymmdd)
	offtime = res;
    }
  printErr("OFFTIME: %d sec", offtime);
  return offtime;
}

void towerParser::fixMonCycleStart(int hhmmss_act, int *hhmmss_fix,
				   int *deltaSec)
{
  int hh, mm, ss, secsm;
  (*deltaSec)=0;
  hh=hhmmss_act/10000;
  mm=(hhmmss_act%10000)/100;
  ss=hhmmss_act%100;
  secsm=3600*hh+60*mm+ss;
  (*deltaSec) += secsm;
  secsm /= 600;
  secsm *= 600;
  (*deltaSec) -= secsm;
  hh = secsm/3600;
  mm = (secsm-(3600*hh))/60;
  ss = secsm - 3600*hh - 60 * mm;
  (*hhmmss_fix) = 10000*hh+100*mm+ss;
}
void towerParser::printErr(const char *form, ...)
{
  char mess[0x400];
  char *fnamep;
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fnamep=strrchr((char *)p0io->GetReadFile(tower_id), '/');
  if (fnamep==0)
    {
      fnamep=(char *)p0io->GetReadFile(tower_id);
    }
  else
    {
      fnamep++;
    }
  fprintf(stderr, "%s:%d: SAVEDATE=%06d CURDATE=%06d:  %s\n",
	  fnamep, p0io->GetReadLine(tower_id),opt.yymmdd,yymmdd_cur,mess);
}
