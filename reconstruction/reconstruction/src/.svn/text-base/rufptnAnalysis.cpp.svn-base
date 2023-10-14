#include "rufptnAnalysis.h"
#include "sdparamborder.h"

#include "sdmc_bsd_bitf.h"

#include "tacoortrans.h"

// Class that handles CLF coordinates
#include "sdxyzclf_class.h"

// Class that handles SDs which are on the border
#include "sdaborder.h"

// used for obtaining SD X,Y,Z coordinates in [1200m] units with respect to CLF frame.
static sdxyzclf_class sdxyzclf;

static sdaborder_class sdb; // To handle SDs on the border

using namespace TMath;

inline integer4 tinsecmdn(integer4 hhmmss)
  {
    return (hhmmss/10000)*3600+((hhmmss%10000)/100)*60+(hhmmss%100);
  }

rufptnAnalysis::rufptnAnalysis(listOfOpt& passed_opt) : opt(passed_opt), fHasTriggered(false)
  {
    
    integer4 i;

    /* Initialize the pointers with null */
    for (i = 0; i < 3; i++)
      {
        gLaxisFit[i] = 0;
      };

    sdorigin_xy[0] = SD_ORIGIN_X_CLF;
    sdorigin_xy[1] = SD_ORIGIN_Y_CLF;

    // Instantiate the geometry fitter class
    p1geomfitter = new p1geomfitter_class;

    // 1 mu response shape, obtained from looking at many 1mu fadc traces.
    muresp_shape[0] = 0.09044301;
    muresp_shape[1] = 0.2011518;
    muresp_shape[2] = 0.2357062;
    muresp_shape[3] = 0.1958141;
    muresp_shape[4] = 0.1346454;
    muresp_shape[5] = 0.07871757;
    muresp_shape[6] = 0.03778135;
    muresp_shape[7] = 0.01320628;
    muresp_shape[8] = 0.003927172;
    muresp_shape[9] = 0.001456995;
    muresp_shape[10] = 0.0007651797;
    muresp_shape[11] = 0.0004421589;
    for (i=12; i<128; i++)
      muresp_shape[i] = 0.0;

    sNpart = new TSpectrum(); // # for de-convolving fadc traces

  }

rufptnAnalysis::~rufptnAnalysis()
  {
  }

// Processes all FADC traces in the event  (for all waveforms, parses upper and lower).
// Finds: signal start and stop channels, zero derivative and largest derivative channels,
// pedestals, signals (in fadc counts), timing of the hits.
// (leading edge time plus offset at the beginning of the waveform, then subtracts the
// earliest hit times)
void rufptnAnalysis::processFADC()
  {

    integer4 i, j, k, l, cn, /* number of channels in checking for signal,
     if we are not at the end of FADC trace,
     it's set to a constant, otherwise it will be a
     smaller number */
    ihit; /* stores index of a hit in rufptn */

    real8 nsigped[2], // to count how many pedestal channels were under the signal
        mped[2], // mean pedestals
        pedrms[2]; // pedestal rms's

    integer4 ntim[2], // needed for finding the rms on time
        tx[2], tx2[2], // to compute the rms on time
        parea[2], // to compute the pulse areas
        deriv[2], // to compute derivatives
        lderiv[2], // for finding the largest derivative
        lderivc[2]; /* record the FADC channel when the largest
     derivative occured since the start signal channel
     and before the place where derivative hits zero */

    integer4 zderivc[2]; /* channel since the first point of inflection
     after which derivative becomes negative */

    bool isSignal[2]; /* If this flag is set, the portion o FADC trace
     is treated as signal starting at channel where
     this flag was first was set to true.  Constantly
     check if signal is returning to the pedestal level,
     if it does, then set this flag back to false */

    real8 nosig_sum[2]; // dummy variable needed in verifying signal endings.

    real8 tearliest[2]; // to determine the earliest signal in the event.

    bool gotTime[2]; /* Hit time is the one that corresponds to the
     point of inflection of the FIRST signal. So if we've
     already measured the time, this flag is set to true
     and any other excursions from the pedestal levels
     latter than the first signal will not be used for
     determining the time. */

    bool continuedWf; /* To indicate whether a given WF is another hit or continuation
     of the previous waveform (in cases of multiple - fold triggers */

    // Time gap in FADC channels b/w current and previous waveforms in a multiple - wf. hit.
    //real8 ntgapch;

    integer4 start_i[2], /* Channels where the signals start and stop */
    stop_i[2];

    integer4 nfchan= rusdraw_nchan_sd; /* number of FADC channels */

    // number of pedestal channels since last signal was seen. useful in multi-fold waveforms.
    integer4 nped_since_last_signal[2] =
      { 0, 0 };

    char problemStr[0x40]; // (abbreviated) bad SD problem description string

    integer4 clkcnt;  // adjusted waveform clock count to account for the cases when the clock count resets itself
    integer4 mclkcnt; // maximum clock count, for convenience
    
    // k-for loops will always go over the lower (k=0) and upper (k=1) counters.
    for (k = 0; k < 2; k++)
      tearliest[k] = 1e10; /* put some large number for the earliest time.  Latter,
       it will be rewritten by the smallest hit  time in
       the event . */

    // Initialized the "goodness flag" for the hits
    for (i=0; i<RUFPTNMH; i++)
      rufptn_.isgood[i] = 0;

    /* initialize the number triggers for each counter. From now on, we will be refering to these triggers
     as "hits" and not the actual waveforms (there can be many waveforms in one hit)  */
    rufptn_.nhits = 0; // reset the number of hits
    ihit = -1; // will be increased, 1st hit corresponding to ihit = 0
    // Loop over all the waveforms
    for (i = 0; i < rusdraw_.nofwf; i++)
      {

        if ((rusdraw_.pchped[i][0] > (SDMON_NMONCHAN/2-8))
            || (rusdraw_.pchped[i][1]/8 < 1) || (rusdraw_.pchped[i][1]/8
            > (SDMON_NMONCHAN/2-8)))
          {
	    sprintf(problemStr,"PEDESTAL");
            printBadSDInfo(i,problemStr,opt.bad_sd_fp);
            continue;
          }

        /* Check if the current waveform is a continuation of the previous one.
         The requirement is that the detector of the current waveform is the same as
         detector of the previous waveform */
        continuedWf = false;
        if ( (i>1) && (rusdraw_.xxyy[i]==rusdraw_.xxyy[i-1]))
          continuedWf = true;

        if (continuedWf)
          {

            // If the current FADC window started with least NPEDPREV channels of pedestals since the
            // last signal in previous 128 FADC window, for both upper and lower layers, then count the waveform as independent.
            if (nped_since_last_signal[0] >= NPEDPREV
                && nped_since_last_signal[1] >= NPEDPREV)
              continuedWf = false;
          }

        if (!continuedWf)
          {
            // record a new hit (waveform from which we've obtained the time of the
            // series of n-fold waveforms, which are continuation of the 1st waveform.
            // needed to use the monitoring information
            rufptn_.nhits++;
            ihit++; // increment the hit index, first hit corresponding to zero.
            rufptn_.wfindex[ihit] = i;
            rufptn_.xxyy[ihit] = rusdraw_.xxyy[i];
            rufptn_.nfold[ihit] = 1; /* reset the foldedness counter. */
            rufptn_.isgood[ihit] = 1; /* mark as good ( so far) */
            //ntgapch = 0; /* initialize the time gap b/w current and previous wf in a multi-fold hit */
            for (k = 0; k < 2; k++)
              {
                // Initialize the quantities

                mped[k] = rusdraw_.pchped[i][k]/(real8) NPEDINTCH;
                pedrms[k] = (rusdraw_.rhpchped[i][k]-rusdraw_.lhpchped[i][k])
                    / (real8) NPEDINTCH / 2.33;
                if (pedrms[k]*pedrms[k]<1e-7)
                  pedrms[k]=1e-3; // to avoid division by zero
                nped_since_last_signal[k] = 0; // initialize the ped. channel since last signal counter.
                nsigped[k] = 0; // to count the pedestal channels under the signal
                // initialize the pulse area calculation
                parea[k] = 0;

                // initialize the timing information
                ntim[k] = 0;
                tx[k] = 0;
                tx2[k] = 0;
                deriv[k] = 0;
                lderiv[k] = 0;

                /* If the derivative never hits zero but we reach the end of the FADC trace,
                 then we just use the last channel. */
                zderivc[k] = nfchan - 1;
                lderivc[k] = 0;
                start_i[k] = 0;
                // If the signal never stops but we reach the end of the FADC trace, then
                // we just use nfchan for the signal stop channel.
                stop_i[k] = nfchan - 1;
                // Time needs to be computed.
                isSignal[k] = gotTime[k] = false;
              }

          }
        else
          {
            rufptn_.nfold[ihit]++; // count the foldedness of the hit

            if ((i - 1) < 0)
              {
                fprintf(stderr,
			"\nINCONSISTENCY: %s(%d): hit %d is supposed to be multi-folded but it is not\n",
			__FILE__,__LINE__,ihit);
                return;
              }

          }

        for (j = 0; j < nfchan; j++)
          {
            for (k = 0; k < 2; k++)
              {

                // Check if we have signal.  If SIGNALCN channels
                // have counts exceeding the pedestal value by NRMSSIGNAL * pedrms, then
                // we've got the signal (if can't check that because at the end of the trace,
                // check whatever remains).
                if (!isSignal[k])
                  {
                    cn = ((SIGNALCN <= (nfchan - j)) ? SIGNALCN : (nfchan - j));
                    isSignal[k] = true;
                    for (l = j; l < (j + cn); l++)
                      {
                        if (((real8) rusdraw_.fadc[i][k][l] - mped[k])
                            < (pedrms[k] * (NRMSSIGNAL)))
                          {
                            isSignal[k] = false;
                            break;
                          }
                      }
                    // record the channel where the 1st signal starts
                    // ( if there is a 2nd pulse, this i is still the signal starting i,
                    // the signal stop i will correspond to the end of the last signal pulse.
                    if (isSignal[k] && start_i[k] == 0)
                      {
                        start_i[k] = j; // channel where the signal was first seen
                      }
                  }

                if (isSignal[k])
                  {

                    // Compute the derivative b/w the current channel and the previous channel
                    // If the current channel is where the signal starts, then the derivative
                    // should be positive
                    if (j!=0)
                      {
                        deriv[k] = rusdraw_.fadc[i][k][j]
                            - rusdraw_.fadc[i][k][j - 1];
                      }
                    else
                      {
                        deriv[k] = rusdraw_.fadc[i][k][j];
                      }

                    if (!gotTime[k] && (deriv[k] < 0))
                      {
                        gotTime[k] = true;

                        // channel at which derivative hits zero: current channel has less fadc counts than the previous.
                        zderivc[k] = j - 1;
                      }

                    if (!gotTime[k])
                      {
                        // keep record of the largest derivative channel since the signal start


                        // Keep track of the largest derivative.
                        if (lderiv[k] < deriv[k])
                          {
                            lderiv[k] = deriv[k];
                            // channel at which there is a biggest jump in fadc counts: current channel has
                            // a bigger signal than the previous and this difference is biggest since the
                            // signal start.  Point of inflection is the left edge of this bin.
                            lderivc[k] = j;
                          }
                        // record the information needed for computing rms on time channel
                        tx[k] += deriv[k] * j;
                        tx2[k] += deriv[k] * j * j;
                        ntim[k] += deriv[k]; // sum of weights
                      }

                    parea[k] += rusdraw_.fadc[i][k][j]; // Add signal to the pulse area.
                    nsigped[k]++; // count the pedestal channels under the signal
                    // Check if we are returning back to the pedestal level starting at the next channel.
                    // See if NOSIGNALCN consequtive channels do not deviate from the pedestal level
                    // by more than SIGNALCN * current pedestal rms.
                    // If can't do that test - don't bother, count the last channels as signal.
                    if (NOSIGNALCN < (nfchan - j - 1))
                      {
                        isSignal[k] = false;
                        for (l = j + 1; l < j + NOSIGNALCN + 1; l++)
                          {
                            if ((rusdraw_.fadc[i][k][l] - mped[k])
                                > NRMSSIGNAL * pedrms[k])
                              {
                                isSignal[k] = true;
                                break;
                              }
                          }

                        // Additional test to verify that the signal has really ended.
                        if (!isSignal[k])
                          {
                            nosig_sum[k] = 0.0;
                            for (l = j + 1; l < (j + 1 + NOSIGNALCN); l++)
                              nosig_sum[k] += (real8) rusdraw_.fadc[i][k][l]
                                  - mped[k];
                            if ((nosig_sum[k] / (real8) NOSIGNALCN)
                                > (NRMSSIGNAL * pedrms[k] /
                                sqrt ((real8) NOSIGNALCN)))
                              isSignal[k] = true;
                          }

                      }
                    // Record the channel where the last signal stops.
                    if ((!isSignal[k]) || (j==(nfchan-1)))
                      {
                        stop_i[k] = j;
                        // restart the ped. channel since last signal counter.
                        nped_since_last_signal[k] = 0;
                      }

                  }
                else
                  {
                    nped_since_last_signal[k]++; // count ped. channels since last signal was seen.
                  }

                // for(k=0;k<2;k++ ... loops over upper and lower counters.
              }

            // (for(j=0;j<nfchan;j++ ...
          }

        for (k = 0; k < 2; k++)
          {
            rufptn_.fadcpa[ihit][k] = (real8) parea[k] - mped[k]
                * ((real8) nsigped[k]);
            /* subtract the pedestals (# of signal
             channels times the mean pedestal)
             to obtain the pulse areas. */

            /* error on fadc pulse area */
            rufptn_.fadcpaerr[ihit][k] = sqrt((real8) parea[k]
                + ((real8) (nsigped[k] * nsigped[k])) * pedrms[k] * pedrms[k]);

            // Record the signal start and stop channels, and the channel with the largest derivative
            // these will not change for subsequent wfms in an n-fold hit.
            if (rufptn_.nfold[ihit] == 1)
              {
                rufptn_.sstart[ihit][k] = start_i[k];
                rufptn_.lderiv[ihit][k] = lderivc[k];
                rufptn_.zderiv[ihit][k] = zderivc[k];

                /* Do  [GPSClockCount/(MaxClockCount)+20ns*Channel]x(1E6)(uS/S)
                 Then multiply by TIMDIST, constant which converts times
                 from uS to units of counter separation distance (1200m). */
                //                rufptn_.reltime[ihit][k]
                //                    = ((1e6 * (real8) rusdraw_.clkcnt[i])
                //                        / (real8) rusdraw_.mclkcnt[i] + 0.02
                //                        * ((real8) lderivc[k])) * TIMDIST; // convert to the units of counter separation distance

		clkcnt  = rusdraw_.clkcnt[i];
		mclkcnt = rusdraw_.mclkcnt[i]; 
		
		// if a waveform time appears much later than the trigger microsecond,
		// then it should be 1 second earlier
		if (((double)clkcnt * 50.0 - (double)rusdraw_.usec) > LARGE_TIME_DIFF_uS)
		  clkcnt -= mclkcnt;
		
		// if a waveform time appears much earlier than the trigger microsecond,
		// then it should be 1 second later
		if (((double)clkcnt * 50.0 - (double)rusdraw_.usec) < -LARGE_TIME_DIFF_uS)
		  clkcnt += mclkcnt;
		
                rufptn_.reltime[ihit][k]
		  = ((1e6 * (real8) clkcnt)
		     / (real8) mclkcnt + 0.02
		     * ((real8) start_i[k])
		     + SD_TIME_CORRECTION_uS) * TIMDIST; // convert to the units of counter separation distance


                if (ntim[k] == 0)
                  ntim[k] = 1; // caution


                // Compute the RMS of the FADC derivative between the start signal channel and the
                // channel where the derivative becomes negative (excluding that channel).  Use
                // derivative as the weight.  Convert this RMS into the units of counter separation
                // distance: 1 FADC channel = 20 nS (0.02uS), so convert to uS and then
                // multiply by TIMDIST, the conversion factor to go from uS to the unints of
                // counter separation distance.
                //                rufptn_.timeerr[ihit][k] = sqrt(((real8) tx2[k]
                //                    - ((real8) tx[k] * (real8) tx[k]) / ((real8) ntim[k]))
                //                    / ((real8) ntim[k])) * 0.02 * TIMDIST;

                // Try difference b/w FADC peak and signal start channel for
                // time resolution.
                rufptn_.timeerr[ihit][k] = (rufptn_.zderiv[ihit][k]
                    -rufptn_.sstart[ihit][k])/2.33 * 0.02 * TIMDIST;

                /* Right now, this is the time relative to the latest T-line.
                 Next, will find (and later subtract) the earliest hit time. */

                // Find the earliest hit time (counter separation distance uints)
                if (rufptn_.reltime[ihit][k] < tearliest[k])
                  tearliest[k] = rufptn_.reltime[ihit][k];

                //if(rufptn_.nfold[ihit]==1...
              }
            // keep in mind that the stop channel may go over many 128 - channel FADC windows,
            // as there can be multiple-fold hits.
            rufptn_.sstop[ihit][k] = stop_i[k];

            // for(k=0; k<2; k++ <-- looping over upper and lower layers
          }

        //  for(i=0;i<rusdraw_.nofwf;i++ .. <-- looping over all the waveforms
      }

    // Find the relative times by subtracting the earliest hit time.
    for (i = 0; i < rufptn_.nhits; i++)
      {
        for (k = 0; k < 2; k++)
          rufptn_.reltime[i][k] -= tearliest[k];
      }

    for (k=0; k<2; k++)
      {
        rufptn_.tearliest[k] = (real8)tinsecmdn(rusdraw_.hhmmss) + 
	  tearliest[k]/TIMDIST*1e-6;
      }

    // processFADC() done ...
  }




// Compute the monitoring information needed for the event (fadc counts / vem & error on fadc counts/vem)
// Also find the GPS coordinates from the table.
void rufptnAnalysis::compEventMon()
  {
    integer4 i, j, k, n;
    char problemStr[0x40];
    bool isgood;
    for (j = 0; j < rufptn_.nhits; j++)
      {
        if (rufptn_.isgood[j] < 1)
          continue; // don't bother with counters that are already bad, they have already been reported

        isgood=true;
        rufptn_.isgood[j] = 1;
        if (!sdxyzclf.get_xyz(rusdraw_.yymmdd, rufptn_.xxyy[j], rufptn_.xyzclf[j]))
          {
            sprintf(problemStr, "NO_GPS_INFO");
            isgood = false;
          }
        i = rufptn_.wfindex[j];
        if (rusdraw_.xxyy[i] <= 0)
          {
            sprintf(problemStr, "INVALID_XXYY");
            isgood = false;
          }
        for (k = 0; k < 2; k++)
          {
            if (!isgood)
              break;

            if ( (rusdraw_.pchped[i][k] < 8) || (rusdraw_.pchped[i][k]
                > (SDMON_NMONCHAN/2 - 8)))
              {
                sprintf(problemStr, "PEDESTAL");
                isgood = false;
                break;
              }
            rufptn_.ped[j][k] = rusdraw_.pchped[i][k] / (real8) NPEDINTCH;
            if ((rusdraw_.pchmip[i][k] < 12) ||(rusdraw_.pchmip[i][k]
                > (SDMON_NMONCHAN-12)))
              {
                sprintf(problemStr, "MIP_PEAK_CHANNEL");
                isgood = false;
                break;
              }
            if (rusdraw_.mftndof[i][k]<=0)
              {
                sprintf(problemStr, "MIP_FIT_NDOF");
                isgood = false;
                break;
              }
            if (rusdraw_.mftchi2[i][k]/(real8)rusdraw_.mftndof[i][k]
                > MAXMFTRCHI2)
              {
                sprintf(problemStr, "MIP_FIT_CHI2");
                isgood = false;
                break;
              }
            rufptn_.pederr[j][k] = ((real8) (rusdraw_.rhpchped[i][k]
                - rusdraw_.lhpchped[i][k])) / 2.33 / (real8) NPEDINTCH;
            rufptn_.vem[j][k] = rusdraw_.mip[i][k] * Cos(DegToRad()
                *MEAN_MU_THETA);
            // compute sigma on mip, taking into account sigma on ped.
            rufptn_.vemerr[j][k] = ((real8) (rusdraw_.rhpchmip[i][k]
                - rusdraw_.lhpchmip[i][k])) / 2.33 * Cos(DegToRad()
                *MEAN_MU_THETA);
            rufptn_.vemerr[j][k] = sqrt(rufptn_.vemerr[j][k]
                * rufptn_.vemerr[j][k] + (real8) NMIPINTCH * (real8) NMIPINTCH
                * rufptn_.pederr[j][k] * rufptn_.pederr[j][k]);
            if (rufptn_.vem[j][k] < 1.0)
              {
                sprintf(problemStr, "MIP_VALUE");
                isgood = false;
                break;
              }

          }
	
	// Check for SDs that have been labeled as bad by the
	// bsdinfo DST bank if not told to ignore bsdinfo DST bank 
	// (if the DST bank is absent in the event stream
	// then it has been zeroed out in the main program and will
	// not cause a problem in the below code)
	if(!opt.ignore_bsdinfo)
	  {
	    // to exclude counters that are not working according to the calibration checking
	    // criteria that is used by the TA SD Monte Carlo
	    int not_working_according_to_sdmc_checks = 0;
 	    for (n=0; n < bsdinfo_.nbsds; n++)
 	      {
 		if (rusdraw_.xxyy[i] == bsdinfo_.xxyy[n])
 		  {
		    sdmc_calib_check_bitf(bsdinfo_.bitf[n],&not_working_according_to_sdmc_checks,0);
		    if(not_working_according_to_sdmc_checks)
		      {
			sprintf(problemStr, "BSDINFO %d",bsdinfo_.bitf[n]);
			isgood = false;
		      }
		    break;
		  }
	      }
	  }
	
        if (!isgood)
          {
            rufptn_.isgood[j] = 0;	    
            printBadSDInfo(rufptn_.wfindex[j],problemStr,opt.bad_sd_fp);
            // record "clean" values.
            for (k=0; k<2; k++)
              {
                rufptn_.ped[j][k] = -1.0;
                rufptn_.pederr[j][k] = -1.0;
                rufptn_.vem[j][k] = -1.0;
                rufptn_.vemerr[j][k] = -1.0;
              }
          }
      }
  }

// Determines the charge of each hit, i.e. converts signals from FADC counts to VEM.
void rufptnAnalysis::findCharge()
  {
    integer4 i, k;
    for (i = 0; i < rufptn_.nhits; i++)
      {

        if (rufptn_.isgood[i] > 0)
          {
            for (k = 0; k < 2; k++)
              {
                rufptn_.pulsa[i][k] = rufptn_.fadcpa[i][k] / rufptn_.vem[i][k];
                // using error propagation formula ("sqrt law")
                rufptn_.pulsaerr[i][k] = 1.0 / rufptn_.vem[i][k]
                    * sqrt(rufptn_.fadcpaerr[i][k] * rufptn_.fadcpaerr[i][k]
                        + +rufptn_.pulsa[i][k] * rufptn_.pulsa[i][k]
                            * rufptn_.vemerr[i][k] * rufptn_.vemerr[i][k]);
              }
          }
        else
          {
            for (k=0; k<2; k++)
              {
                rufptn_.pulsa[i][k] = 0.0;
                rufptn_.pulsaerr[i][k] = 0.0;
              }
          }
      }
  }

// checks if the two hits are adjacent in space.
integer4 rufptnAnalysis::areSadjacent(integer4 ih1, integer4 ih2,
    integer4 spaceadj, integer4 * xxyy)
  {
    integer4 xy1[2], xy2[2];
    xycoor(xxyy[ih1], xy1);
    xycoor(xxyy[ih2], xy2);
    // return 1 if the square distance b/w them is less than or equal to 1,
    // otherwise return 0
    return ((((xy1[0] - xy2[0]) * (xy1[0] - xy2[0]) + (xy1[1] - xy2[1])
        * (xy1[1] - xy2[1])) <= spaceadj) ? 1 : 0);
  }

// checks if the two hits can belong to the shower front plane
// use nonzero tolerance if necessary.
integer4 rufptnAnalysis::areInTime(integer4 ih1, integer4 ih2, real8 xyz[][3],
    real8 * xxyyt, real8 timeadjtol)
  {
    real8 dr2, dt2;
    integer4 i;

    // Counter separation distance squared
    dr2=0.0;
    for (i=0; i<3; i++)
      dr2 += (xyz[ih2][i]-xyz[ih1][i])*(xyz[ih2][i]-xyz[ih1][i]);

    // Counter separation time squared
    dt2 = (xxyyt[ih1]-xxyyt[ih2]) * (xxyyt[ih1]-xxyyt[ih2]);

    // Time separation (of adjacent counters) does not exceed their spatial
    // separation wihtin some tolerance.  Speed of light is unitiy in the units used
    // ( both space and time are measured in 1200m units).  But in case
    // one wants to relax this criterea, one can set the speed of light 
    // to be less than unity: the quanity that's passed to rufptn analysis
    // is stc, default value is 1 but one can changed that by 
    // setting the corresponding parameter in the argument line
    return ((dt2*opt.stc*opt.stc < (dr2 + timeadjtol * timeadjtol)) ? 1 : 0);
  }

// space pattern recognition, find largest space cluster
void rufptnAnalysis::spacePatRecog(integer4 nofwf, integer4 * xxyy,
    integer4 * nclust, integer4 * clust)
  {
    integer4 a[RUFPTNMH], b[RUFPTNMH];
    integer4 i, j, k, asize, bsize, no_adjacent;
    real8 q1,q2;
    // pick counters which have (either upper or lower layers) > 1.5 VEM charge
    asize = 0;
    for (i = 0; i < nofwf; i++)
      {
        if (rufptn_.isgood[i] < 1)
          continue;
        if ((rufptn_.pulsa[i][0]+rufptn_.pulsa[i][1])/2.0 < QMIN)
          continue;
        a[asize] = i;
        asize++;
      }
    if (asize==0)
      a[0]=0;
    // Initialize the maximum number of events in a cluster, and
    // set the first hit to be the only hit in the largest cluster.
    // Latter this will get changed, if can find bigger clusters than
    // 1 counter.
    *nclust = 1;
    clust[0] = a[0];
    // repeat the procedure until there are no entries
    // unchecked for space clustering
    while (asize > 0)
      {
        // Move 1st entry in a-array to b-array
        b[0] = a[0];
        bsize = 1;
        asize--; // decrease the size of a (& shift all the entries up)
        for (i = 0; i < asize; i++)
          a[i] = a[i + 1];
        // Go over the remaining a-entries
        k = 0;
        // Check if an a-entry clusters with any b-entries
        while (k < asize)
          {
            j = -1;
            // This flag remains true if we haven't found any
            // adjacent for a given k-value
            no_adjacent = 1;
            do
              {
                j++;
                if (areSadjacent(a[k], b[j], SPACEADJ, xxyy))
                  {
                    no_adjacent = 0; // don't need to loop over the b-entries
                    b[bsize] = a[k]; // add that a-entry to b-array
                    bsize++; // increase the size of b
                    asize--; // remove the k-entry from a-array
                    for (i = k; i < asize; i++)
                      a[i] = a[i + 1];
                    k = 0; // b is changed, so need to re-check all a-entries
                  } // if(areSadjacent(pass0,a[k] ...
                // stop if j equals to bsize-1 because the next j
                // will be equal to bsize but b[bsize] doesn't contain
                // entries (we did b++ after assigning b[bsize])
              } while (no_adjacent && (j < (bsize - 1)));
            if (no_adjacent)
              k++; // no adjacent b's, so consider next a-entry
          } // while (k<asize) ...
        // Left with events which don't cluster with the first b-entry.
        // Take a record of the size of the cluster if
        // it exceeds the size of the one measured previously and
        // also record the intra-event indecies in that cluster
        if (bsize >= *nclust)
          {
            if (bsize > *nclust)
              {
                *nclust = bsize;
                for (i = 0; i < bsize; i++)
                  {
                    clust[i] = b[i];
                  }
              }
            // If clusters are equal in size, choose the one that has a larger
            // total charge
            else
              {
                q1 = 0.0;
                q2 = 0.0;
                for (i=0; i<bsize; i++)
                  {
                    q1 += (rufptn_.pulsa[clust[i]][0]+rufptn_.pulsa[clust[i]][1])/2.0;
                    q2 += (rufptn_.pulsa[b[i]][0]+rufptn_.pulsa[b[i]][1])/2.0;
                  }
                if ( q2 > q1)
                  {
                    for (i = 0; i < bsize; i++)
                      {
                        clust[i] = b[i];
                      }
                  }
              }
          }

        // while(asize>0)
      }
    // spacePatRecog ...
  }

// To find cluster of time contiguous hits
void rufptnAnalysis::timePatRecog(integer4 * xxyy, real8 xyz[][3],
    real8 * xxyyt, integer4 nsclust, integer4 * sclust, integer4 * nclust,
    integer4 * clust)
  {
    integer4 a[RUFPTNMH], b[RUFPTNMH];
    integer4 i, j, k, l, asize, bsize, no_adjacent;
    asize = 0;
    // don't use counters with bad time / charge information in plane pattern recognition.
    for (i = 0; i < nsclust; i++)
      {
        if (sclust[i] < 0 || sclust[i] >= RUFPTNMH)
          {
            fprintf(stderr, "\n fatal error: %s(%d): sclust[i] = %d\n", 
		    __FILE__,__LINE__,sclust[i]);
            exit(2);
          }
        if (rufptn_.isgood[sclust[i]] > 0)
          {
            a[asize] = sclust[i];
            asize++;
          }
      }
    if (asize==0)
      a[0] = 0;
    // Initialize the maximum number of events in a space-time cluster
    // Let the initial S-T cluster contain the 1st hit that's in the
    // largest space cluster.  Latter, this will change, if can
    // find clusters bigger than 1 counter.
    *nclust = 1;
    clust[0] = a[0];
    // repeat the procedure untill there are no entries
    // unchecked for space clustering
    while (asize > 0)
      {
        // Move 1st entry in a-array to b-array
        b[0] = a[0];
        bsize = 1;
        asize--; // decrease the size of a (& shift all the entries up)
        for (i = 0; i < asize; i++)
          a[i] = a[i + 1];
        // Go over the remaining a-entries
        k = 0;
        // Check if an a-entry clusters with any b-entries
        while (k < asize)
          {
            j = -1;
            // This flag remains true if we haven't found any
            // adjacenct for a given k-value
            no_adjacent = 1;
            do
              {
                j++;
                if (areSadjacent(a[k], b[j], SPACEADJ, xxyy) && areInTime(a[k],
                    b[j], xyz, xxyyt, TIMEADJTOL))
                  {

                    /******** Excluding the multiple hits (below) ********/
                    // Multiple hits are not allowed in any S-T cluster
                    l = 0;
                    while (no_adjacent && l < bsize)
                      {
                        if (xxyy[b[l]] == xxyy[a[k]])
                          no_adjacent = 0;
                        l++;
                      }
                    if (no_adjacent == 0)
                      {
                        no_adjacent = 1;
                        continue;
                      }
                    /**** Excluding the multiple hits (above) */

                    no_adjacent = 0; // don't need to loop over the b-entries
                    b[bsize] = a[k]; // add that a-entry to b-array
                    bsize++; // increase the size of b
                    asize--; // remove the k-entry from a-array
                    for (i = k; i < asize; i++)
                      a[i] = a[i + 1];
                    k = 0; // b is changed, so need to re-check all a-entries
                  } // if(areTadjacent(pass0,a[k] ...
                // stop if j equals to bsize-1 because the next j
                // will be equal to bsize but b[bsize] doesn't contain
                // entries (we did b++ after assigning b[bsize])
              } while (no_adjacent && (j < (bsize - 1)));
            if (no_adjacent)
              k++; // no adjacent b's, so consider next a-entry
          } // while (k<asize) ...
        // Left with events which don't cluster with the first b-entry.
        // Take a record of the size of the cluster if
        // it exceeds the size of the one measured previously and
        // also record the intra-event indecies in that cluster
        if (bsize > *nclust)
          {
            *nclust = bsize;
            for (i = 0; i < bsize; i++)
              clust[i] = b[i];
          }

      } // while(asize>0)


    // Now, we remove counters that are not in time with all of their neighbors.
    // We count how many times a given counter is out of time with its neighbors, and if
    // this number is 2 or more, then this counter is clearly out of time.


    for (i=0; i<(*nclust); i++)
      {
        l=0;
        for (j=0; j<(*nclust); j++)
          {
            if (j==i)
              continue;
            if ((areSadjacent(clust[i], clust[j], SPACEADJ, xxyy)==1)
                && (areInTime(clust[i], clust[j], xyz, xxyyt, TIMEADJTOL)==0))
              l++;
          }
        if (l>1)
          {
            (*nclust) -= 1;
            for (j=i; j<(*nclust); j++)
              clust[j]=clust[j+1];
          }

      }

    // time pattern recognition
  }

void rufptnAnalysis::combineMfHits(integer4 ihit1, integer4 ihit2)
  {
    integer4 i, k;
    // Foldedness of the combined signal is a sum of the two
    // foldednesses.
    rufptn_.nfold[ihit1] += rufptn_.nfold[ihit2];

    // Additional quantities that need to be merged when we combine a signal
    for (k=0; k<2; k++)
      {
        // Signal stop channel is that of the 2nd signal
        rufptn_.sstop[ihit1][k] = rufptn_.sstop[ihit2][k];

        // FADC pulse area of the combined signal
        rufptn_.fadcpa[ihit1][k] += rufptn_.fadcpa[ihit2][k];

        // Error on FADC pulse area of the combined signal adds in quadrature
        rufptn_.fadcpaerr[ihit1][k] = sqrt(rufptn_.fadcpaerr[ihit1][k]
            * rufptn_.fadcpaerr[ihit1][k] + rufptn_.fadcpaerr[ihit2][k]
            * rufptn_.fadcpaerr[ihit2][k]);

        // Pulse area in VEM of the combined signal
        rufptn_.pulsa[ihit1][k] += rufptn_.pulsa[ihit2][k];

        // Error on the combined pulse area in VEM adds in quadrature
        rufptn_.pulsaerr[ihit1][k] = sqrt(rufptn_.pulsaerr[ihit1][k]
            * rufptn_.pulsaerr[ihit1][k] + rufptn_.pulsaerr[ihit2][k]
            * rufptn_.pulsaerr[ihit2][k]);

      }

    // Now, remove the 2nd signal from the rufptn list.
    rufptn_.nhits--;
    for (i=ihit2; i<rufptn_.nhits; i++)
      {
        rufptn_.isgood [i] = rufptn_.isgood [i+1];
        rufptn_.wfindex [i] = rufptn_.wfindex [i+1];
        rufptn_.xxyy [i] = rufptn_.xxyy [i+1];
        rufptn_.nfold [i] = rufptn_.nfold [i+1];
        memcpy(rufptn_.xyzclf[i], rufptn_.xyzclf[i+1], 3*sizeof(real8));
        for (k=0; k<2; k++)
          {
            rufptn_.sstart [i][k] = rufptn_.sstart [i+1][k];
            rufptn_.sstop [i][k] = rufptn_.sstop [i+1][k];
            rufptn_.lderiv [i][k] = rufptn_.lderiv [i+1][k];
            rufptn_.zderiv [i][k] = rufptn_.zderiv [i+1][k];
            rufptn_.reltime [i][k] = rufptn_.reltime [i+1][k];
            rufptn_.timeerr [i][k] = rufptn_.timeerr [i+1][k];
            rufptn_.fadcpa [i][k] = rufptn_.fadcpa [i+1][k];
            rufptn_.fadcpaerr [i][k] = rufptn_.fadcpaerr [i+1][k];
            rufptn_.pulsa [i][k] = rufptn_.pulsa [i+1][k];
            rufptn_.pulsaerr [i][k] = rufptn_.pulsaerr [i+1][k];
            rufptn_.ped [i][k] = rufptn_.ped [i+1][k];
            rufptn_.pederr [i][k] = rufptn_.pederr [i+1][k];
            rufptn_.vem [i][k] = rufptn_.vem [i+1][k];
            rufptn_.vemerr [i][k] = rufptn_.vemerr [i+1][k];
          }

      }

  }

void rufptnAnalysis::combineSignals()
  {
    integer4 ihit;

    // Flag to indicate that some of the signals were combined
    bool combined_signals;

    combined_signals = false;
    for (ihit=0; ihit < (rufptn_.nhits - 1); ihit++)
      {
        // Check if a given signal is picked out by time pattern regnition as
        // signal from shower and there is another signal in the same counter
        // right after the chosen signal
        if ((rufptn_.isgood[ihit]>=3) && (rufptn_.xxyy[ihit]
            ==rufptn_.xxyy[ihit+1]))
          {

            if ( ((rufptn_.reltime[ihit+1][0]-rufptn_.reltime[ihit][0])
                < SAMESD_TIME_CORR) || ((rufptn_.reltime[ihit+1][1]
                -rufptn_.reltime[ihit][1]) < SAMESD_TIME_CORR))
              {

                combineMfHits(ihit, ihit+1);
                combined_signals = true;

                // So that the next adjacent signal is also checked for time
                // correlation with the current combined signal.
                ihit--;
              }
          }
      }

    // If some of the signals were combined, then need to re-do the array
    // which contain space and time cluster indices.
    if (combined_signals)
      {
        nspaceclust = 0;
        nspacetimeclust = 0;
        for (ihit = 0; ihit < rufptn_.nhits; ihit++)
          {
            if (rufptn_.isgood[ihit] >= 2)
              {
                spaceclust[nspaceclust] = ihit;
                nspaceclust ++;
              }
            if (rufptn_.isgood[ihit] >= 3)
              {
                spacetimeclust[nspacetimeclust] = ihit;
                nspacetimeclust ++;
              }

          }
        rufptn_.nsclust = nspaceclust;
        rufptn_.nstclust = nspacetimeclust;
      }
  }

// Tyro geometry reconstruction by using plane approximation to shower front.
bool rufptnAnalysis::tyroGeom()
  {
    // For preparing a t vs u graph
    real8 t[RUSDRAWMWF], dt[RUSDRAWMWF], u[RUSDRAWMWF], du[RUSDRAWMWF];

    integer4 i, j, k, l, n;

    Double_t xy[2]; // to obtain the coordinates of a given SD
    /* [0]-lower, [1]-upper, [2] - using both upper and lower
     nMu[] - muon # from calibration or just FADC areas
     if Calibration is false (calibration not done)
     xm[],ym[] - to find the core location
     a11[],a12[],a22[] - symmetric 2nd moment matrix,
     about the core location.
     w[] - weight (total muon #), to divide by at the end
     and find means, moments, etc
     */
    real8 xm[3], ym[3], a11[3], a22[3], a12[3], w[3];
    /* Auxiallary variables ([0]-lower,[1]-upper, [2] - upper & lower):
     lambda[] - current eigenvalue
     delta[] - needed for computing the eigenvectors
     nfactor[] - needed for normalizing the eigenvectors to unity
     */
    real8 lambda[3], delta[3], nfactor[3];

    bool errFlag;

    // make sure have charge and time for upper, lower, and both
    for (i = 0; i < rufptn_.nhits; i++)
      {
        for (k = 0; k < 3; k++)
          {
            if (k < 2)
              {
                charge[i][k] = rufptn_.pulsa[i][k];
                relTime[i][k] = rufptn_.reltime[i][k];
                chargeErr[i][k] = rufptn_.pulsaerr[i][k];
                timeErr[i][k] = rufptn_.timeerr[i][k];
              }
            else
              {
                charge[i][k] = (rufptn_.pulsa[i][0] + rufptn_.pulsa[i][1])
                    / 2.0;
                chargeErr[i][k] = sqrt(rufptn_.pulsaerr[i][0]
                    * rufptn_.pulsaerr[i][0] + rufptn_.pulsaerr[i][1]
                    * rufptn_.pulsaerr[i][1]) / 2.0;
                relTime[i][k] = (rufptn_.reltime[i][0] + rufptn_.reltime[i][1])
                    / 2.0;
                timeErr[i][k] = sqrt(rufptn_.timeerr[i][0]
                    * rufptn_.timeerr[i][0] + rufptn_.timeerr[i][1]
                    * rufptn_.timeerr[i][1]) / 2.0;
              }
          }
      }

    //initialize the variables
    for (k = 0; k < 3; k++)
      { // k=0 - lower, k=1 - upper counters
        xm[k] = 0.0;
        ym[k] = 0.0;
        a11[k] = 0.0;
        a22[k] = 0.0;
        a12[k] = 0.0;
        w[k] = 0.0;

        rufptn_.tyro_phi[k] = 0.0;
        rufptn_.tyro_theta[k] = 0.0;
      }
    errFlag = false;
    for (i = 0; i < nspacetimeclust; i++)
      { // go over counters in the largest ST-cluster
        j = spacetimeclust[i]; // intra-event index of the counter in the cluster

        // SD XY coordinates in CLF frame with respect to SD origin
        xy[0] = rufptn_.xyzclf[j][0] - sdorigin_xy[0];
        xy[1] = rufptn_.xyzclf[j][1] - sdorigin_xy[1];

        /////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////
        // EXCLUDE ANY BAD COUNTERS HERE AND USE THE "continue" statement.
        /////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////


        // exclude counters that were not calibrated
        if (rufptn_.vem[j][0] < 1.0 || rufptn_.vem[j][1] < 1.0)
	  continue;

        // To find the first moments for upper and lower
        for (k = 0; k < 3; k++)
          { // k=0 - lower, k=1 - upper, k=2 - both
            // Use pulse areas in upper, lower, or both counters, depending
            // on the k-values
            xm[k] += xy[0] * charge[j][k];
            ym[k] += xy[1] * charge[j][k];
            // To find the 2nd moments
            a11[k] += xy[0] * xy[0] * charge[j][k];
            a22[k] += xy[1] * xy[1] * charge[j][k];
            a12[k] += xy[0] * xy[1] * charge[j][k];
            w[k] += charge[j][k]; // sum of all weights

          }
      } // for(i=0;i<nspacetimeclust  (to find the moments)

    for (k = 0; k < 3; k++)
      { // k=0 - lower, k=1 - upper, k=2 - both
        if (k<2)
          rufptn_.qtot[k] = w[k]; // record the total charge in lower and upper
	if (w[k] * w[k] < 1e-5)
	  errFlag = true;
	if (errFlag)              // don't proceed if the total charge in either upper or lower layers is zero; bad event
          continue;
        // Compute the core location and 2nd moments
        // put them into corresponding class variables
        xm[k] /= w[k]; // Core of the shower (<x>,<y>)
        rufptn_.tyro_xymoments[k][0] = xm[k];
        ym[k] /= w[k];
        rufptn_.tyro_xymoments[k][1] = ym[k];
        a11[k] /= w[k]; // <x^2> about the core at (<x>,<y>)
        a11[k] -= xm[k] * xm[k];
        rufptn_.tyro_xymoments[k][2] = a11[k];
        a12[k] /= w[k]; // <xy> about the core at (<x>,<y>)
        a12[k] -= xm[k] * ym[k];
        rufptn_.tyro_xymoments[k][3] = a12[k];
        a22[k] /= w[k]; // <y^2> about the core at (<x>,<y>)
        a22[k] -= ym[k] * ym[k];
        rufptn_.tyro_xymoments[k][4] = a22[k];

        // Do linear algebra (analytically):
        // find the principal 2nd moments and find the
        // corresponding long and short axes
        // If the matrix is already diagonal, don't do any extra work
        if (a12[k] * a12[k] < 1e-10)
          {
            // if <x^2> is the largest one, pick (1,0) as the long axis
            // and (0,1) as the short axis; record the principal
            // moments, larger comes first
            if (rufptn_.tyro_xymoments[k][2] > rufptn_.tyro_xymoments[k][4])
              {
                rufptn_.tyro_xypmoments[k][0] = rufptn_.tyro_xymoments[k][2];
                rufptn_.tyro_u[k][0] = 1.0;
                rufptn_.tyro_u[k][1] = 0.0;
                rufptn_.tyro_xypmoments[k][1] = rufptn_.tyro_xymoments[k][4];
                rufptn_.tyro_v[k][0] = 0.0;
                rufptn_.tyro_v[k][1] = 1.0;
              }
            else
              {
                // otherwise, pick (1,0) as long and (0,1) as long
                // also, flip the principal moments so that the larger
                // one comes first
                rufptn_.tyro_xypmoments[k][0] = rufptn_.tyro_xymoments[k][4];
                rufptn_.tyro_u[k][0] = 0.0;
                rufptn_.tyro_u[k][1] = 1.0;
                rufptn_.tyro_xypmoments[k][1] = rufptn_.tyro_xymoments[k][2];
                rufptn_.tyro_v[k][0] = 1.0;
                rufptn_.tyro_v[k][1] = 0.0;
              }
          }
        else
          {
            // The larger eigenvalue:
            lambda[k] = ((a11[k] + a22[k]) + sqrt((a11[k] - a22[k]) * (a11[k]
                - a22[k]) + 4.0 * a12[k] * a12[k])) / 2.0;
            rufptn_.tyro_xypmoments[k][0] = lambda[k]; // record the larger eigenvalue
            // useful variable to express the eigenvectors
            delta[k] = (lambda[k] - a22[k]) / a12[k];
            // variable to normalize the eigenvectors
            nfactor[k] = 1.0 / sqrt(1.0 + delta[k] * delta[k]);
            // eigenvector x,y components for long axis
            rufptn_.tyro_u[k][0] = nfactor[k] * delta[k];
            rufptn_.tyro_u[k][1] = nfactor[k];
            // The smaller eigenvalue:
            lambda[k] = ((a11[k] + a22[k]) - sqrt((a11[k] - a22[k]) * (a11[k]
                - a22[k]) + 4.0 * a12[k] * a12[k])) / 2.0;
            rufptn_.tyro_xypmoments[k][1] = lambda[k]; // record the smaller eigenvalue
            // useful variable to express the eigenvectors
            delta[k] = (lambda[k] - a22[k]) / a12[k];
            // variable to normalize the eigenvectors
            nfactor[k] = 1.0 / sqrt(1.0 + delta[k] * delta[k]);
            // eigenvector x,y components for long axis
            rufptn_.tyro_v[k][0] = nfactor[k] * delta[k];
            rufptn_.tyro_v[k][1] = nfactor[k];
          } // else (cases when the matrix is not already diagonal)


        // These are necessary for the time fitting (t vs u)
        n = 0;
        for (i = 0; i < nspacetimeclust; i++)
          {
            l = spacetimeclust[i];
            t[n] = relTime[l][k];
            // dt[n] = timeErr[l][k];
            // temporarily, set errors to some pre-defined constant
            dt[n] = RUFPTN_TYRO_dt;
            xy[0] = rufptn_.xyzclf[l][0] - sdorigin_xy[0];
            xy[1] = rufptn_.xyzclf[l][1] - sdorigin_xy[1];
            u[n] = (rufptn_.tyro_u[k][0] * (xy[0]
                - rufptn_.tyro_xymoments[k][0]) + rufptn_.tyro_u[k][1] * (xy[1]
                - rufptn_.tyro_xymoments[k][1]));
            du[n] = 0.0;
            n++;
          }

        if (n < 1)
          n = 1; // for safety

        // Fit if have 2 or more fit points
        if (n > 1)
          {
            if (gLaxisFit[k] != 0)
              {
                gLaxisFit[k]->Delete();
                gLaxisFit[k] = 0;
              }
            gLaxisFit[k] = new TGraphErrors (n, u, t, du, dt);
            gLaxisFit[k]->Fit("pol1", "F,0,Q");
            // constant offset
            rufptn_.tyro_tfitpars[k][0] = gLaxisFit[k]->GetFunction ("pol1")->GetParameter(0);
            // slope
            rufptn_.tyro_tfitpars[k][1] = gLaxisFit[k]->GetFunction ("pol1")->GetParameter(1);
            rufptn_.tyro_chi2[k] = (real8)(gLaxisFit[k]->GetFunction("pol1")->GetChisquare());
            rufptn_.tyro_ndof[k] = (real8)(gLaxisFit[k]->GetFunction("pol1")->GetNDF());
          }
        else
          {
            rufptn_.tyro_tfitpars[k][0] = t[0]; // Set time offset to just the time of the data point
            rufptn_.tyro_tfitpars[k][1] = 1e-3; // Can't possibly know the slope, so set it to some non-zero value
            rufptn_.tyro_chi2[k] = 1.e6;
            rufptn_.tyro_ndof[k] = -1;
          }
        // If the slope is negative, flip the IND's ordered along axis,
        // direction of the long axis, and the slope itself
        if (rufptn_.tyro_tfitpars[k][1] < 0.0)
          {
            // Flip the shower long axis and counter ordering along that axis
            for (i = 0; i < 2; i++)
              rufptn_.tyro_u[k][i] *= -1.0;

            // Reflect the distances along the long axis
            for (i = 0; i < n; i++)
              u[i] *= -1.0;
            // Do another fit.
            if (gLaxisFit[k] != 0)
              {
                gLaxisFit[k]->Delete();
                gLaxisFit[k] = 0;
              }
            gLaxisFit[k] = new TGraphErrors (n, u, t, du, dt);
            gLaxisFit[k]->Fit("pol1", "0,F,Q");
            // constant offset
            rufptn_.tyro_tfitpars[k][0] = gLaxisFit[k]->GetFunction ("pol1")->GetParameter(0);
            // slope
            rufptn_.tyro_tfitpars[k][1] = gLaxisFit[k]->GetFunction ("pol1")->GetParameter(1);
            rufptn_.tyro_chi2[k] = (real8)(gLaxisFit[k]->GetFunction("pol1")->GetChisquare());
            rufptn_.tyro_ndof[k] = (real8)(gLaxisFit[k]->GetFunction("pol1")->GetNDF());
          }


	rufptn_.tyro_phi[k] = ATan2(rufptn_.tyro_u[k][1], rufptn_.tyro_u[k][0]) * RadToDeg();
        rufptn_.tyro_phi[k] = tacoortrans::range(rufptn_.tyro_phi[k],360.0);
        
	if (rufptn_.tyro_tfitpars[k][1] <= 1.0)
          {
            rufptn_.tyro_theta[k] = RadToDeg()
                * ASin(rufptn_.tyro_tfitpars[k][1]);
          }
        else
	  rufptn_.tyro_theta[k] = 90.0;

        // for(k=0;k<3;k++) (k=0-lower,k=1-upper, k=2 - both)
      }

    if (errFlag)
      return false;

    /*
     Find distances from core of all counters, using core reconstructions from:
     k = 0: lower
     k = 1: upper
     k = 2: upper and lower (average)
     */
    for (i = 0; i < rufptn_.nhits; i++)
      {
        xy[0] = rufptn_.xyzclf[i][0] - sdorigin_xy[0];
        xy[1] = rufptn_.xyzclf[i][1] - sdorigin_xy[1];
        for (k = 0; k < 3; k++)
          {
            rufptn_.tyro_cdist[k][i] = sqrt((xy[0]
                - rufptn_.tyro_xymoments[k][0]) * (xy[0]
                - rufptn_.tyro_xymoments[k][0]) + (xy[1]
                - rufptn_.tyro_xymoments[k][1]) * (xy[1]
                - rufptn_.tyro_xymoments[k][1]));
          }
      }

    return true;
  }

bool rufptnAnalysis::geomFit()
  {
    integer4 i;

    // Load variables into the fitter
    p1geomfitter->loadVariables();

    // Clean space-time cluster using  modified Linsley's fit.
    p1geomfitter->cleanClust(P1GEOM_DCHI2);

    // Don't do any further fitting if # of good points is less than 3
    if (p1geomfitter->ngpts < 3)
      {
        for (i=0; i<3; i++)
          {
            rusdgeom_.xcore[i] = 0.0;
            rusdgeom_.dxcore[i] = 0.0;

            rusdgeom_.ycore[i] = 0.0;
            rusdgeom_.dycore[i] = 0.0;

            rusdgeom_.t0[i] = 0.0;
            rusdgeom_.dt0[i] = 0.0;

            rusdgeom_.theta[i] = 0.0;
            rusdgeom_.dtheta[i] = 0.0;

            rusdgeom_.phi[i] = 0.0;
            rusdgeom_.dphi[i] = 0.0;

            rusdgeom_.chi2[i] = 1.e6;
            rusdgeom_.ndof[i] = p1geomfitter->ndof;
          }

        rusdgeom_.a  = 0.0;
        rusdgeom_.da = 0.0;

        return false;
      }

    // Fitting into modified Linsley
    p1geomfitter->doFit(1);
    if (p1geomfitter->chi2 > 1.e6)
      p1geomfitter->chi2 = 1.e6;
    rusdgeom_.ndof[1] = p1geomfitter->ndof;

    rusdgeom_.xcore[1] = p1geomfitter->R[0]; // Core X position
    rusdgeom_.dxcore[1] = p1geomfitter->dR[0];

    rusdgeom_.ycore[1] = p1geomfitter->R[1]; // Core Y position
    rusdgeom_.dycore[1] = p1geomfitter->dR[1];

    rusdgeom_.t0[1] = p1geomfitter->T0; // Time of the core hit
    rusdgeom_.dt0[1] = p1geomfitter->dT0;

    rusdgeom_.theta[1] = p1geomfitter->theta; // Zenith angle
    rusdgeom_.dtheta[1] = p1geomfitter->dtheta;

    rusdgeom_.phi[1] = p1geomfitter->phi; // Azimuthal angle
    rusdgeom_.dphi[1] = p1geomfitter->dphi;

    rusdgeom_.chi2[1] = p1geomfitter->chi2; // Chi2
    rusdgeom_.ndof[1] = p1geomfitter->ndof; // # of degrees of freedom

    if(!p1geomfitter->hasConverged())
      rusdgeom_.chi2[1]=1.e6;


    // Plane fitting.
    p1geomfitter->doFit(0);
    if (p1geomfitter->chi2 > 1.e6)
      p1geomfitter->chi2 = 1.e6;
    rusdgeom_.xcore[0] = p1geomfitter->R[0]; // Core X position
    rusdgeom_.dxcore[0] = p1geomfitter->dR[0];

    rusdgeom_.ycore[0] = p1geomfitter->R[1]; // Core Y position
    rusdgeom_.dycore[0] = p1geomfitter->dR[1];

    rusdgeom_.t0[0] = p1geomfitter->T0; // Time of the core hit
    rusdgeom_.dt0[0] = p1geomfitter->dT0;

    rusdgeom_.theta[0] = p1geomfitter->theta; // Zenith angle
    rusdgeom_.dtheta[0] = p1geomfitter->dtheta;

    rusdgeom_.phi[0] = p1geomfitter->phi; // Azimuthal angle
    rusdgeom_.dphi[0] = p1geomfitter->dphi;

    rusdgeom_.chi2[0] = p1geomfitter->chi2; // Chi2
    rusdgeom_.ndof[0] = p1geomfitter->ndof; // # of degrees of freedom

    if(!p1geomfitter->hasConverged())
      rusdgeom_.chi2[0]=1.e6;

    // Final fit variables (Linsley's with variable curvature)

    p1geomfitter->doFit(3);
    if (p1geomfitter->chi2 > 1.e6)
      p1geomfitter->chi2 = 1.e6;
    rusdgeom_.xcore[2] = p1geomfitter->R[0]; // Core X position
    rusdgeom_.dxcore[2] = p1geomfitter->dR[0];

    rusdgeom_.ycore[2] = p1geomfitter->R[1]; // Core Y position
    rusdgeom_.dycore[2] = p1geomfitter->dR[1];

    rusdgeom_.t0[2] = p1geomfitter->T0; // Time of the core hit
    rusdgeom_.dt0[2] = p1geomfitter->dT0;

    rusdgeom_.theta[2] = p1geomfitter->theta; // Zenith angle
    rusdgeom_.dtheta[2] = p1geomfitter->dtheta;

    rusdgeom_.phi[2] = p1geomfitter->phi; // Azimuthal angle
    rusdgeom_.dphi[2] = p1geomfitter->dphi;

    rusdgeom_.a = p1geomfitter->a;         // Curvature parameter
    rusdgeom_.da = p1geomfitter->da;

    rusdgeom_.chi2[2] = p1geomfitter->chi2; // Chi2
    rusdgeom_.ndof[2] = p1geomfitter->ndof; // # of degrees of freedom

    if(!p1geomfitter->hasConverged())
      rusdgeom_.chi2[2]=1.e6;

    return true;
  }

void rufptnAnalysis::changeStclust()
  {
    integer4 ipoint;
    integer4 ihit;
    nspacetimeclust = 0;
    for (ipoint=0; ipoint<p1geomfitter->ngpts; ipoint++)
      {
        ihit=p1geomfitter->goodpts[ipoint];
        rufptn_.isgood[ihit]=4;
        spacetimeclust[nspacetimeclust]=ihit;
        nspacetimeclust++;
      }
    rufptn_.nstclust = nspacetimeclust;
  }

bool rufptnAnalysis::isSaturated(integer4 ihit)

  {
    integer4 k;
    TSPECTRUM_DECONVOLUTION_TYPE source[2][128];

    integer4 iwf0, iwf;
    integer4 nbins, ibin;

    // First of all, if the total charge of the hit is less than the value
    // needed in 20nS to saturate the counter, we don't do the convolution, as
    // it is clear that the counter will not be saturated.
    if ( (rufptn_.pulsa[ihit][0] < QSAT20NSVEM) && (rufptn_.pulsa[ihit][1]
        < QSAT20NSVEM))
      return false;

    nbins = 128;

    // Get the rusdraw index of the first 128 fadc window in the signal
    iwf0 = rufptn_.wfindex[ihit];

    // Check each 128 FADC trace window that goes into signal for saturation
    for (iwf = iwf0; iwf < (iwf0+rufptn_.nfold[ihit]); iwf++)
      {

        // Go over lower and upper scintillators
        for (k=0; k<2; k++)
          {
            // Fill the (signal-pedestal) histograms
            for (ibin = 0; ibin < nbins; ibin ++)
              {

                source[k][ibin] = (TSPECTRUM_DECONVOLUTION_TYPE)rusdraw_.fadc[iwf][k][ibin]
                    - (TSPECTRUM_DECONVOLUTION_TYPE)rufptn_.ped[ihit][k];

                // Set the negative bins to zero.
                if (source[k][ibin] < 0.0)
                  source[k][ibin] = 0.0;

              }

            // Do the de-convolution
            sNpart->Deconvolution(source[k], muresp_shape, nbins, NDECONV_ITER,
                1, 1.0);

            // If any bin in the de-convoluted signal exceeds the threshold,
            // the signal is saturating the counter.
            for (ibin = 0; ibin < nbins; ibin ++)
              {
                if (source[k][ibin] / (real4)rufptn_.vem[ihit][k] > QSAT20NSVEM)
                  return true;
              }

          }
      }

    return false;

  }

void rufptnAnalysis::labelSaturatingHits()
  {
    integer4 ihit;
    for (ihit = 0; ihit < rufptn_.nhits; ihit ++)
      {
        if (rufptn_.isgood[ihit] < 4)
          continue;
        if (isSaturated(ihit))
          {
            rufptn_.isgood[ihit] = 5;
          }
      }
  }

bool rufptnAnalysis::put2rusdgeom()
  {
    integer4 ic, ih, is;

    ic = 0;
    for (ih=0; ih<rufptn_.nhits; ih++)
      {
        if ((ic>0) && (rufptn_.xxyy[ih]==rusdgeom_.xxyy[ic-1]))
          {
            rusdgeom_.irufptn[ic-1][rusdgeom_.nsig[ic-1]]=ih;
            rusdgeom_.sdsigq[ic-1][rusdgeom_.nsig[ic-1]]= 0.5
                *(rufptn_.pulsa[ih][0]+rufptn_.pulsa[ih][1]);
            rusdgeom_.sdsigt[ic-1][rusdgeom_.nsig[ic-1]]= 0.5
                *(rufptn_.reltime[ih][0]+rufptn_.reltime[ih][1]);
            rusdgeom_.sdsigte[ic-1][rusdgeom_.nsig[ic-1]]= 0.5
                *sqrt(rufptn_.timeerr[ih][0]*rufptn_.timeerr[ih][0]
                    + rufptn_.timeerr[ih][1]*rufptn_.timeerr[ih][1]);
            rusdgeom_.igsig[ic-1][rusdgeom_.nsig[ic-1]]=rufptn_.isgood[ih];
            rusdgeom_.nsig[ic-1]++;
          }
        else
          {
            rusdgeom_.igsd[ic]=1;
            rusdgeom_.xxyy[ic]=rufptn_.xxyy[ih];
            memcpy(rusdgeom_.xyzclf[ic], rufptn_.xyzclf[ih], (integer4)(3
                *sizeof(real8)));
            rusdgeom_.irufptn[ic][0]=ih;
            rusdgeom_.sdsigq[ic][0]= 0.5*(rufptn_.pulsa[ih][0]
                +rufptn_.pulsa[ih][1]);
            rusdgeom_.sdsigt[ic][0]= 0.5*(rufptn_.reltime[ih][0]
                +rufptn_.reltime[ih][1]);
            rusdgeom_.sdsigte[ic][0]= 0.5*sqrt(rufptn_.timeerr[ih][0]
                *rufptn_.timeerr[ih][0]+ rufptn_.timeerr[ih][1]
                *rufptn_.timeerr[ih][1]);
            rusdgeom_.igsig[ic][0]=rufptn_.isgood[ih];
            rusdgeom_.nsig[ic]=1;
            ic++;
          }
        // Hit is a part of space-time cluster, and passed the chi2-cleaning procedure
        if (rufptn_.isgood[ih]==4)
          {
            rusdgeom_.igsd[ic-1] = 2;
          }
        // Label the saturated counters
        if (rufptn_.isgood[ih]==5)
          {
            rusdgeom_.igsd[ic-1] = 3;
          }
        // Bad counter
        if (rufptn_.isgood[ih]==0)
          {
            rusdgeom_.igsd[ic-1] = 0;
          }
      }

    // Use good signal information for counters
    rusdgeom_.nsds=ic;
    for (ic=0; ic<rusdgeom_.nsds; ic++)
      {
        rusdgeom_.pulsa[ic]=rusdgeom_.sdsigq[ic][0];
        rusdgeom_.sdtime[ic]=rusdgeom_.sdsigt[ic][0];
        rusdgeom_.sdterr[ic]=rusdgeom_.sdsigte[ic][0];
        rusdgeom_.sdirufptn[ic]=rusdgeom_.irufptn[ic][0];
        for (is=0; is<rusdgeom_.nsig[ic]; is++)
          {
            if (rusdgeom_.igsig[ic][is]>=4)
              {
                rusdgeom_.pulsa[ic]=rusdgeom_.sdsigq[ic][is];
                rusdgeom_.sdtime[ic]=rusdgeom_.sdsigt[ic][is];
                rusdgeom_.sdterr[ic]=rusdgeom_.sdsigte[ic][is];
                rusdgeom_.sdirufptn[ic] = rusdgeom_.irufptn[ic][is];
                break;
              }
          }
      }

    // Earliest time in the event readout
    rusdgeom_.tearliest = 0.5*(rufptn_.tearliest[0]+rufptn_.tearliest[1]);

    return true;
  }

// Puts together all the above routines to analyze the event.
bool rufptnAnalysis::analyzeEvent()
  {
    integer4 i, j;
    // array of hit times computed using upper and lower (average b/w the two)
    real8 xxyyt[RUFPTNMH];
    static integer4 ievent=0;
    if (opt.verbose >= 2)
      {
        fprintf(stdout,
        "*********************** EVENT %d ***************\n",
        rusdraw_.event_num);
      }
    ievent++;
    fHasTriggered = true;
    if (rusdraw_.nofwf < 1)

      {
        memset(&rufptn_, 0, sizeof(rufptn_dst_common));
        memset(&rusdgeom_, 0, sizeof(rusdgeom_dst_common));
        fHasTriggered = false;
	return true;
      }

    // Parse fadc traces over the event, obtain the hit information.
    processFADC();

    // Get the monitoring information for every hit.
    compEventMon();

    // Determine the pulse area sizes in VEM units
    findCharge();

    if (opt.verbose >= 2)
      {
        fprintf(stdout, "%s%9s%25s%25s\n", "hit", "XXYY", "PULSE AREA",
        "REL. TIME");
        for (i = 0; i < rufptn_.nhits; i++)
          {
            if (rufptn_.isgood[i] > 0)
              {
                fprintf(stdout, "%.02d%10.04d%15f%15f%15f%15f\n",
                i, rufptn_.xxyy[i], rufptn_.pulsa[i][0],
                rufptn_.pulsa[i][1], rufptn_.reltime[i][0],
                rufptn_.reltime[i][1]);
              }
          }

      }
    
    // if there are no waveforms for which we can form a pulse
    // (e.g. few SDs and not one of them with good pedestal values) then
    // zero out the result banks and return
    if (rufptn_.nhits < 1)
      {
	memset(&rufptn_, 0, sizeof(rufptn_dst_common));
	memset(&rusdgeom_, 0, sizeof(rusdgeom_dst_common));
	return true;
      }
    
    // Find space cluster
    spacePatRecog(rufptn_.nhits, rufptn_.xxyy, &nspaceclust, spaceclust);
    
    // Label hits that are in space cluster
    rufptn_.nsclust = nspaceclust;
    for (i=0; i < nspaceclust; i++)
      {
        j = spaceclust[i];
        if (rufptn_.isgood[j] == 1)
          rufptn_.isgood[j] = 2;
      }

    // Find space-time cluster
    for (i = 0; i < rufptn_.nhits; i++)
      xxyyt[i] = 0.5 * (rufptn_.reltime[i][0] + rufptn_.reltime[i][1]);
    timePatRecog(rufptn_.xxyy, rufptn_.xyzclf, xxyyt, nspaceclust, spaceclust,
        &nspacetimeclust, spacetimeclust);
    
    // Label hits that are in space-time cluster
    rufptn_.nstclust = nspacetimeclust;
    for (i=0; i < nspacetimeclust; i++)
      {
        j = spacetimeclust[i];
        if (rufptn_.isgood[j] == 2)
          rufptn_.isgood[j] = 3;
      }

    // Some signals need to be combined: if we have hits in the same counter,
    // then there is a correlation b/w the signal chosen by time pattern
    // recognition and the later signals (in the same counter).
    combineSignals();

    // Find SDs that pass time pattern recognition and that lie on the border.
    rufptn_.nborder = 0;
    for (i=0; i<nspacetimeclust; i++)
      {
        if (sdb.isOnBorder(rufptn_.xxyy[spacetimeclust[i]]))
          {
            rufptn_.nborder ++;
          }
      }

    // Simple plane fitting using 2nd moments of charge to find the event XY-axis
    tyroGeom();

    // Pass 1 geometry fit
    geomFit();

    // Change space-time cluster, so that it contains hits which passed
    // chi2-cleaning procedure in geometrical fitting.
    changeStclust();

    // Go over all hits and label the ones that saturate the counter
    labelSaturatingHits();

    // Put good counters into geometry dst bank
    put2rusdgeom();

    return true;
  }

void rufptnAnalysis::comp_rusdmc1()
  {
    double xcore, ycore;
    double bdist, v[2];
    double tdistbr, vbr[2];
    double tdistlr, vlr[2];
    double tdistsk, vsk[2];
    double tdist;

    xcore = ((double) rusdmc_.corexyz[0]) / 1.2e5 - SD_ORIGIN_X_CLF;
    ycore = ((double) rusdmc_.corexyz[1]) / 1.2e5 - SD_ORIGIN_Y_CLF;

    sdbdist(xcore, ycore, &v[0], &bdist, &vbr[0], &tdistbr, &vlr[0], &tdistlr,
        &vsk[0], &tdistsk);

    // Pick out the actual T-shape boundary distance for whatever subarray
    tdist = tdistbr;
    if (tdistlr > tdist)
      tdist = tdistlr;
    if (tdistsk > tdist)
      tdist = tdistsk;
    rusdmc1_.xcore = xcore;
    rusdmc1_.ycore = ycore;

    // Time of the core hit with respect to the
    // earliest time in the event readout, [uS]:
    rusdmc1_.t0 = (((double) rusdmc_.tc) / 50.0 - 1.0e6 * (rusdgeom_.tearliest
        - floor(rusdgeom_.tearliest)));

    // Converting the relative time of the core hit into [1200m] units:
    rusdmc1_.t0 *= RUFPTN_TIMDIST;

    // Saving the border distances
    rusdmc1_.bdist = bdist;
    rusdmc1_.tdistbr = tdistbr;
    rusdmc1_.tdistlr = tdistlr;
    rusdmc1_.tdistsk = tdistsk;
    rusdmc1_.tdist = tdist;
  }

void rufptnAnalysis::comp_rusdmc1_no_tref()
{
  rusdgeom_.tearliest = 0;
  comp_rusdmc1();
}

bool rufptnAnalysis::printBadSDInfo(int iwf, char *problem_description, FILE *fp)
{
  static bool first_call = true;
  
  if (iwf < 0 || iwf > rusdraw_.nofwf)
    {
      fprintf (stderr, "error: printBadSDInfo: iwf must be in 0 - %d range!\n",
	       rusdraw_.nofwf);
      return false;
    }
  
  if(!fp) return true;

  if (first_call)
    {
      fprintf(fp,"xxyy iwf date time lhpchmip(lower,upper) pchmip rhpchmip ");
      fprintf(fp,"lhpchped pchped rhpchped mftndof mip mftchi2 problem");
      fprintf(fp,"\n");
      first_call = false;
    }

  fprintf(fp,"%04d %03d %06d %06d.%06d %03d,%03d %03d,%03d %03d,%03d ",
	  rusdraw_.xxyy[iwf],iwf,rusdraw_.yymmdd, rusdraw_.hhmmss,rusdraw_.usec,
	  rusdraw_.lhpchmip[iwf][0],rusdraw_.lhpchmip[iwf][1],
	  rusdraw_.pchmip[iwf][0],rusdraw_.pchmip[iwf][1],
	  rusdraw_.rhpchmip[iwf][0],rusdraw_.rhpchmip[iwf][1]);
  fprintf (fp,"%03d,%03d %03d,%03d %03d,%03d %03d,%03d %7.1f,%7.1f %7.1f,%7.1f %s\n",
	   rusdraw_.lhpchped[iwf][0],rusdraw_.lhpchped[iwf][1],
	   rusdraw_.pchped[iwf][0],rusdraw_.pchped[iwf][1],
	   rusdraw_.rhpchped[iwf][0],rusdraw_.rhpchped[iwf][1],
	   rusdraw_.mftndof[iwf][0],rusdraw_.mftndof[iwf][1],
	   rusdraw_.mip[iwf][0],rusdraw_.mip[iwf][1],
	   rusdraw_.mftchi2[iwf][0],rusdraw_.mftchi2[iwf][1],
	   problem_description
	   );
  
  return true;
}
