#include "sdinfo_class.h"

static bool chk_xxyy(int xxyy)
{
  int x, y;
  x = xxyy / 100;
  y = xxyy % 100;
  if (x < 1 || x > SDMON_X_MAX)
    return false;
  if (y < 1 || y > SDMON_Y_MAX)
    return false;
  return true;
}

sdinfo_class::sdinfo_class()
{
  Clean();
}
sdinfo_class::~sdinfo_class()
{
  Clean();
}

void sdinfo_class::Clean()
{
  sdindex = 0;
  raw_bankid = 0;
  xxyy = 0;
  tlim[0] = 1.0e20;
  tlim[1] = -1.0e20;
  d_ped = 0;
  nl1 = 0;
  il2sig = 0;
}

bool sdinfo_class::init_sd(tasdevent_dst_common *p, int isd, int iwf)
  {
    Clean();
    if (iwf >= p->num_trgwf)
      {
        printWarn("init_sd: iwf = %d >= numTrgWf = %d", iwf, p->num_trgwf);
        return false;
      }
    xxyy = p->sub[iwf].lid;
    if (!chk_xxyy(xxyy))
      {
        printWarn("init_sd: invalid XXYY: iwf = %d", iwf);
        Clean();
        return false;
      }
    sdindex = isd;
    raw_bankid = TASDEVENT_BANKID;
    wfindex_cal = iwf;
    nwf = 0;

    // Pedestal information (rounded to the nearest integer)
    ped[0] = p->sub[iwf].lavr; // in 8 FADC time slices
    ped[1] = p->sub[iwf].uavr;

    ped1[0] = p->sub[iwf].lavr * 16; // in 128 FADC time slices
    ped1[1] = p->sub[iwf].uavr * 16;

    // Mip information (tasdevent bank doesn't have it)
    mip[0] = 40.0;
    mip[1] = 40.0;
    return true;
  }

bool sdinfo_class::init_sd(tasdcalibev_dst_common *p, int isd, int iwf)
{
  Clean();
  if (iwf >= p->numTrgwf)
    {
      printWarn("init_sd: iwf = %d >= numTrgWf = %d", iwf, p->numTrgwf);
      return false;
    }
  xxyy = p->sub[iwf].lid;
  if (!chk_xxyy(xxyy))
    {
      printWarn("init_sd: invalid XXYY: iwf = %d", iwf);
      Clean();
      return false;
    }
  sdindex = isd;
  raw_bankid = TASDCALIBEV_BANKID;
  wfindex_cal = iwf;
  nwf = 0;

  // Pedestal information (rounded to the nearest integer)
  ped[0] = (int) floor(p->sub[iwf].lpedAvr * 8.0 + 0.5); // in 8 FADC time slices
  ped[1] = (int) floor(p->sub[iwf].upedAvr * 8.0 + 0.5);

  ped1[0] = (int) floor(p->sub[iwf].lpedAvr * 128.0 + 0.5); // in 128 FADC time slices
  ped1[1] = (int) floor(p->sub[iwf].upedAvr * 128.0 + 0.5);

  // Mip information
  mip[0] = p->sub[iwf].mip[0];
  mip[1] = p->sub[iwf].mip[1];
  return true;
}

bool sdinfo_class::init_sd(rusdraw_dst_common *p, int isd, int iwf)
{
  int k;
  Clean();
  if (iwf >= p->nofwf)
    {
      printWarn("init_sd: iwf = %d >= nofwf = %d", iwf, p->nofwf);
      return false;
    }
  xxyy = p->xxyy[iwf];
  if (!chk_xxyy(xxyy))
    {
      printWarn("init_sd: invalid XXYY: %04d, iwf = %d", xxyy, iwf);
      Clean();
      return false;
    }
  sdindex = isd;
  raw_bankid = RUSDRAW_BANKID;
  wfindex_cal = iwf;
  nwf = 0;
  for (k = 0; k < 2; k++)
    {
      ped[k] = p->pchped[iwf][k]; // Pedestal information ( in 8 FADC time slices)
      ped1[k] = 16 * p->pchped[iwf][k]; // Pedestal in 128 FADC time slices
      mip[k] = p->mip[iwf][k]; // Mip information
    }
  return true;
}

bool sdinfo_class::add_wf(tasdevent_dst_common *p, int iwf)
{
  int j;
  double t;
  if (nwf >= NWFPSD)
    {
      printWarn("number of waveforms exceeds the maximum: %d", NWFPSD);
      return false;
    }

  if (xxyy == 0 || raw_bankid == 0)
    {
      printWarn("was not properly initialized before adding waveforms");
      return false;
    }
  if (iwf >= p->num_trgwf)
    {
      printWarn("add_wf: iwf = %d >= numTrgWf = %d", iwf, p->num_trgwf);
      return false;
    }

  // clock count information
  clkcnt[nwf] = p->sub[iwf].clock;
  mclkcnt[nwf] = p->sub[iwf].max_clock;

  // save the waveform index
  wfindex[nwf] = iwf;

  // FADC trace
  for (j = 0; j < rusdraw_nchan_sd; j++)
    {
      fadc[nwf][0][j] = p->sub[iwf].lwf[j];
      fadc[nwf][1][j] = p->sub[iwf].uwf[j];
    }

  // calculate the earliest and latest possible times for this SD
  // calculate the earliest and latest possible times for this SD
  t = ((double) clkcnt[nwf]) / ((double) mclkcnt[nwf]);
  if (t < tlim[0])
    tlim[0] = t;
  t = ((double) clkcnt[nwf]) / ((double) mclkcnt[nwf]) + 2.56e-6;
  if (t > tlim[1])
    tlim[1] = t;

  nwf++;
  return true;
}
bool sdinfo_class::add_wf(tasdcalibev_dst_common *p, int iwf)
{
  int j;
  double t;
  if (nwf >= NWFPSD)
    {
      printWarn("number of waveforms exceeds the maximum: %d", NWFPSD);
      return false;
    }

  if (xxyy == 0 || raw_bankid == 0)
    {
      printWarn("was not properly initialized before adding waveforms");
      return false;
    }
  if (iwf >= p->numTrgwf)
    {
      printWarn("add_wf: iwf = %d >= numTrgWf = %d", iwf, p->numTrgwf);
      return false;
    }

  // clock count information
  clkcnt[nwf] = p->sub[iwf].clock;
  mclkcnt[nwf] = p->sub[iwf].maxClock;

  // save the waveform index
  wfindex[nwf] = iwf;

  // FADC trace
  for (j = 0; j < rusdraw_nchan_sd; j++)
    {
      fadc[nwf][0][j] = p->sub[iwf].lwf[j];
      fadc[nwf][1][j] = p->sub[iwf].uwf[j];
    }

  // calculate the earliest and latest possible times for this SD
  // calculate the earliest and latest possible times for this SD
  t = ((double) clkcnt[nwf]) / ((double) mclkcnt[nwf]);
  if (t < tlim[0])
    tlim[0] = t;
  t = ((double) clkcnt[nwf]) / ((double) mclkcnt[nwf]) + 2.56e-6;
  if (t > tlim[1])
    tlim[1] = t;

  nwf++;
  return true;
}
bool sdinfo_class::add_wf(rusdraw_dst_common *p, int iwf)
{
  int j, k;
  double t;
  if (nwf >= NWFPSD)
    {
      printWarn("number of waveforms exceeds the maximum: %d", NWFPSD);
      return false;
    }

  if (xxyy == 0 || raw_bankid == 0)
    {
      printWarn("was not properly initialized before adding waveforms (iwf = %d)", iwf);
      return false;
    }
  if (iwf >= p->nofwf)
    {
      printWarn("add_wf: iwf = %d >= numTrgWf = %d", iwf, p->nofwf);
      return false;
    }

  // clock count information
  clkcnt[nwf] = p->clkcnt[iwf];
  mclkcnt[nwf] = p->mclkcnt[iwf];

  // save the waveform index
  wfindex[nwf] = iwf;

  // FADC trace
  for (j = 0; j < rusdraw_nchan_sd; j++)
    {
      for (k = 0; k < 2; k++)
        fadc[nwf][k][j] = p->fadc[iwf][k][j];
    }

  // calculate the earliest and latest possible times for this SD
  t = ((double) clkcnt[nwf]) / ((double) mclkcnt[nwf]);
  if (t < tlim[0])
    tlim[0] = t;
  t = ((double) clkcnt[nwf]) / ((double) mclkcnt[nwf]) + 2.56e-6;
  if (t > tlim[1])
    tlim[1] = t;

  nwf++;
  return true;
}

int sdinfo_class::find_l1_sig(int DeltaPed)
{
  int iwf, j, k, l;
  int sum[2];
  int l1trig;
  d_ped = DeltaPed;
  nl1 = 0;
  il2sig = 0;
  // go over all waveforms in a given sd
  for (iwf = 0; iwf < nwf; iwf++)
    {
      // going over all fadc time slices
      for (j = 0; j <= (128 - SLWsize); j++)
        {
          l1trig = 1;
          // going over upper and lower
          for (k = 0; k < 2; k++)
            {
              // find signals in the sliding window
              sum[k] = 0;
              for (l = 0; l < SLWsize; l++)
                sum[k] += fadc[iwf][k][j + l];
              sum[k] -= ped[k]; // subtract the pedestal

              /////////////////////////////////////////////////////
              // ALTERING THE PEDESTALS, IF d_ped IS NOT ZERO
              // if d_ped < 0, pedestals are LOWERED (counts above pedestal increased)
              // if d_ped > 0, pedestals are RAISED (counts above pedestal decreased)
              /////////////////////////////////////////////////////
              sum[k] -= d_ped;

              // level-1 trigger occurs if both upper and lower layer
              // have at least NL1cnt FADC counts above the pedestal inside
              // the sliding window
              l1trig *= (int) (sum[k] >= NL1cnt);
            }
          // if level-1 trigger is found, add it to the table for the given SD
          if (l1trig)
            {
              // preventing the buffer overflow for level-1 signals
              if (nl1 >= NSIGPSD)
                {
                  printWarn("Number of level-1 signals exceeds the maximum: ", NSIGPSD);
                  return nl1;
                }
              secf[nl1] = ((double) clkcnt[iwf]) / ((double) mclkcnt[iwf]) + ((double) j) * 2.0e-8;
              ich[nl1] = j;
              q[nl1][0] = sum[0];
              q[nl1][1] = sum[1];
              iwfsd[nl1] = iwf;
              nl1++;
            }
        }
    }
  return nl1;
}

int sdinfo_class::find_l1_sig_correct(int DeltaPed)
{
  int iwf, j, k;
  int sum[2];
  d_ped = DeltaPed;
  nl1 = 0;
  il2sig = 0;
  // go over all waveforms in a given sd
  for (iwf = 0; iwf < nwf; iwf++)
    {
      // going over lower and upper layers
      for (k = 0; k < 2; k++)
        {
          // Initializing the sum inside the whole waveform
          // as pedestal summed over the ENTIRE waveform
          // ped1[k] is the pedestal in 128 fadc time slices
          sum[k] = -ped1[k];
          /////////////////////////////////////////////////////
          // ALTERING THE PEDESTALS, IF d_ped IS NOT ZERO
          // if d_ped < 0, pedestals are LOWERED (counts above pedestal increased)
          // if d_ped > 0, pedestals are RAISED (counts above pedestal decreased)
          /////////////////////////////////////////////////////
          sum[k] -= d_ped;
	  // Find the fadc sums above the pedestal
          for (j = 0; j < rusdraw_nchan_sd; j++)
            sum[k] += fadc[iwf][k][j];
        }
      // if level-1 trigger is found, add it to the table for the given SD
      if ((sum[0] >= NL1cnt) && (sum[1] >= NL1cnt))
        {
          // preventing the buffer overflow for level-1 signals
          if (nl1 >= NSIGPSD)
            {
              printWarn("Number of level-1 signals exceeds the maximum: ", NSIGPSD);
              return nl1;
            }
          secf[nl1] = ((double) clkcnt[iwf]) / ((double) mclkcnt[iwf]) + ((double) TRIGT_FADCTSLICE) * 2.0e-8;
          ich[nl1] = TRIGT_FADCTSLICE;
          q[nl1][0] = sum[0];
          q[nl1][1] = sum[1];
          iwfsd[nl1] = iwf;
          nl1++;
        }
    }
  return nl1;
}

void sdinfo_class::printWarn(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "WARNING (sdinfo_class): xxyy=%04d: %s\n", xxyy, mess);
}
