/*
 * mc2tasdcalibev_class.cpp
 *
 *  Dmitri Ivanov <ivanov@physics.rutgers.edu>
 */

#include "mc2tasdcalibev_class.h"
#include "sduti.h"

// these definitions are used for labeling the
// saturated waveforms in the way Okuda's code does it.
#define usatLedNbit  0
#define lsatLedNbit  1
#define usatMipNbit  2
#define lsatMipNbit  3
#define usatFadcNbit 4
#define lsatFadcNbit 5
#define daqErrorNbit 6


using namespace TMath;

/// calibration index handler
calibInd_class::calibInd_class()
{

}
calibInd_class::~calibInd_class()
{

}
void calibInd_class::clear()
{
  int ix, iy;
  for (ix = 0; ix < 30; ix++)
    {
      for (iy = 0; iy < 30; iy++)
        calibInd[ix][iy] = -1;
    }
}
int calibInd_class::getInd(int xxyy)
{
  int ix, iy;
  ix = xxyy / 100 - 1;
  iy = xxyy % 100 - 1;
  if (ix < 0 || ix >= 30 || iy < 0 || iy >= 30)
    return -1;
  return calibInd[ix][iy];
}
bool calibInd_class::putInd(int ind, int xxyy)
{
  int ix, iy;
  ix = xxyy / 100 - 1;
  iy = xxyy % 100 - 1;
  if (ix < 0 || ix >= 30 || iy < 0 || iy >= 30)
    return false;
  calibInd[ix][iy] = ind;
  return true;
}

mc2tasdcalibev_class::mc2tasdcalibev_class()
{
  icrr = &tasdcalibev_;
  ru = &rusdraw_;
  rumc = &rusdmc_;
  yymmdd_calib = -1; // initialize calibration date and time
  hhmmss_calib = -1;
}

mc2tasdcalibev_class::~mc2tasdcalibev_class()
{

}

bool mc2tasdcalibev_class::Convert()
{

  // first, zero out the entire icrr banks
  memset(icrr, 0, sizeof(tasdcalibev_dst_common));
  convThrown(); // convert the thrown MC information
  if (!findCalib())
    {
      printErr("error: no calibration for %06d %06d.%06d", ru->yymmdd, ru->hhmmss, ru->usec);
      return false;
    }
  convWf(); // convert the waveform information
  // the rest of the information is not relevant if the event did not trigger
  if (icrr->numTrgwf == 0)
    return true;
  fillDeadAlive(); // fill information about the dead and alive counters
  fillWeather(); // fill the weather information
  return true;
}

//

void mc2tasdcalibev_class::convThrown()
{

  double phi;

  // QGSJET-II protons used by default
  sprintf(icrr->sim.interactionModel, "qgsjet2");
  sprintf(icrr->sim.primaryParticleType, "p");

  // ICRR MC energy in eV
  icrr->sim.primaryEnergy = (float) (1.0e18 * rumc->energy);

  // cosine of zenith angle
  icrr->sim.primaryCosZenith = (float) Cos(rumc->theta);

  // ICRR azimuthal angle points back to the source and is in degrees; rusdmc phi is along
  // the direction of the primary momentum and is in radians
  phi = RadToDeg() * (rumc->phi + Pi());
  while (phi > 360.0)
    phi -= 360.0;
  icrr->sim.primaryAzimuth = (float) phi;

  // first interaction (vertical) depth, g/cm^2.  using us76 model. rusdmc stores
  // the first interaction height in cm.  Here we calculate the mass overburden of 
  // the 1st interaction point in g/mc^2 by calling a CORSIKA atmospheric model 22 
  // (in C 22 in Fortran 23) 
  icrr->sim.primaryFirstIntDepth = (float)SDGEN::h2mo((double)rumc->height,22);
  
  // arrival time for PPS is a second fraction. rusdmc stores the core time in clock counts from PPS.
  icrr->sim.primaryArrivalTimeFromPps = (float) ((double) rumc->tc / 5e7);
  
  // rusdmc stores core in CLF frame in cm, ICRR stores core in CLF frame in meters.
  icrr->sim.primaryCorePosX = rumc->corexyz[0] / 100.0;
  icrr->sim.primaryCorePosY = rumc->corexyz[1] / 100.0;
  icrr->sim.primaryCorePosZ = rumc->corexyz[2] / 100.0;

  // Ben's MC starts with 1e-7 thinning
  icrr->sim.thinRatio = (float) (1e7);

  // Then it is de-weighted ( all particles have weight 1)
  icrr->sim.maxWeight = (float) 1.0;
  icrr->sim.trgCode = (int)(ru->nofwf > 0); // RECENTLY LEARNED THAT HAS TO BE NON-ZERO IF THE EVENT TRIGGERS
  icrr->sim.userInfo = 4346963; // MC SIGNATURE ... 

  // enough information already
  memset(icrr->sim.detailUserInfo, 0, 10 * sizeof(float));

}

bool mc2tasdcalibev_class::findCalib()
{
  int hhmmss_event = ru->hhmmss;
  int min_calib0; // calibration minute, since midnight for the calibration cycle in which the event occurred
  int min_calib; // trial calibration minute
  int min_event; // event minute, since midnight
  int delta; // calibration minute increment
  // event minute since midnight
  min_event = (hhmmss_event / 10000) * 60 + ((hhmmss_event % 10000) / 100);
  min_calib0 = 10 * (min_event / 10);
  fflush(stdout);
  // finding the calibration cycle in which the event occurred or the closest one
  delta = 0;
  do
    {
      min_calib = min_calib0 - delta;
      hhmmss_calib = (min_calib / 60) * 10000 + (min_calib % 60) * 100;
      if (calib.find(hhmmss_calib) != calib.end())
        return true;
      min_calib = min_calib0 + delta;
      hhmmss_calib = (min_calib / 60) * 10000 + (min_calib % 60) * 100;
      if (calib.find(hhmmss_calib) != calib.end())
        return true;
      delta += 10;
      // failed to find the calibration cycle ...
      if ((min_calib0 - delta) < 0 && (min_calib0 + delta) > 1440)
        {
          hhmmss_calib = -1;
          return false;
        }
    }
  while (true);

}

void mc2tasdcalibev_class::convWf()
{
  int i, itower, iwf, jwf, ical, ila;
  SDCalibSubData *det; // detector calibration data
  SDCalibevData *wf; // waveform information

  tasdconst_dst_common *sdc; // constants for each SD
  tasdcalib_dst_common *cal; // calibration cycle used
  calibInd_class *ind; // calibration indexing for each SD

  // pick the correct calibration cycle, calibration indexing
  cal = &calib[hhmmss_calib];
  ind = &calibInd[hhmmss_calib];

  icrr->eventCode = ru->event_code;
  icrr->date = ru->yymmdd;
  icrr->time = ru->hhmmss;
  icrr->usec = ru->usec;
  icrr->trgBank = 0;
  icrr->trgPos = (ru->nofwf > 0 ? rumc->corecounter : 0);

  icrr->trgMode = 0; // INITIALIZE HERE, FILL LATER
  icrr->daqMode = 0; // INITIALIZE HERE, FILL LATER

  icrr->numWf = ru->nofwf;
  icrr->numTrgwf = ru->nofwf;
  icrr->numWeather = 0; // INITIALIZE HERE, FILL LATER
  icrr->numAlive = 0;   // INITIALIZE HERE, FILL LATER
  icrr->numDead = 0;    // INITIALIZE HERE, FILL LATER

  for (itower = 0; itower < 3; itower++)
    {
      icrr->runId[itower] = ru->run_id[itower];
      icrr->daqMiss[itower] = 0;
      for (i = 0; i < 600; i++)
        {
          if (cal->host[itower].miss[i])
            icrr->daqMiss[itower]++;
        }
    }

  for (iwf = 0; iwf < ru->nofwf; iwf++)
    {
      wf = &icrr->sub[iwf]; // waveform in the particular detector
      if ((ical = ind->getInd(ru->xxyy[iwf])) == -1)
        {
          fprintf(stderr, "warning: %06d %06d.%06d %04d: no calibration\n", ru->yymmdd, ru->hhmmss, ru->usec,
		  ru->xxyy[iwf]);
          wf->lid = ru->xxyy[iwf];
          wf->dontUse = 0xff;
          continue;
        }
      det = &cal->sub[ical]; // calibration for the particular detector
      if (det->site == 0)
        icrr->daqMode |= 1;
      if (det->site == 1)
        icrr->daqMode |= 2;
      if (det->site == 2)
        icrr->daqMode |= 4;
      wf->site = det->site;
      wf->lid = ru->xxyy[iwf];
      wf->clock = ru->clkcnt[iwf];
      wf->maxClock = ru->mclkcnt[iwf];
      wf->wfId = ru->wf_id[iwf];
      wf->numTrgwf = 0;     // count the number of joining waveforms
      for (jwf = 0; jwf < ru->nofwf; jwf++)
	{
	  if (ru->xxyy[iwf] == ru->xxyy[jwf])
	    wf->numTrgwf ++;
	}
      wf->trgCode  = 2; // NO DESCRIPTION, THIS SEEMS TO BE THE VALUE USED IN ICRR MC
      wf->wfError  = 0; // INITIALIZE HERE, FILL LATER
      for (i = 0; i < tasdcalibev_nfadc; i++)
        {
          wf->lwf[i] = ru->fadc[iwf][0][i];
          wf->uwf[i] = ru->fadc[iwf][1][i];
        }
      wf->clockError = det->clockError;
      wf->upedAvr = det->upedAvr;
      wf->lpedAvr = det->lpedAvr;
      wf->upedStdev = det->upedStdev;
      wf->lpedStdev = det->lpedStdev;
      wf->umipNonuni = det->umipNonuni;
      wf->lmipNonuni = det->lmipNonuni;
      wf->umipMev2cnt = det->umipMev2cnt;
      wf->lmipMev2cnt = det->lmipMev2cnt;
      wf->umipMev2pe = det->umipMev2pe;
      wf->lmipMev2pe = det->lmipMev2pe;
      wf->lvl0Rate = det->lvl0Rate;
      wf->lvl1Rate = det->lvl1Rate;
      wf->scintiTemp = det->scinti_temp;
      wf->warning = det->warning;
      wf->dontUse = det->dontUse;
      wf->dataQuality = det->dataQuality;
      wf->trgMode0 = 3;
      wf->trgMode1 = 7;
      wf->gpsRunMode = det->gpsRunMode;
      wf->uthreLvl0 = 15;
      wf->lthreLvl0 = 15;
      wf->uthreLvl1 = 150;
      wf->lthreLvl1 = 150;

      if (cnst.find(wf->lid) == cnst.end())
        fprintf(stderr, "warning: %06d %06d.%06d %04d constants information missing\n", ru->yymmdd, ru->hhmmss,
		ru->usec, ru->xxyy[iwf]);
      else
        {
          sdc = &cnst[wf->lid];
          wf->posX = sdc->posX;
          wf->posY = sdc->posY;
          wf->posZ = sdc->posZ;
          wf->delayns = sdc->delayns;
          wf->ppsofs = sdc->ppsofs;
          wf->ppsflu = sdc->ppsflu3D;
          wf->lonmas = sdc->lonmas;
          wf->latmas = sdc->latmas;
          wf->heicm = sdc->heicm;
          wf->udec5pled = sdc->udec5pled;
          wf->ldec5pled = sdc->ldec5pled;
          wf->udec5pmip = sdc->udec5pmip;
          wf->ldec5pmip = sdc->ldec5pmip;
        }
      for (ila = 0; ila < 2; ila++)
        {
          wf->pchmip[ila] = det->pchmip[ila];
          wf->pchped[ila] = det->pchped[ila];
          wf->lhpchmip[ila] = det->lhpchmip[ila];
          wf->rhpchmip[ila] = det->rhpchmip[ila];
          wf->lhpchped[ila] = det->lhpchped[ila];
          wf->rhpchped[ila] = det->rhpchped[ila];
          wf->mftndof[ila] = det->mftndof[ila];
          wf->mip[ila] = det->mip[ila];
          wf->mftchi2[ila] = det->mftchi2[ila];
          for (i = 0; i < 4; i++)
            {
              wf->mftp[ila][i] = det->mftp[ila][i];
              wf->mftpe[ila][i] = det->mftpe[ila][i];
            }
        }

    }

  icrr->trgMode = icrr->daqMode;


  for(iwf=0;iwf<icrr->numTrgwf;iwf++)
    {
      // if there has been a daq error wfError will be non-zero at this point.
      // daq errors are to be implemented later if its necessary. 
      // if(icrr->sub[iwf].wfError)
      // 	icrr->sub[iwf].wfError=(1<<daqErrorNbit);
      icrr->sub[iwf].wfError = 0; // make sure this is initialized
      
      for(i=0;i<tasdcalibev_nfadc;i++)
	{
	  //      if(icrr->sub[iwf].uwf[iwf]>icrr->sub[iwf].udec5pled){//okuda rem
	  if(icrr->sub[iwf].uwf[i]>icrr->sub[iwf].udec5pled)
	    icrr->sub[iwf].wfError|=(1<<usatLedNbit);
	  
	  //      if(icrr->sub[iwf].uwf[iwf]>icrr->sub[iwf].udec5pmip){//okuda rem
	  if(icrr->sub[iwf].uwf[i]>icrr->sub[iwf].udec5pmip)
	    icrr->sub[iwf].wfError|=(1<<usatMipNbit);
	  
	  //      if(icrr->sub[iwf].uwf[iwf]>=4095){//okuda rem
	  if(icrr->sub[iwf].uwf[i]>=4095)
	    icrr->sub[iwf].wfError|=(1<<usatFadcNbit);

	  //      if(icrr->sub[iwf].lwf[iwf]>icrr->sub[iwf].ldec5pled){//okuda rem
	  if(icrr->sub[iwf].lwf[i]>icrr->sub[iwf].ldec5pled)
	    icrr->sub[iwf].wfError|=(1<<lsatLedNbit);
	  
	  //      if(icrr->sub[iwf].lwf[iwf]>icrr->sub[iwf].ldec5pmip){//okuda rem
	  if(icrr->sub[iwf].lwf[i]>icrr->sub[iwf].ldec5pmip)
	    icrr->sub[iwf].wfError|=(1<<lsatMipNbit);
	  
	  //      if(icrr->sub[iwf].lwf[iwf]>=4095){//okuda rem
	  if(icrr->sub[iwf].lwf[i]>=4095)
	    icrr->sub[iwf].wfError|=(1<<lsatFadcNbit);
	}
    }
  
}

void mc2tasdcalibev_class::fillDeadAlive()
{
  int lid, ical;
  bool isAlive;
  tasdcalib_dst_common *cal; // calibration cycle used
  calibInd_class *ind; // calibration indexing for each SD
  SDCalibSubData *det; // detector calibration data
  tasdconst_dst_common *sdc; // constants for each SD
  cal = &calib[hhmmss_calib];
  ind = &calibInd[hhmmss_calib];
  map<int, tasdconst_dst_common>::iterator isd;
  icrr->numAlive = 0;
  icrr->numDead = 0;
  for (isd = cnst.begin(); isd != cnst.end(); ++isd)
    {
      lid = (*isd).first;
      sdc = &((*isd).second);
      isAlive = true;
      if ((ical = ind->getInd(lid)) == -1)
        isAlive = false;
      if (isAlive)
        {
          det = &cal->sub[ical];
          if (det->lvl0Rate < 1.0 && det->lvl1Rate < 1.0 && det->livetime < 1)
            isAlive = false;
          if (det->dontUse)
            isAlive = false;
        }
      if (isAlive)
        {
          if (is_alive_det_relevant(lid))
            {
              icrr->aliveDetLid[icrr->numAlive] = sdc->lid;
              icrr->aliveDetSite[icrr->numAlive] = sdc->site;
              icrr->aliveDetPosX[icrr->numAlive] = sdc->posX;
              icrr->aliveDetPosY[icrr->numAlive] = sdc->posY;
              icrr->aliveDetPosZ[icrr->numAlive] = sdc->posZ;
              icrr->numAlive++;
            }
        }
      else
        {
          icrr->deadDetLid[icrr->numDead] = sdc->lid;
          icrr->deadDetSite[icrr->numDead] = sdc->site;
          icrr->deadDetPosX[icrr->numDead] = sdc->posX;
          icrr->deadDetPosY[icrr->numDead] = sdc->posY;
          icrr->deadDetPosZ[icrr->numDead] = sdc->posZ;
          icrr->numDead++;
        }
    }
}

void mc2tasdcalibev_class::fillWeather()
{

  int iwea;
  SDCalibWeatherData *cwea; // calibration weather data
  SDCalibevWeatherData *ewea; // event weather information
  tasdcalib_dst_common *cal = &calib[hhmmss_calib];
  icrr->numWeather = cal->num_weather;
  for (iwea = 0; iwea < cal->num_weather; iwea++)
    {
      cwea = &cal->weather[iwea];
      ewea = &icrr->weather[iwea];
      ewea->site = cwea->site;
      ewea->atmosphericPressure = cwea->atmosphericPressure;
      ewea->temperature = cwea->temperature;
      ewea->humidity = cwea->humidity;
      ewea->rainfall = cwea->rainfall;
      ewea->numberOfHails = cwea->numberOfHails;
    }
}

void mc2tasdcalibev_class::add_cnst_info(tasdconst_dst_common *x)
{
  memcpy(&cnst[tasdconst_.lid], x, sizeof(tasdconst_dst_common));
}
bool mc2tasdcalibev_class::add_calib_info(tasdcalib_dst_common *x)
{
  int isd;
  if (yymmdd_calib == -1)
    yymmdd_calib = x->date;
  else
    {
      if (x->date != yymmdd_calib)
        {
          printErr("fatal error: calibration date %06d not equal to that of the 1st calibration cycle (%06d)", x->date,
		   yymmdd_calib);
          return false;
        }
    }
  memcpy(&calib[tasdcalib_.time], x, sizeof(tasdcalib_dst_common));
  calibInd[x->time].clear();
  for (isd = 0; isd < x->num_det; isd++)
    calibInd[x->time].putInd(isd, x->sub[isd].lid);
  return true;
}

double mc2tasdcalibev_class::get_xdepth(double h)
{
  double pres, xdepth, tcent, rho;
  SDGEN::stdz76(h, &pres, &xdepth, &tcent, &rho);
  return xdepth;
}

bool mc2tasdcalibev_class::is_alive_det_relevant(int xxyy)
{
  double xx0, yy0, xx, yy;
  xx0 = (double) (icrr->trgPos / 100);
  yy0 = (double) (icrr->trgPos % 100);
  xx = (double) (xxyy / 100);
  yy = (double) (xxyy % 100);
  return (1.2 * sqrt((xx - xx0) * (xx - xx0) + (yy - yy0) * (yy - yy0)) < alivedet_radius);

}
void mc2tasdcalibev_class::printErr(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "mc2tasdcalibev_class: %s\n", mess);
}
