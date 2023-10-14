#include "icrr2ru.h"
#include "sduti.h"
#include "TMath.h"

using namespace TMath;

icrr2ru::icrr2ru()
{
  // set pointers to corresponding dst banks
  icrr = &tasdcalibev_;
  ru = &rusdraw_;
  rumc = &rusdmc_;
  reset_event_num();
}

icrr2ru::~icrr2ru()
{
}

void icrr2ru::reset_event_num()
{
  event_num = 0;
}

bool icrr2ru::Convert()
{
  int itower, iwf, ii, jj;
  int secnum, ilayer, ipar;
  int towerflag; // label towers participating in event reconstruction
  bool eflag;

  // event date and time
  ru->yymmdd = icrr->date;
  ru->hhmmss = icrr->time;
  ru->usec = icrr->usec;

  // Just the event number in the DST file
  ru->event_num = event_num;
  ru->event_code = icrr->eventCode;

  ru->errcode = 0; // a simple measure on how severe the event problems are

  for (itower = 0; itower < 3; itower++)
    {
      ru->run_id[itower] = icrr->runId[itower];
      ru->trig_id[itower] = event_num; // not essential right now
      if (icrr->daqMiss[itower] != 0)
        {
          if ((icrr->daqMiss[itower] & 16) || (icrr->daqMiss[itower] & 8))
            ru->errcode += 5;
          if ((icrr->daqMiss[itower] & 32))
            ru->errcode += 10;
        }
    }

  // a simple guess to what monitoring cycle time was used for this event calibration.
  // not important for the analysis programs, good for consistency
  ru->monyymmdd = icrr->date;
  secnum = (3600 * (icrr->time / 10000) + 60 * ((icrr->time % 10000) / 100) + (icrr->time % 100));
  secnum -= secnum % 600;
  ru->monhhmmss = 10000 * (secnum / 3600) + 100 * ((secnum % 3600) / 60) + (secnum % 60);

  // Loop over all the waveforms while saving the detector information

  iwf = 0; // waveform index inside rusdraw

  towerflag = 0; // to determine what towers reported waveforms for this event

  for (jj = 0; jj < icrr->numTrgwf; jj++)
    {
      // needed for determining which towers reported waveforms for this event
      if (icrr->sub[jj].site == 0)
        towerflag |= 1; // BR
      if (icrr->sub[jj].site == 1)
        towerflag |= 2; // LR
      if (icrr->sub[jj].site == 2)
        towerflag |= 4; // SK
      
      // make sure that can add waveforms
      if (iwf >= RUSDRAWMWF)
        {
          fprintf(stderr,"warning: date=%06d time=%06d.%06d: too many waveforms, maximum is %d\n", 
		  ru->yymmdd, ru->hhmmss, ru->usec, RUSDRAWMWF);
          ru->errcode += 10000;
          break;
        }

      // this flag is set to true if there is something wrong with
      // the waveform information
      eflag = false;

      ru->nretry[iwf] = 0; // not essential
      ru->wf_id[iwf] = (int) icrr->sub[jj].wfId;
      ru->xxyy[iwf] = (int) icrr->sub[jj].lid;
      ru->clkcnt[iwf] = (int) icrr->sub[jj].clock;
      ru->mclkcnt[iwf] = icrr->sub[jj].maxClock;

      // fill the fadc traces
      ru->fadcti[iwf][0] = 0; // lower
      ru->fadcti[iwf][1] = 0; // upper
      for (ii = 0; ii < tasdevent_nfadc; ii++)
        {
          // lower layer
          ru->fadcti[iwf][0] += (int) icrr->sub[jj].lwf[ii];
          ru->fadc[iwf][0][ii] = (int) icrr->sub[jj].lwf[ii];

          // upper layer
          ru->fadc[iwf][1][ii] = (int) icrr->sub[jj].uwf[ii];
          ru->fadcti[iwf][1] += (int) icrr->sub[jj].uwf[ii];

          // making sure that the fadc readout has reasonable values everywhere
          if ((ru->fadc[iwf][0][ii] < 0) || (ru->fadc[iwf][1][ii] < 0))
            eflag = true;

        }

      // Simple fadc pedestal average as integer value, for some consistency checks
      ru->fadcav[iwf][0] = (int) icrr->sub[jj].lpedAvr;
      ru->fadcav[iwf][1] = (int) icrr->sub[jj].upedAvr;

      // Transfer rutgers calibration block
      for (ilayer = 0; ilayer < 2; ilayer++)
        {
          ru->pchmip[iwf][ilayer] = icrr->sub[jj].pchmip[ilayer];
          ru->pchped[iwf][ilayer] = icrr->sub[jj].pchped[ilayer];
          ru->lhpchmip[iwf][ilayer] = icrr->sub[jj].lhpchmip[ilayer];
          ru->lhpchped[iwf][ilayer] = icrr->sub[jj].lhpchped[ilayer];
          ru->rhpchmip[iwf][ilayer] = icrr->sub[jj].rhpchmip[ilayer];
          ru->rhpchped[iwf][ilayer] = icrr->sub[jj].rhpchped[ilayer];

          /* Results from fitting 1MIP histograms */
          ru->mftndof[iwf][ilayer] = icrr->sub[jj].mftndof[ilayer];
          ru->mip[iwf][ilayer] = icrr->sub[jj].mip[ilayer];
          ru->mftchi2[iwf][ilayer] = icrr->sub[jj].mftchi2[ilayer];

          /*
           1MIP Fit function:
           [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
           4 fit parameters:
           [0]=Gauss Mean
           [1]=Gauss Sigma
           [2]=Linear Coefficient
           [3]=Overall Scaling Factor
           */
          for (ipar = 0; ipar < 4; ipar++)
            {
              ru->mftp[iwf][ilayer][ipar] = icrr->sub[jj].mftp[ilayer][ipar];
              ru->mftpe[iwf][ilayer][ipar] = icrr->sub[jj].mftpe[ilayer][ipar];
            }
        }

      // Making sure that certain waveform variables have reasonable values

      if (ru->wf_id[iwf] < 0)
        eflag = true;

      if ((ru->xxyy[iwf] / 100) < 1 || (ru->xxyy[iwf] % 100) < 1)
        eflag = true;

      if (ru->clkcnt[iwf] < 0)
        eflag = true;

      if (ru->mclkcnt[iwf] < 1)
        eflag = true;

      if (ru->fadcti[iwf][0] < 1 || ru->fadcti[iwf][1] < 1)
        eflag = true;

      // for broken waveforms only ( saturated does necessarily mean broken)
      if ((icrr->sub[jj].wfError & 0x40))
        eflag = true;

      // If some waveform is broken then increase the error code by 1
      // ( if many waveforms are corrupted, then this code will be large)
      if (eflag)
        {
          ru->errcode++;
          continue;
        }
      iwf++; // waveform counter index
    }

  ru->nofwf = iwf; // number of waveforms to analyze
  
  // calculate rusdraw site using information from the towerflag variable
  ru->site = rusdraw_site_from_bitflag(towerflag);
  
  // if this is a MC event, then also convert the thrown MC information
  if (icrr->eventCode == 0)
    addThrownMCinfo();
  
  // event number in the dst file
  event_num++; 
  
  // return success
  return true;

}

bool icrr2ru::addThrownMCinfo()
{
  const SDCalibevSimInfo *sim = &icrr->sim;
  
  // maximum string length of the particle name is 8 in SDCalibevSimInfo
  char pname[8] = {0};

  // particle type set to zero;  if any particles recognized in the ICRR bank then this number
  // will be set to a correct corsika ID of the particle
  rumc->parttype = 0;

  // get rid of any spaces and put everything to lower case
  // to simplify the particle identification
  int pname_len = 0;
  for (int i = 0; i < 7; i++)
    {
      if (sim->primaryParticleType[i] == '\0')
        break;
      if (sim->primaryParticleType[i] != ' ')
        {
          pname[pname_len] = tolower(sim->primaryParticleType[i]);
          pname_len++;
        }
    }
  pname[pname_len] = '\0';

  if ((strcmp(pname, "p") == 0) || (strcmp(pname, "proton") == 0))
    rumc->parttype = 14;
  else if ((strcmp(pname, "fe") == 0) || (strcmp(pname, "iron") == 0))
    rumc->parttype = 5626;
  else if ((strcmp(pname, "g") == 0) || (strcmp(pname, "gamma") == 0))
    rumc->parttype = 1;
  else
    rumc->parttype = 0;

  rumc->event_num = sim->trgCode; // put in something for event number.  irrelevant really

  rumc->energy = sim->primaryEnergy / 1.0e18; // Thrown energy in [EeV]

  // rusdmc convention is the core time with respect to PPS is in clock count units
  // with 50e6 clock counts per second
  rumc->tc = (int) Floor(0.5 + (sim->primaryArrivalTimeFromPps * 50e6));

  // In rusdmc, core position in CLF frame is measured in [cm]:
  rumc->corexyz[0] = sim->primaryCorePosX * 100.0;
  rumc->corexyz[1] = sim->primaryCorePosY * 100.0;
  rumc->corexyz[2] = sim->primaryCorePosZ * 100.0;

  // height of the first interaction is in [cm] in rumc
  rumc->height = 100.0 * SDGEN::stdz76_inv(sim->primaryFirstIntDepth);

  //Zenith angle, radians
  rumc->theta = ACos(sim->primaryCosZenith);

  //Azimuthal angle, radians. In rumc, azimuth angle is along the direction of propagation.
  rumc->phi = DegToRad() * (sim->primaryAzimuth + 180.0);
  while (rumc->phi < 0.0)
    rumc->phi += 2.0 * Pi();
  while (rumc->phi > 2.0 * Pi())
    rumc->phi -= 2.0 * Pi();

  return true;
}

