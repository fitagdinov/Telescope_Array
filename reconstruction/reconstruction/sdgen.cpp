// GENERAL SD ROUTINES WHICH WILL BE USED A LOT
// IN THE ANALYSIS PROGRAMS.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "sduti.h"
// Get corsika ID from particle name 
int SDGEN::get_corid(const char* pname)
{
  if (strcmp(pname, "gamma") == 0)
    return 1;
  else if (strcmp(pname, "e+") == 0 || strcmp(pname, "eplus") == 0)
    return 2;
  else if (strcmp(pname, "e-") == 0 || strcmp(pname, "eminus") == 0)
    return 3;
  else if (strcmp(pname, "mu+") == 0 || strcmp(pname, "muplus") == 0)
    return 5;
  else if (strcmp(pname, "mu-") == 0 || strcmp(pname, "muminus") == 0)
    return 6;
  else if (strcmp(pname, "pi0") == 0)
    return 7;
  else if (strcmp(pname, "pi+") == 0 || strcmp(pname, "piplus") == 0)
    return 8;
  else if (strcmp(pname, "pi-") == 0 || strcmp(pname, "piminus") == 0)
    return 9;
  else if (strcmp(pname, "n") == 0 || strcmp(pname, "neutron") == 0)
    return 13;
  else if (strcmp(pname, "p") == 0 || strcmp(pname, "proton") == 0)
    return 14;
  else if (strcmp(pname,"pbar")==0)
    return 15;
  else if(strcmp(pname,"nbar")==0)
    return 25;
  else
    fprintf(stderr,"PARTICLE = %s is not supported\n",pname);
  return 0;
}

// Get PDG ID and PDG mass from corsika ID
bool SDGEN::get_pdginfo(int corID, int* pdgid, double* pdgm)
{
  switch (corID)
    {
      // gamma
    case 1:
      (*pdgid) = 22;
      (*pdgm) = 0.0;
      break;

      // e+
    case 2:
      (*pdgid) = -11;
      (*pdgm) = 0.510999;
      break;

      // e-
    case 3:
      (*pdgid) = 11;
      (*pdgm) = 0.510999;
      break;

      // mu+
    case 5:
      (*pdgid) = -13;
      (*pdgm) = 105.658;
      break;

      // mu-
    case 6:
      (*pdgid) = 13;
      (*pdgm) = 105.658;
      break;

      // pi0
    case 7:
      (*pdgid) = 111;
      (*pdgm) = 134.976;
      break;

      // pi+
    case 8:
      (*pdgid) = 211;
      (*pdgm) = 139.57;
      break;

      // pi-
    case 9:
      (*pdgid) = -211;
      (*pdgm) = 139.57;
      break;

      // n
    case 13:
      (*pdgid) = 2112;
      (*pdgm) = 939.566;
      break;

      // p
    case 14:
      (*pdgid) = 2212;
      (*pdgm) = 938.272;
      break;

      // pbar
    case 15:
      (*pdgid) = -2212;
      (*pdgm) = 938.272;
      break;

      // nbar
    case 25:
      (*pdgid) = -2112;
      (*pdgm) = 939.566;
      break;

    default:
      fprintf(stderr, "particle corid=%d is not supported\n", corID);
      return false;
      break;
    }
  return true;
}

// Get PDG ID from CORSIKA ID
int SDGEN::corid2pdgid(int corid)
{
  double m;
  int pdgid;
  if (get_pdginfo(corid,&pdgid, &m))
    return pdgid;
  return 0;
}

// Get CORSIKA ID from PDG ID
int SDGEN::pdgid2corid(int pdgid)
{
  switch (pdgid)
    {
      // gamma
    case 22:
      return 1;
      break;

      // e+
    case -11:
      return 2;
      break;

      // -e
    case 11:
      return 3;
      break;

      // mu+
    case -13:
      return 5;
      break;

      // mu-
    case 13:
      return 6;
      break;

      // pi0
    case 111:
      return 7;
      break;

      // pi+
    case 211:
      return 8;
      break;

      // pi-
    case -211:
      return 9;
      break;

      // n
    case 2112:
      return 13;
      break;

      // p
    case 2212:
      return 14;
      break;

      // pbar
    case -2212:
      return 15;
      break;

      // nbar
    case -2112:
      return 25;
      break;

    default:
      fprintf(stderr, "particle pdgid=%d is not supported\n", pdgid);
      return 0;
      break;
    }
}

const char* SDGEN::pdgid2name(int pdgid)
{
  switch (pdgid)
    {
      // gamma
    case 22:
      return "gamma";
      break;

      // e+
    case -11:
      return "eplus";
      break;

      // -e
    case 11:
      return "eminus";
      break;

      // mu+
    case -13:
      return "muplus";
      break;

      // mu-
    case 13:
      return "muminus";
      break;

      // pi0
    case 111:
      return "pi0";
      break;

      // pi+
    case 211:
      return "piplus";
      break;

      // pi-
    case -211:
      return "piminus";
      break;

      // n
    case 2112:
      return "neutron";
      break;

      // p
    case 2212:
      return "proton";
      break;

      // pbar
    case -2212:
      return "pbar";
      break;

      // nbar
    case -2112:
      return "nbar";
      break;

    default:
      fprintf(stderr, "particle pdgid=%d is not supported\n", pdgid);
      return "";
      break;
    }
}

// for getting the sd coordinates
void SDGEN::xxyy2xy(int xxyy, int *xx, int *yy)
  {
    *xx = xxyy / 100;
    *yy = xxyy % 100;
  }

// To get year,month,day from yymmdd, or hour,minute second from hhmmss
void SDGEN::parseAABBCC(int aabbcc, int *aa, int *bb, int *cc)
  {
    *aa = aabbcc / 10000;
    *bb = (aabbcc % 10000) / 100;
    *cc = aabbcc % 100;
  }
void SDGEN::toAABBCC(int aa, int bb, int cc, int *aabbcc)
  {
    (*aabbcc) = aa*10000+bb*100+cc;
  }
// returns the time after midnight in seconds
int SDGEN::timeAftMNinSec(int hhmmss)
  {
    int hh, mm, ss;
    hh = hhmmss / 10000;
    mm = (hhmmss % 10000) / 100;
    ss = hhmmss % 100;
    return 3600*hh+60*mm+ss;
  }
// Convert year, month, day to julian days since 1/1/2000
int SDGEN::greg2jd(int year, int month, int day)
  {
    int a, b, c, e, f;
    int iyear, imonth, iday;
    iyear = year;
    imonth = month;
    iday = day;
    if (imonth <= 2)
      {
        iyear -= 1;
        imonth += 12;
      }
    a = iyear/100;
    b = a/4;
    c = 2-a+b;
    e = (int)floor(365.25 * (double)(iyear+4716));
    f = (int)floor(30.6001 * (imonth+1));
    // Julian days corresponding to midnight since Jan 1, 2000
    return (int) ((double)(c+iday+e+f)-1524.5 - 2451544.5);
  }

// Obtain number of days since midnight of Jan 1, 2000 using gregorian
// date in yymmdd format
int SDGEN::greg2jd(int yymmdd)
{
  int yy,mm,dd;
  parseAABBCC(yymmdd,&yy,&mm,&dd);
  yy += 2000;
  return greg2jd(yy,mm,dd);
}

// Convert julian days corresponding to midnight since Jan 1, 2000
// to gregorian date
void SDGEN::jd2greg(double julian, int *year, int *month, int *day)
  {
    int ja, jalpha, jb, jc, jd, je;
    static int igreg = 2299161;
    // below code works for full julian days
    julian += 2451544.5;
    if ( (julian+0.5) >= igreg)
      {
        jalpha = (int)( ( ( (julian+0.5) - 1867216.0 ) - 0.25 ) / 36524.25 );
        ja = (int)( (julian+0.5) + 1.00 + 0.75 * (double)jalpha );
      }
    else
      ja = (int) (julian+0.5);
    jb = (int)( (double)ja + 1524.5 );
    jc = (int)( 6680.0 + ( (float)(jb-2439870)-122.1) / 365.25 );
    jd = (int)( 365.25*(double)jc );
    je = (int)( (double)(jb - jd ) / 30.6001 );
    *day = jb - jd - (int)( 30.6001 * (double)je );
    *month = je - 1;
    if ( *month > 12)
      *month -= 12;
    *year = jc - 4715;
    if ( *month > 2)
      --(*year);
    if ( *year <= 0)
      --(*year);
  }

// Convert julian days corresponding to midnight since Jan 1, 2000
// to gregorian date in yymmdd format
int SDGEN::jd2yymmdd(double julian)
{
  int yy,mm,dd;
  jd2greg(julian,&yy,&mm,&dd);
  yy -= 2000;
  return 10000*yy+100*mm+dd;
}

// Change second by an integer ammount, original date and time variables
// will be overwritten with those corresponding to new second
void SDGEN::change_second(int *year, int *month, int *day, int *hr, int *min,
    int *sec, int correction_sec)
  {
    double jday;
    int jsec;
    // apply the correction to second after midnight
    jsec=(*hr)*3600 + (*min)*60 + (*sec) + correction_sec;
    // convert to normal UT time with correction applied
    jday = greg2jd((*year), (*month), (*day));
    while (jsec >= 86400)
      {
        jday ++;
        jsec -= 86400;
      }
    while (jsec < 0)
      {
        jday --;
        jsec += 86400;
      }
    jd2greg(jday, year, month, day);
    (*hr) = jsec/3600;
    (*min) = (jsec/60) % 60;
    (*sec) = jsec % 60;
  }

// Get time in seconds since midnight of Jan 1, 2000
int SDGEN::time_in_sec_j2000(int year, int month, int day, int hour, int minute, int second)
{
  return 86400 * greg2jd(year, month, day) + 3600*hour +60*minute+second;
}

// Get time in seconds since midnight of Jan 1, 2000
int SDGEN::time_in_sec_j2000(int yymmdd, int hhmmss)
{
  int yr,mo,da,hr,mi,sec;
  parseAABBCC(yymmdd,&yr,&mo,&da);
  yr += 2000;
  parseAABBCC(hhmmss,&hr,&mi,&sec);
  return time_in_sec_j2000(yr,mo,da,hr,mi,sec);
}

// Get time in seconds since midnight of Jan 1, 2000 including second fraction
double SDGEN::time_in_sec_j2000f(int year, int month, int day, int hour, int minute, int second, int usec)
{
  return ( ((double)usec)/1.0e6 + 
	   (double)time_in_sec_j2000(year,month,day,hour,minute,second) );
}

// Get time in seconds since midnight of Jan 1, 2000 including second fraction
double SDGEN::time_in_sec_j2000f(int yymmdd, int hhmmss, int usec)
{
  int yr,mo,da,hr,mi,sec;
  parseAABBCC(yymmdd,&yr,&mo,&da);
  yr += 2000;
  parseAABBCC(hhmmss,&hr,&mi,&sec);
  return time_in_sec_j2000f(yr,mo,da,hr,mi,sec,usec);
}

// Get calendar date from time in seconds since midnight of Jan 1, 2000
void SDGEN::j2000sec2greg(int j2000sec, int *year, int *month, int *day)
  {
    jd2greg((((double)j2000sec)/86400.0),year,month,day);
  }

// Get calendar date in yymmdd format from time in seconds since midnight of Jan 1, 2000
int SDGEN::j2000sec2yymmdd(int j2000sec)
  {
    int yr,mo,da;
    jd2greg((((double)j2000sec)/86400.0),&yr,&mo,&da);
    yr-=2000;
    return (yr*10000+mo*100+da);
  }

int SDGEN::getFitStatus(char *Migrad)
/*****/
  {
    /* Transforming the MIGRAD char[]-status to a int-status 
     *
     ***********************************************************/
    int i;
    char Status[6][12]=
      { "FAILED", "PROBLEMS", "CALL LIMIT", "NOT POSDEF", "CONVERGED",
          "SUCCESSFUL" };
    for (i=0; i < 6; i++)
      if (strstr(Migrad, Status[i]) != NULL)
        return i;
    return -1;
  }
