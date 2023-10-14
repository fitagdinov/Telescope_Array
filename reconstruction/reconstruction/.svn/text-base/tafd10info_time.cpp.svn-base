#include "tafd10info.h"
#include "time.h"
#include <math.h>

static void brlr_julday2caldat(int julday, int *year, int *month, int *day) 
{
  // Routine by Sean R. Stratton
  int j, g, dg, c, dc, b, db, a, da, y, m, d;
  j = julday + 32044;
  g = j / 146097;
  dg = j % 146097;
  c = ( dg/36524 + 1 ) * 3 / 4;
  dc = dg - c*36524;
  b = dc / 1461;
  db = dc % 1461;
  a = ( db/365 + 1 ) * 3 / 4;
  da = db - a*365;
  y = g*400 + c*100 + b*4 + a;
  m = ( da*5 + 308 ) / 153 - 2;
  d = da - (m+4)*153/5 + 122;
  *year  = y - 4800 + (m+2)/12;
  *month = (m+2)%12 + 1;
  *day   = d + 1;
  return;
}

void tafd10info::get_brlr_time(int julian, int jsecond, int *yymmdd, int *hhmmss)
{
  int yr, mo, da;
  int hr, mi, sec;
  hr = jsecond / 3600 + 12;
  if (hr >= 24)
    {
      brlr_julday2caldat(julian + 1, &yr, &mo, &da);
      hr -= 24;
    }
  else
    brlr_julday2caldat(julian, &yr, &mo, &da);
  mi = (jsecond / 60) % 60;
  sec = jsecond % 60;
  yr -= 2000;
  (*yymmdd) = 10000 * yr + 100 * mo + da;
  (*hhmmss) = 10000 * hr + 100 * mi + sec;
}

// Compute MD time (jday, jsec are those appearing in hraw1, mcraw banks)
void tafd10info::get_md_time(int jday, int jsec, int *yymmdd, int *hhmmss)
{
  int year, month, day, hour, min, sec;
  const double JD0 = 2440587.5;
  double jd = (double) (jday) + (double) (jsec) / 86400.0e0 + 2440000;
  time_t s = (time_t) floor((jd - JD0) * 86.4e3+0.5);
  struct tm *t = gmtime(&s);
  year = t->tm_year + 1900;
  month = t->tm_mon + 1;
  day = t->tm_mday;
  hour = t->tm_hour;
  min = t->tm_min;
  sec = t->tm_sec;
  (*yymmdd) = (year - 2000) * 10000 + month * 100 + day;
  (*hhmmss) = hour * 10000 + min * 100 + sec;
}
