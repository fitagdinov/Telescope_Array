#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "event.h"
#include "filestack.h"
#include "convtime.h"
#include "sddstio.h"


typedef struct
{
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
  int microsecond;
} event_time_struct;

static bool printevtime_have_time = false;

void parseCmdLine (int argc, char *argv[]);

// BR/LR FD
static void compute_fdplane_time(fdplane_dst_common *fdpln, event_time_struct *t)
{
  int yr, mo, da;
  int hr, mi, sec, nsec;
  hr = fdpln->jsecond / 3600 + 12;
  if (hr >= 24)
    {
      julday2caldat(fdpln->julian+1, &yr, &mo, &da);
      hr -= 24;
    }
  else
    julday2caldat(fdpln->julian, &yr, &mo, &da);
  mi = (fdpln->jsecond / 60 ) % 60;
  sec = fdpln->jsecond % 60;
  nsec = fdpln->jsecfrac;
  t->year = yr;
  t->month = mo;
  t->day = da;
  t->hour=hr;
  t->minute = mi;
  t->second = sec;
  t->microsecond = (int)floor((double)nsec/1.0e3 + 0.5);
}
//////////////////////////////////////// MD /////////////////////////////////////
static void printevtime_astroFromjday(double jd, int *year, int *month, int *day, 
				      int *hour, int *min, int *sec, char *zone,
				      int local_time = 0)
{
  const double JD0=2440587.5;
  time_t s = (time_t)floor((jd - JD0) * 86.4e3+0.5);
  struct tm *t;
  if (local_time)
    t = localtime(&s); 
  else
    t = gmtime(&s);
  if (year != NULL) *year = t->tm_year + 1900;
  if (month != NULL) *month = t->tm_mon + 1;
  if (day != NULL) *day = t->tm_mday;
  if (hour != NULL) *hour = t->tm_hour;
  if (min != NULL) *min = t->tm_min;
  if (sec != NULL) *sec = t->tm_sec;
  if (zone != NULL) strncpy(zone, local_time ? tzname[(t->tm_isdst > 0)] : "UTC", 4);
}
static void printevtime_get_md_time(int jday,  int jsec, 
				    int *year, int *month, int *day, 
				    int *hour, int *min, int *sec, 
				    char *zone,int local_time = 0)
{
  double jdayReal;
  jdayReal  = (double) (jday) + (double) (jsec) / 86400.0e0 + 2440000;
  printevtime_astroFromjday(jdayReal,year,month,day,hour,min,sec,zone,local_time);
}
// MD (hraw1)
static void compute_hraw1_time(event_time_struct *t)
{
  int yr, mo, da;
  int hr, mi, sec, nsec;
  printevtime_get_md_time(hraw1_.jday,hraw1_.jsec,&yr,&mo,&da,&hr,&mi,&sec,0,0);
  nsec = hraw1_.mirtime_ns[0];
  t->year = yr;
  t->month = mo;
  t->day = da;
  t->hour=hr;
  t->minute = mi;
  t->second = sec;
  t->microsecond = (int)floor((double)nsec/1.0e3 + 0.5);
}
// MD (mcraw)
static void compute_mcraw_time(event_time_struct *t)
{
  int yr, mo, da;
  int hr, mi, sec, nsec;
  printevtime_get_md_time(mcraw_.jday,mcraw_.jsec,&yr,&mo,&da,&hr,&mi,&sec,0,0);
  nsec = mcraw_.mirtime_ns[0];
  t->year = yr;
  t->month = mo;
  t->day = da;
  t->hour=hr;
  t->minute = mi;
  t->second = sec;
  t->microsecond = (int)floor((double)nsec/1.0e3 + 0.5);
}
// SD (tasdcalibev)
static void compute_tasdcalibev_time(event_time_struct *t)
{
  t->year        =  2000 + tasdcalibev_.date / 10000;
  t->month       =  (tasdcalibev_.date % 10000) / 100;
  t->day         =  tasdcalibev_.date % 100;
  t->hour        =  tasdcalibev_.time / 10000;
  t->minute      =  (tasdcalibev_.time % 10000) / 100;
  t->second      =  tasdcalibev_.time % 100;
  t->microsecond =  tasdcalibev_.usec;
}
// SD (rusdraw)
static void compute_rusdraw_time(event_time_struct *t)
{
  t->year        =  2000 + rusdraw_.yymmdd / 10000;
  t->month       =  (rusdraw_.yymmdd % 10000) / 100;
  t->day         =  rusdraw_.yymmdd % 100;
  t->hour        =  rusdraw_.hhmmss / 10000;
  t->minute      =  (rusdraw_.hhmmss % 10000) / 100;
  t->second      =  rusdraw_.hhmmss % 100;
  t->microsecond =  rusdraw_.usec;
}


// MD (hraw1)


/* Print event times for any detector */
static void print_event_time(FILE *fp, event_time_struct *t, char *dname)
{
  if (t)
    {
      fprintf (fp, "%s_time: %d %d %d %d %d %d %d\t",
	       dname,t->year,t->month,t->day,
	       t->hour,t->minute,t->second,
	       t->microsecond);
      printevtime_have_time = true;
    }
  else
    fprintf (fp, "0: 0 0 0 0 0 0 0\t");
}

int main(int argc, char **argv)
{
  char *dstfile;
  FILE *fp;
  sddstio_class *dstio = new sddstio_class();
  event_time_struct t;
  int bankid;
  parseCmdLine(argc,argv);
  fp = stdout;
  while((dstfile=pullFile()))
    {
      if(!dstio->openDSTinFile(dstfile))
	continue;
      while (dstio->readEvent()) 
	{
	  printevtime_have_time = false;
	  // BR-FD time
	  if (dstio->haveBank(BRPLANE_BANKID))
	    {
	      compute_fdplane_time(&brplane_,&t);
	      print_event_time(fp,&t,(char *)"BRFD");
	    }
	  // LR-FD time
	  if (dstio->haveBank(LRPLANE_BANKID))
	    {
	      compute_fdplane_time(&lrplane_,&t);
	      print_event_time(fp,&t,(char *)"LRFD");
	    }
	  // MD-FD time
	  bankid = 0;
	  if (dstio->haveBank(MCRAW_BANKID))
	    bankid = MCRAW_BANKID;
	  if(dstio->haveBank(HRAW1_BANKID))
	    bankid = HRAW1_BANKID;	  
	  if (bankid)
	    {
	      if (bankid==MCRAW_BANKID)
		{
		  compute_hraw1_time(&t);
		  print_event_time(fp,&t,(char *)"MDFD");
		}
	      if (bankid==HRAW1_BANKID)
		{
		  compute_mcraw_time(&t);
		  print_event_time(fp,&t,(char *)"MDFD");
		}
	    }
	  // SD time
	  bankid = 0;
	  if (dstio->haveBank(RUSDRAW_BANKID))
	    bankid = RUSDRAW_BANKID;
	  if (dstio->haveBank(TASDCALIBEV_BANKID))
	    bankid = TASDCALIBEV_BANKID;	  
	  if (bankid)
	    {
	      if (bankid == RUSDRAW_BANKID)
		{
		  compute_rusdraw_time(&t);
		  print_event_time(fp,&t,(char *)"SD");
		}
	      if (bankid == TASDCALIBEV_BANKID)
		{
		  compute_tasdcalibev_time(&t);
		  print_event_time(fp,&t,(char *)"SD");
		}
	    }
	  if (!printevtime_have_time)
	    print_event_time(fp,0,0);
	  fprintf (fp, "\n");
	}
      dstio->closeDSTinFile();
    }
  return 0;
}

void parseCmdLine (int argc, char *argv[])
{
  char line[1024], *name;
  FILE *listFile;
  integer4 i;  
  if (argc == 1) {
    
    fprintf (stderr, "\n\nPrint events times stored in DST files\n");
    fprintf (stderr, "Usage: %s [-i dst list file] ...\n",argv[0]);
    fputs ("  -i <string>:   read input files from a dst list file\n", stderr);
    fputs ("or just pass DST file names as arguments without any prefixes or switches\n",stderr);
    fprintf (stderr, "\n");
    exit (2);
  }  
  for (i = 1; i < argc; ++i) {
    if (strcmp (argv[i], "-i") == 0) {
      if (++i >= argc || argv[i][0] == '-') {
	fprintf (stderr, "Input list option specified but no filename "
		 "given\n");
	exit (1);
      }
      else if ( (listFile=fopen (argv[i], "r")) ) {
	while (fgets (line, 1024, listFile)) {
	  name = strtok (line, " \t\r\n");
	  if (strlen (name) > 0) {
	    pushFile (name);
	  }
	}
	fclose (listFile);
      }
      else {
	fprintf (stderr, "Failed to open input list file %s\n", argv[i]);
	exit (1);
      }
    }
    else if (argv[i][0] != '-') {
      pushFile (argv[i]);
    }
    else {
      fprintf (stderr, "Unrecognized option: %s\n", argv[i]);
      exit (1);
    }
  }
  if (countFiles () < 1) {
    fprintf (stderr, "Input file(s) must be specified!\n");
    exit (1);
  }
}
