#include "pass1plot.h"


// Set the event time stamp in a best way possible
Bool_t pass1plot::set_event_time_stamp(Bool_t set_to_default_value)
{
  // Initialize the event time stamp information to a default value:
  // 2001/01/01, midnight, 0 nano seconds
  event_time_stamp.Set(20010101,0,0,true,0);
  
  // If only the initialization requested then return
  if(set_to_default_value)
    return true;
  
  // Next, examine which detectros triggered
  if(have_rusdraw)
    {
      if(rusdraw->nofwf > 0)
	{
	  Int_t yyyymmdd = 2000*10000+rusdraw->yymmdd;
	  Int_t hhmmss   = rusdraw->hhmmss;
	  Int_t nsec     = 1000*rusdraw->usec;
	  event_time_stamp.Set(yyyymmdd,hhmmss,nsec,true,0);
	  return true;
	}
    }
  if(have_hraw1)
    {
      if(hraw1->nmir > 0)
	{
	  Int_t yymmdd,hhmmss;
	  tafd10info::get_md_time(hraw1->jday,hraw1->jsec,&yymmdd,&hhmmss);
	  Int_t yyyymmdd=2000*10000+yymmdd;
	  Int_t nsec = hraw1->mirtime_ns[0];
	  event_time_stamp.Set(yyyymmdd,hhmmss,nsec,true,0);
	  return true;
	}
    }
  if(have_mcraw)
    {
      get_hraw1();
      if(hraw1->nmir > 0)
	{
	  Int_t yymmdd,hhmmss;
	  tafd10info::get_md_time(hraw1->jday,hraw1->jsec,&yymmdd,&hhmmss);
	  Int_t yyyymmdd=2000*10000+yymmdd;
	  Int_t nsec = hraw1->mirtime_ns[0];
	  event_time_stamp.Set(yyyymmdd,hhmmss,nsec,true,0);
	  return true;
	}
    }
  // Also try BR/LR, if they have triggered, and use fdplane for timing
  if(have_fdplane[0] || have_fdplane[1])
    {
      if(!(have_fdplane[0] && set_fdplane(0)) || fdplane->ntube < 1)
	{
	  if(have_fdplane[1])
	    {
	      if(!set_fdplane(1))
		return false;
	    }
	}
      Int_t yymmdd,hhmmss;
      tafd10info::get_brlr_time(fdplane->julian,fdplane->jsecond,&yymmdd,&hhmmss);
      Int_t yyyymmdd = 2000*10000+yymmdd;
      Double_t fnsec=(Double_t)fdplane->jsecfrac - 25.6e3;
      Double_t tbtime_earliest = 2e9;
      for (Int_t itube=0; itube < fdplane->ntube; itube++)
	{
	  if(fdplane->tube_qual[itube] != 1)
	    continue;
	  if(fdplane->time[itube] < tbtime_earliest)
	    tbtime_earliest = fdplane->time[itube];
	}
      if(tbtime_earliest < 1e9)
	fnsec += tbtime_earliest;
      Int_t nsec=(Int_t)TMath::Floor(fnsec+0.5);
      event_time_stamp.Set(yyyymmdd,hhmmss,nsec,true,0);
      return true;
    }
  
  // Finally, if none of the above detectors triggered, try using rusdraw again
  // This time without requiring that the TALE SD triggered, because if rusdraw
  // is present but the event did not trigger then it's likely that this is a
  // thrown TALE SD MC event and rusdraw has the time information
  if(have_rusdraw)
    {
      Int_t yyyymmdd = 2000*10000+rusdraw->yymmdd;
      Int_t hhmmss   = rusdraw->hhmmss;
      Int_t nsec     = 1000*rusdraw->usec;
      event_time_stamp.Set(yyyymmdd,hhmmss,nsec,true,0);
      return true;
    }
  return false;
}
