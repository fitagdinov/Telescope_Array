#include "sddstio.h"
#include "TStopwatch.h"
#include "pass1plot.h"

Int_t pass1plot::get_yymmdd()
{
  return (Int_t)(event_time_stamp.GetDate()-10000*2000);
}
Int_t pass1plot::get_hhmmss()
{
  return (Int_t)(event_time_stamp.GetTime());
}
Int_t pass1plot::get_usec()
{
  return (Int_t)TMath::Floor((1e-3*(Double_t)event_time_stamp.GetNanoSec())+0.5);
}

// Convert year, month, day to julian days since 1/1/2000
int pass1plot::greg2jd(int year, int month, int day)
{
  return SDGEN::greg2jd(year,month,day);
}
  
// Convert julian days corresponding to midnight since Jan 1, 2000 to gregorian date
void pass1plot::jd2greg(double julian, int *year, int *month, int *day)
{
  return SDGEN::jd2greg(julian,year,month,day);
}
  
// Get time in seconds since midnight of Jan 2000
int pass1plot::time_in_sec_j2000(int year,int month,int day,
		      int hour,int minute,int second)
{
  return SDGEN::time_in_sec_j2000(year,month,day,hour,minute,second);
}
  
// fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
// xfd[3]  - vector in FD frame, [meters] (INPUT)
// xclf[3] - vector in CLF frame, [meters] (OUTPUT)
bool pass1plot::fdsite2clf(int fdsiteid, double *xfd, double *xclf)
{ return tacoortrans::fdsite2clf(fdsiteid,xfd,xclf);}

// fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
// xclf[3] - vector in CLF frame, [meters], (INPUT)
// xfd[3]  - vector in FD frame, [meters], (OUTPUT)
bool pass1plot::clf2fdsite(int fdsiteid, double *xclf, double *xfd)
{ return tacoortrans::clf2fdsite(fdsiteid,xclf,xfd); }


bool pass1plot::dump_dst_class(dstbank_class* dstbank, Int_t format, FILE* fp)
{
  if(!dstbank)
    {
      fprintf(stderr,"error: pass1plot::dump_dst_class: dstbank not initialized\n");
      return false;
    }
  dstbank->loadToDST();
  TString bank_name = dstbank->ClassName();
  bank_name.ToLower();
  bank_name.ReplaceAll("_class","");
  if(bank_name.Length() < 1)
    {
      fprintf(stderr,"error: object has an invalid dst class name %s\n",bank_name.Data());
      return false;
    }
  Int_t bankid;
  if (!(bankid=eventIdFromName((integer1 *)bank_name.Data())))
    {
      fprintf(stderr,"error:pass1plot::dump_dst_class: failed to indentify bank name = %s\n",
              bank_name.Data());
      return false;
    }
  Int_t banklist = newBankList(5);
  addBankList(banklist,bankid);
  eventSetDumpFormat(banklist,format);
  eventDumpf(fp,banklist);
  clrBankList(banklist);
  delBankList(banklist);
  return true;
} 

bool pass1plot::events_to_dst_file(const char* dst_file_name, Int_t imin, Int_t imax)
{
  if(!pass1tree)
    {
      fprintf(stderr,"error: pass1tree not initialized!\n");
      return false;
    }
  if (imin < 0 || imin > pass1tree->GetEntries() - 1 )
    {
      fprintf(stderr,"error: imin must be in 0 to %d - 1 = %d range\n", 
	      (Int_t)pass1tree->GetEntries(), (Int_t)pass1tree->GetEntries() - 1);
      return false;
    }
  if(imax < 0 || imax > pass1tree->GetEntries() - 1)
    {
      fprintf(stderr,"error: imax must be in 0 to %d - 1 = %d range\n", 
	      (Int_t)pass1tree->GetEntries(), (Int_t)pass1tree->GetEntries() - 1);
      return false;
    }
  if(imax < imin)
    {
      fprintf(stderr,"error: imax must be >= imin (imax==imin for just one event)\n");
      return false;
    }
  // iterate over all branches that are present in the event, 
  // create the output DST bank list, and load all dst classes
  // into the corresponding DST banks
  Int_t outBanks = newEmptyBankList();
  sddstio_class* dstio = new sddstio_class; // initialize DST I/O
  Int_t i_cur = pass1tree->GetReadEvent(); // save current event
  for (Int_t i=imin; i<=imax; i++)
    {	  
      GetEntry(i);
      clrBankList(outBanks); // clean up the output banks
      for (vector<dstbank_class**>::iterator idst = dst_branches.begin();
	   idst != dst_branches.end(); idst++)
	{
	  dstbank_class *dstbank = *(*idst);
	  TString bank_name = dstbank->ClassName();
	  bank_name.ToLower();
	  bank_name.ReplaceAll("_class","");
	  if(!bank_name.Length())
	    {
	      fprintf(stderr,"error: object has an invalid dst class name\n");
	      return false;
	    }
	  Int_t bankid = eventIdFromName((integer1 *)bank_name.Data());
	  if (!bankid)
	    {
	      fprintf(stderr,"error:pass1plot::event_to_dst_file: failed to indentify bank name = %s\n",
		      bank_name.Data());
	      return false;
	    }
	  addBankList(outBanks,bankid);
	  dstbank->loadToDST();
	}
      // start the DST file if it hasn't been started yet
      if(!dstio->outFileOpen())
	{
	  TString outfile = (dst_file_name ? dst_file_name : "");
	  if(!outfile.Length())
	    outfile.Form("events_%06d_%06d.dst.gz",imin,imax);
	  dstio->openDSToutFile(outfile.Data(),true);
	}
      // write the event out
      dstio->writeEvent(outBanks);
    }
  // clean up and close the DST file
  clrBankList(outBanks);
  delBankList(outBanks);
  if(dstio->outFileOpen())
    dstio->closeDSToutFile();
  if(dstio)
    delete dstio;
  GetEntry(i_cur); // restore current event
  return true;
}

Bool_t pass1plot::continue_activity(Double_t time_interval_seconds)
{
  // To continue doing some activity every time_interval_seconds
  // until enter is pressed
  
  for(TStopwatch t; t.RealTime() < time_interval_seconds && !have_stdin(); t.Continue());

  // return false if ENTER was pressed
  if(have_stdin())
    return false;
  
  // return true if enter was not pressed
  return true;
}
