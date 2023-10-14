// print dst bank by passing dst bank name, e.g. "rusdraw", "rufptn", "rufldf", etc
bool print_bank(const char* bank_name, Int_t format = 0, FILE *fp = stdout)
{
  TBranch* b = p1.pass1tree->GetBranch(bank_name);
  if(!b)
    {
      fprintf(stderr,"error: branch with name %s not found\n",bank_name);
      return false;
    }
  dstbank_class** dstbank = (dstbank_class**) b->GetAddress();
  if(!(*dstbank))
    {
      fprintf(stderr,"error: branch %s address was not set properly\n",bank_name);
      return false;
    }
  return p1.dump_dst_class((*dstbank),format,fp);
}

// print dst bank by passing a pointer to a dst class (e.g. p1.rufptn, p1.rufdf, etc)
bool print_bank(dstbank_class* dstbank, Int_t format = 0, FILE *fp = stdout)
{ 
  return p1.dump_dst_class(dstbank,format,fp); 
}



bool event_to_dst_file(const char* dst_file_name=0)
{
  // if dst_file_name is a zero pointer then use the time stamp
  // to create the output file name
  TString outfile;
  if(dst_file_name)
    outfile=dst_file_name;
  else
    {
      TString bname;
      Int_t yyyymmdd=p1.get_yymmdd()+2000*10000;
      Int_t hhmmss=p1.get_hhmmss();
      Int_t usec=p1.get_usec();
      outfile.Form("event_%08d_%06d_%06d.dst.gz",yyyymmdd,hhmmss,usec);
    }
  Int_t ievent = (Int_t)p1.GetReadEvent();
  return p1.events_to_dst_file(outfile,ievent,ievent);
}

bool events_to_dst_file(const char* dst_file_name=0, 
			Int_t imin=0, Int_t imax=0)
{
  return p1.events_to_dst_file(dst_file_name,imin,imax);
}
