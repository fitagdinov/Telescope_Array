{
  
  // Directory where various tlanalysis libraries are
  TString SDLIBDIR=SDDIR+"/lib";
  if(gSystem->cd(SDLIBDIR))
    gSystem->cd(CURDIR);
  else
    {
      fprintf(stderr,"\nERROR: %s directory not found\n",SDLIBDIR.Data());
      fprintf(stderr,"set the SDDIR enironmental variable ");
      fprintf(stderr,"to point to correct /full/path/to/tlanalysis\n");
      fprintf(stderr,"on your machine\n\n");
      exit(2);
    }
  
  
  std::vector<TString> SHARED_LIBS_FOR_PASS1PLOT;
  SHARED_LIBS_FOR_PASS1PLOT.push_back("pass1plot");           // Event display plotting manager
  SHARED_LIBS_FOR_PASS1PLOT.push_back("sdgeomfitter");        // SD geometry fitter
  SHARED_LIBS_FOR_PASS1PLOT.push_back("ldffitter");         // SD LDF fitter
  SHARED_LIBS_FOR_PASS1PLOT.push_back("gldffitter");        // SD heometry and LDF fitter
  
  //////////// need to load explicitly on some platforms ////////
  if(gSystem->Load(TString("libMinuit.") + gSystem->GetSoExt()) < 0)
    {
      fprintf(stderr,"error: no libMinuit.%s; check your ROOT installation!\n", gSystem->GetSoExt());
      exit(2);
    }

  for (Int_t ilib = 0; ilib < (Int_t) SHARED_LIBS_FOR_PASS1PLOT.size(); ilib++)
    {
      if(gSystem->Load(
		       SDLIBDIR+
		       TString("/")+
		       TString("lib")+SHARED_LIBS_FOR_PASS1PLOT[ilib]+TString(".")+TString(gSystem->GetSoExt())
		       ) < 0)
	{
	  fprintf(stderr,"error: need to compile pass1plot\n");
	  exit(2);
	}
    }
   
}
