{
  TString SDDIR =
    TString(gSystem->Getenv("SDDIR") ?
	    gSystem->Getenv("SDDIR") :
	    gSystem->DirName(gSystem->DirName(TString(__FILE__).ReplaceAll("/./","/")))
	    ).Strip(TString::kTrailing,'/');
  if(gSystem->Load(SDDIR+TString("/lib/libatmparfitter.") + TString(gSystem->GetSoExt())) < 0)
    exit(2);
}
