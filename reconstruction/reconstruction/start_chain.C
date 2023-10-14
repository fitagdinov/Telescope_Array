{

  Bool_t *ran_start;
  Bool_t successFlag;
  if (ran_start != 0)
    {
      fprintf(stderr, 
	      "A start script can be executed only once per session\n");
      return;
    }
  else
    ran_start = new Bool_t;
   
  // Full path to SD analysis folder, get it from the environmental
  // variable SDDIR. User can set it by sourcing the environmental file)
  TString SDDIR = 
    TString(gSystem->Getenv("SDDIR") ?
	    gSystem->Getenv("SDDIR") :
	    gSystem->DirName(gSystem->DirName(TString(__FILE__).ReplaceAll("/./","/")))
	    ).Strip(TString::kTrailing,'/');
  TString PASS1PLOTDIR = SDDIR + "/pass1plot";
  // Save the current directory, will be needed when the script will
  // be changing paths to check if certain directories/files exist
  TString CURDIR = gSystem->pwd();
  
  if(!gSystem->cd(SDDIR))
    {
      fprintf(stderr,"\nERROR: %s directory not found\n",SDDIR.Data());
      fprintf(stderr,"set the SDDIR enironmental variable ");
      fprintf(stderr,"to point to correct /full/path/to/sdanalysis/build\n");
      fprintf(stderr,"on your machine\n\n");
      exit(2);
    }
  if(!gSystem->cd(PASS1PLOTDIR))
    {
      fprintf(stderr,"\nERROR: %s directory not found\n",PASS1PLOTDIR.Data());
      fprintf(stderr,"set the SDDIR enironmental variable ");
      fprintf(stderr,"to point to correct /full/path/to/sdanalysis/build\n");
      fprintf(stderr,"on your machine\n\n");
      exit(2);
    }
  gSystem->cd(CURDIR);


  // directory wherre SD PASS1PLOT macro files are
  TString MACRODIR=SDDIR+"/pass1plot/macro";
  if(gSystem->cd(MACRODIR))
    gSystem->cd(CURDIR);
  else
    {
      fprintf(stderr,"\nERROR: %s directory not found\n",MACRODIR.Data());
      fprintf(stderr,"set the SDDIR enironmental variable ");
      fprintf(stderr,"to point to correct /full/path/to/sdanalysis\n");
      fprintf(stderr,"on your machine\n\n");
      exit(2);
    }
  
  // Directory where various sdanalysis libraries are
  TString SDLIBDIR=SDDIR+"/lib";
  if(gSystem->cd(SDLIBDIR))
    gSystem->cd(CURDIR);
  else
    {
      fprintf(stderr,"\nERROR: %s directory not found\n",SDLIBDIR.Data());
      fprintf(stderr,"set the SDDIR enironmental variable ");
      fprintf(stderr,"to point to correct /full/path/to/sdanalysis\n");
      fprintf(stderr,"on your machine\n\n");
      exit(2);
    }
  
  // Directory where various sdanalysis includes are
  TString SDINCDIR=SDDIR+"/inc";
  if(gSystem->cd(SDINCDIR))
    gSystem->cd(CURDIR);
  else
    {
      fprintf(stderr,"\nERROR: %s directory not found\n",SDINCDIR.Data());
      fprintf(stderr,"set the SDDIR enironmental variable ");
      fprintf(stderr,"to point to correct /full/path/to/sdanalysis\n");
      fprintf(stderr,"on your machine\n\n");
      exit(2);
    }
  
  gROOT->Macro(MACRODIR+"/"+"load_lib.C");
  gROOT->Macro(MACRODIR+"/"+"glob_var.C");
  gROOT->Macro(MACRODIR+"/"+"init_event_display.C");
  gROOT->LoadMacro(MACRODIR+"/"+"drawing.C");
  gROOT->LoadMacro(MACRODIR+"/"+"gtplot.C");
  gROOT->LoadMacro(MACRODIR+"/"+"coortrans.C");
  gROOT->LoadMacro(MACRODIR+"/"+"geomFitting.C");
  gROOT->LoadMacro(MACRODIR+"/"+"ldfFitting.C");
  gROOT->LoadMacro(MACRODIR+"/"+"gldfFitting.C");
  gROOT->LoadMacro(MACRODIR+"/"+"plotStats.C");
  gROOT->LoadMacro(MACRODIR+"/"+"drawEvent.C");
  gROOT->LoadMacro(MACRODIR+"/"+"plotEvent.C");
  gROOT->LoadMacro(MACRODIR+"/"+"debugging.C");
  gROOT->LoadMacro(MACRODIR+"/"+"miscFun.C");
  gROOT->LoadMacro(MACRODIR+"/"+"formattedWriting.C");
  gROOT->LoadMacro(MACRODIR+"/"+"plotVars.C");
  gROOT->LoadMacro(MACRODIR+"/"+"waveform.C");
  gROOT->LoadMacro(MACRODIR+"/"+"dst_banks.C");
  gROOT->LoadMacro(MACRODIR+"/"+"user_steering.C");
  
  
  // apply gStyle settings
  set_glob_style();
  
  TCanvas* c1 = new TCanvas("c1","c1",0,10,700,700);
  set_grid(c1);
  
  TCanvas* c2 = new TCanvas("c2","c2",0,10,700,700);
  set_grid(c2);
  
  TCanvas* c3 = new TCanvas("c3","c3",0,10,700,700);
  set_grid(c3);
  
  TCanvas* c4 = new TCanvas("c4","c4",0,10,700,700);
  set_grid(c4);
  
  TCanvas* c5 = new TCanvas("c5","c5",0,10,700,700);
  set_grid(c5);
  
  TCanvas* c6 = new TCanvas("c6","c6",0,10,700,700);
  set_grid(c6);
  
  // by default, arrange canvases for SD event display
  sd_mode();
  
  // make all titles small
  smallTitle();
  
  /////////////// CANVAS INITIALIZATION (ABOVE) //////////////////

  
  plotEventBoth(0);
  gtplot();

  
}
