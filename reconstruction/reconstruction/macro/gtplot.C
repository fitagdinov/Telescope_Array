//
// Collection of routines for manipulating canvases
//

const Int_t GTPLOT_NCANV = 6;

void set_grid(TVirtualPad* p)
{
  if(use_grid)
    {
      p->SetGridx(1);
      p->SetGridy(1);
    }
  else
    {
      p->SetGridx(0);
      p->SetGridy(0);
    }
  if(use_ticks)
    {
      p->SetTickx(1);
      p->SetTicky(1);
    }
  else
    {
      p->SetTickx(0);
      p->SetTicky(0);
    }
}

void update_canv(TCanvas* canv)
{
  if(!canv)
    return;
  canv->Modified();
  canv->Update();
}

void zoomin(TCanvas* canv, Int_t xsize, Int_t ysize)
{  
  canv->SetWindowSize(xsize,ysize);
  canv->SetWindowPosition(1,1);
  canv->SetWindowPosition(0,0);
  update_canv(canv);
}

void zoomin(Int_t icanv, Int_t xsize, Int_t ysize)
{ 
  if(icanv < 1 || icanv > GTPLOT_NCANV)
    {
      fprintf(stderr,"error: zoomin: icanv must be in 1-%d range\n",
	      GTPLOT_NCANV);
      return;
    }
  TCanvas* canv = 0;
  TString canvName="c";
  canvName += icanv;
  if((canv=(TCanvas *)gROOT->FindObject(canvName)) && 
     canv->InheritsFrom("TCanvas"))
    zoomin(canv,xsize,ysize);
  else
    fprintf(stderr,"error: canvas %s not found\n",canvName.Data());
}

void zoomin(TCanvas* canv, Int_t xysize=600)
{ zoomin(canv,xysize,xysize); }

void zoomin(Int_t icanv, Int_t xysize=600)
{ zoomin(icanv,xysize,xysize); }

void updateAll()
{
  Long_t icanvas;
  TString canvName;
  TCanvas *canv;
  for (icanvas=1; icanvas <= GTPLOT_NCANV; icanvas++)
    {
      canvName="c";  
      canvName+=icanvas;
      canv=(TCanvas *)gROOT->FindObject(canvName);
      if (canv)
	update_canv(canv);
    }
}

void clearAll()
{
  Long_t icanvas;
  TString canvName;
  TCanvas *canv;
  for (icanvas=1; icanvas <= GTPLOT_NCANV; icanvas++)
    {
      canvName="c";  
      canvName+=icanvas;
      canv=(TCanvas *)gROOT->FindObject(canvName);
      if (canv)
        {
	  canv->Clear();
	  update_canv(canv);
        }
    }
}

void gtplot()
{
  for (Int_t icanv=1; icanv<=GTPLOT_NCANV; icanv++)
    zoomin(icanv,300); 
  c1->SetWindowPosition(0,0);
  c1->SetBottomMargin(0.125);
  c1->SetLeftMargin(0.125);
  c2->SetWindowPosition(309,0); 
  c2->SetBottomMargin(0.125);
  c2->SetLeftMargin(0.125);
  c3->SetWindowPosition(618,0);
  c4->SetWindowPosition(0,350);
  c5->SetWindowPosition(309,350);
  c6->SetWindowPosition(618,350);
}

void toggle()
{ 
  Long_t icanvas;
  TString canvName;
  TCanvas *canv;
  for (icanvas=1; icanvas <= GTPLOT_NCANV; icanvas++)
    {
      canvName="c";  
      canvName+=icanvas;
      canv=(TCanvas *)gROOT->FindObject(canvName);
      if ((canv=(TCanvas *)gROOT->FindObject(canvName)))
	{
	  canv->Iconify();
	  update_canv(canv);
	}
    }
}



void bigTitle()
{
  gStyle->SetTitleFontSize(0.1);
}


void smallTitle()
{
  gStyle->SetTitleFontSize(0.05);
}



bool save_canv(TCanvas* canv, const char* fname = "plot.png", Int_t xysize=0)
{
  if(xysize)
    zoomin(canv,xysize);
  if(canv)
    {
      canv->SaveAs(fname);
      return true;
    }
  return false;
}

bool save_canv(Int_t icanvas=1, const char* fname = "plot.png", Int_t xysize=0)
{
  TString canvName = "";
  TCanvas *canv = 0;
  canvName.Form("c%d",icanvas);
  canv=(TCanvas *)gROOT->FindObject(canvName);
  return save_canv(canv,fname,xysize);
}

void save_plots(const char* basename="plots_", 
		const char* fExt=".png",
		Int_t xysize = 800)
{
  TString fname;
  Long_t icanvas;
  TString canvName;
  TCanvas *canv;
  for (icanvas=1; icanvas <= GTPLOT_NCANV; icanvas++)
    {
      canvName="c";  
      canvName+=icanvas;
      canv=(TCanvas *)gROOT->FindObject(canvName);
      if (canv)
        {
	  if(xysize)
	    zoomin(canv,xysize);
	  fname=basename;
	  fname+=canvName;
	  fname+=fExt;
	  canv->SaveAs(fname);
        }
    }
  gtplot();
}

void save_plots_tstamp(const char* outdir="./", 
                       const char* fExt=".png",
                       Int_t xysize = 800)
{
  
  Int_t yyyymmdd=p1.get_yymmdd()+2000*10000;
  Int_t hhmmss=p1.get_hhmmss();
  Int_t usec=p1.get_usec();
  TString outdir_s=outdir;
  outdir_s=outdir_s.Strip(TString::kTrailing,'/');
  TString bname;
  bname.Form("%s/%08d_%06d_%06d_",outdir_s.Data(),yyyymmdd,hhmmss,usec);
  save_plots(bname,fExt,xysize);
}

void sd_mode()
{
  clearAll();
  
  c1->Divide();
  set_grid(c1);
  c1->SetLogx(0); c1->SetLogy(0);
  
  c2->Divide();
  set_grid(c2);
  c2->SetLogx(0); c2->SetLogy(0);
    
  c3->Divide(1,2);
  for(Int_t i=1;i<=2;i++)
    {
      c3->cd(i);
      set_grid(gPad);
      gPad->SetLogx(0); gPad->SetLogy(0);
    }
  
  c4->Divide(1,3);
  for(Int_t i=1;i<=3;i++)
    {
      c4->cd(i);
      set_grid(gPad);
      gPad->SetLogx(0); gPad->SetLogy(1);
    }
  
  c5->Divide(1,2);
  for(Int_t i=1;i<=2;i++)
    {
      c5->cd(i); 
      set_grid(gPad);
      gPad->SetLogx(0); gPad->SetLogy(0);
    }
  
  c6->Divide(1,2);
  for(Int_t i=1;i<=2;i++)
    {
      c6->cd(i); 
      set_grid(gPad);
      gPad->SetLogx(0); gPad->SetLogy(0);
    }
  updateAll();
  curEdMode = ED_SD_MODE;
}

void fd_mode()
{  
  Long_t icanvas;
  TString canvName;
  TCanvas *canv;
  for (icanvas=1; icanvas <= GTPLOT_NCANV; icanvas++)
    {
      canvName="c";  
      canvName+=icanvas;
      canv=(TCanvas *)gROOT->FindObject(canvName);
      if (canv)
        {
	  canv->Clear();
	  set_grid(canv);
	  canv->SetLogy(0);
	  canv->SetLogx(0);
        }
    }
  updateAll();
  curEdMode = ED_FD_MODE;
}
void set_glob_style()
{
  gROOT->SetStyle("Plain");
  gStyle->SetLineWidth(2);
  gStyle->SetTitleFontSize(0.1);
  gStyle->SetOptFit(1);
  gStyle->SetPalette(1,0);
}



void SetUseGrid(Bool_t use_grid_flag = 1)
{
  use_grid = use_grid_flag;
  gtplot();
  if(curEdMode==ED_SD_MODE)
    sd_mode();
  if(curEdMode==ED_FD_MODE)
    fd_mode();
}
void SetUseTicks(Bool_t use_ticks_flag = 1)
{
  use_ticks = use_ticks_flag;
  if(curEdMode==ED_SD_MODE)
    sd_mode();
  if(curEdMode==ED_FD_MODE)
    fd_mode();
}



