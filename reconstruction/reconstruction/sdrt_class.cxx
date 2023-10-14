#include "sdrt_class.h"
// Root tree class for talex00 prototype DST bank.
// Last modified: Jan 15, 2020
// Dmitri Ivanov <dmiivanov@gmail.com>


ClassImp (talex00_class)

#ifndef TALEX00_BANKID
_dstbank_not_implemented_(talex00);
#else
talex00_class::talex00_class() : dstbank_class(TALEX00_BANKID,TALEX00_BANKVERSION) {;}
talex00_class::~talex00_class() {;}
void talex00_class::loadFromDST()
{
  event_num = talex00_.event_num;
  event_code = talex00_.event_code;
  site = talex00_.site;
  run_id.resize(TALEX00_NCT);
  trig_id.resize(TALEX00_NCT);
  for (Int_t i=0; i < TALEX00_NCT; i++)
    {
      run_id[i] = talex00_.run_id[i];
      trig_id[i] = talex00_.trig_id[i];
    }
  errcode = talex00_.errcode;
  yymmdd = talex00_.yymmdd;
  hhmmss = talex00_.hhmmss;
  usec = talex00_.usec;
  monyymmdd = talex00_.monyymmdd;
  monhhmmss = talex00_.monhhmmss;
  nofwf = talex00_.nofwf;
  nretry.resize(nofwf);
  wf_id.resize(nofwf);
  trig_code.resize(nofwf);
  xxyy.resize(nofwf);
  clkcnt.resize(nofwf);
  mclkcnt.resize(nofwf);
  fadcti.resize(nofwf);
  fadcav.resize(nofwf);
  fadc.resize(nofwf);
  pchmip.resize(nofwf);
  pchped.resize(nofwf);
  lhpchmip.resize(nofwf);
  lhpchped.resize(nofwf);
  rhpchmip.resize(nofwf);
  rhpchped.resize(nofwf);
  mftndof.resize(nofwf);
  mip.resize(nofwf);
  mftchi2.resize(nofwf);
  mftp.resize(nofwf);
  mftpe.resize(nofwf);
  lat_lon_alt.resize(nofwf);
  xyz_cor_clf.resize(nofwf);
  for (Int_t i=0; i < (Int_t)nretry.size(); i++)
    {
      nretry[i] = talex00_.nretry[i];
      wf_id[i] = talex00_.wf_id[i];
      trig_code[i] = talex00_.trig_code[i];
      xxyy[i] = talex00_.xxyy[i];
      clkcnt[i] = talex00_.clkcnt[i];
      mclkcnt[i] = talex00_.mclkcnt[i];
      fadcti[i].resize(2);
      fadcav[i].resize(2);
      fadc[i].resize(2);
      pchmip[i].resize(2);
      pchped[i].resize(2);
      lhpchmip[i].resize(2);
      lhpchped[i].resize(2);
      rhpchmip[i].resize(2);
      rhpchped[i].resize(2);
      mftndof[i].resize(2);
      mip[i].resize(2);
      mftchi2[i].resize(2);
      mftp[i].resize(2);
      mftpe[i].resize(2);
      lat_lon_alt[i].resize(3);
      xyz_cor_clf[i].resize(3);
      for (Int_t j=0; j < (Int_t)fadcti[i].size(); j++)
	{
	  fadcti[i][j] = talex00_.fadcti[i][j];
	  fadcav[i][j] = talex00_.fadcav[i][j];
	  fadc[i][j].resize(talex00_nchan_sd);
	  for (Int_t k=0; k < (Int_t)fadc[i][j].size(); k++)
	    fadc[i][j][k] = talex00_.fadc[i][j][k];
	  pchmip[i][j] = talex00_.pchmip[i][j];
	  pchped[i][j] = talex00_.pchped[i][j];
	  lhpchmip[i][j] = talex00_.lhpchmip[i][j];
	  lhpchped[i][j] = talex00_.lhpchped[i][j];
	  rhpchmip[i][j] = talex00_.rhpchmip[i][j];
	  rhpchped[i][j] = talex00_.rhpchped[i][j];
	  mftndof[i][j] = talex00_.mftndof[i][j];
	  mip[i][j] = talex00_.mip[i][j];
	  mftchi2[i][j] = talex00_.mftchi2[i][j];
	  mftp[i][j].resize(4);
	  mftpe[i][j].resize(4);
	  for (Int_t k=0; k < (Int_t)mftp[i][j].size(); k++)
	    {
	      mftp[i][j][k] = talex00_.mftp[i][j][k];
	      mftpe[i][j][k] = talex00_.mftpe[i][j][k];
	    }
	}
      for (Int_t j=0; j < (Int_t)lat_lon_alt[i].size(); j++)
	{
	  lat_lon_alt[i][j] = talex00_.lat_lon_alt[i][j];
	  xyz_cor_clf[i][j] = talex00_.xyz_cor_clf[i][j];
	}
    }
}

void talex00_class::loadToDST()
{
  talex00_.event_num  = event_num;
  talex00_.event_code = event_code ;
  talex00_.site = site;
  for (Int_t i=0; i < (Int_t)run_id.size() && i < TALEX00_NCT; i++)
    {
      talex00_.run_id[i]  = run_id[i];
      talex00_.trig_id[i] = trig_id[i];
    }
  talex00_.errcode = errcode;
  talex00_.yymmdd = yymmdd;
  talex00_.hhmmss = hhmmss;
  talex00_.usec = usec;
  talex00_.monyymmdd = monyymmdd;
  talex00_.monhhmmss = monhhmmss;
  talex00_.nofwf = nofwf;
  for (Int_t i=0; i < (Int_t)nretry.size(); i++)
    {
      talex00_.nretry[i] = nretry[i];
      talex00_.wf_id[i] = wf_id[i];
      talex00_.trig_code[i] = trig_code[i];
      talex00_.xxyy[i] = xxyy[i];
      talex00_.clkcnt[i] = clkcnt[i];
      talex00_.mclkcnt[i] = mclkcnt[i];
      for (Int_t j=0; j < (Int_t)fadcti[i].size(); j++)
	{
	  talex00_.fadcti[i][j] = fadcti[i][j];
	  talex00_.fadcav[i][j] = fadcav[i][j];
	  for (Int_t k=0; k < (Int_t)fadc[i][j].size(); k++)
	    talex00_.fadc[i][j][k] = fadc[i][j][k];
	  talex00_.pchmip[i][j] = pchmip[i][j];
	  talex00_.pchped[i][j] = pchped[i][j];
	  talex00_.lhpchmip[i][j] = lhpchmip[i][j];
	  talex00_.lhpchped[i][j] = lhpchped[i][j];
	  talex00_.rhpchmip[i][j] = rhpchmip[i][j];
	  talex00_.rhpchped[i][j] = rhpchped[i][j];
	  talex00_.mftndof[i][j] = mftndof[i][j];
	  talex00_.mip[i][j] = mip[i][j];
	  talex00_.mftchi2[i][j] = mftchi2[i][j];
	  for (Int_t k=0; k < (Int_t)mftp[i][j].size(); k++)
	    {
	      talex00_.mftp[i][j][k] = mftp[i][j][k];
	      talex00_.mftpe[i][j][k] = mftpe[i][j][k];
	    }
	}
      for (Int_t j=0; j < (Int_t)lat_lon_alt[i].size(); j++)
	{
	  talex00_.lat_lon_alt[i][j] = lat_lon_alt[i][j];
	  talex00_.xyz_cor_clf[i][j] = xyz_cor_clf[i][j];
	}
    }
}
void talex00_class::clearOutDST()
{
  memset(&talex00_,0,sizeof(talex00_dst_common));
  loadFromDST();
}
#endif

ClassImp (rusdraw_class)

#ifdef RUSDRAW_BANKID
rusdraw_class::rusdraw_class() : dstbank_class(RUSDRAW_BANKID,RUSDRAW_BANKVERSION) {;}
rusdraw_class::~rusdraw_class() {;}
void rusdraw_class::loadFromDST()
{
  event_num = rusdraw_.event_num;
  event_code = rusdraw_.event_code;
  site = rusdraw_.site;
  run_id.resize(3);
  trig_id.resize(3);
  for (Int_t i = 0; i < (Int_t)run_id.size(); i++)
    {
      run_id[i] = rusdraw_.run_id[i];
      trig_id[i] = rusdraw_.trig_id[i];
    }
  errcode = rusdraw_.errcode;
  yymmdd = rusdraw_.yymmdd;
  hhmmss = rusdraw_.hhmmss;
  usec = rusdraw_.usec;
  monyymmdd = rusdraw_.monyymmdd;
  monhhmmss = rusdraw_.monhhmmss;
  nofwf = rusdraw_.nofwf;
  nretry.resize(nofwf);
  wf_id.resize(nofwf);
  trig_code.resize(nofwf);
  xxyy.resize(nofwf);
  clkcnt.resize(nofwf);
  mclkcnt.resize(nofwf);
  fadcti.resize(nofwf);
  fadcav.resize(nofwf);
  fadc.resize(nofwf);
  pchmip.resize(nofwf);
  pchped.resize(nofwf);
  lhpchmip.resize(nofwf);
  lhpchped.resize(nofwf);
  rhpchmip.resize(nofwf);
  rhpchped.resize(nofwf);
  mftndof.resize(nofwf);
  mip.resize(nofwf);
  mftchi2.resize(nofwf);
  mftp.resize(nofwf);
  mftpe.resize(nofwf);
  for (Int_t i = 0; i < (Int_t)nretry.size(); i++)
    {
      nretry[i] = rusdraw_.nretry[i];
      wf_id[i] = rusdraw_.wf_id[i];
      trig_code[i] = rusdraw_.trig_code[i];
      xxyy[i] = rusdraw_.xxyy[i];
      clkcnt[i] = rusdraw_.clkcnt[i];
      mclkcnt[i] = rusdraw_.mclkcnt[i];
      fadcti[i].resize(2);
      fadcav[i].resize(2);
      fadc[i].resize(2);       
      pchmip[i].resize(2);
      pchped[i].resize(2);
      lhpchmip[i].resize(2);
      lhpchped[i].resize(2);
      rhpchmip[i].resize(2);
      rhpchped[i].resize(2);       
      mftndof[i].resize(2);
      mip[i].resize(2);
      mftchi2[i].resize(2);
      mftp[i].resize(2);
      mftpe[i].resize(2);
      for (Int_t j  = 0; j < (Int_t)fadcti[i].size(); j++)
	{
	  fadcti[i][j] = rusdraw_.fadcti[i][j];
	  fadcav[i][j] = rusdraw_.fadcav[i][j];
	  fadc[i][j].resize(rusdraw_nchan_sd);
	  for (Int_t k = 0; k < (Int_t)fadc[i][j].size(); k++)
	    fadc[i][j][k] = rusdraw_.fadc[i][j][k];
	  pchmip[i][j] = rusdraw_.pchmip[i][j];
	  pchped[i][j] = rusdraw_.pchped[i][j];
	  lhpchmip[i][j] = rusdraw_.lhpchmip[i][j];
	  lhpchped[i][j] = rusdraw_.lhpchped[i][j];
	  rhpchmip[i][j] = rusdraw_.rhpchmip[i][j];
	  rhpchped[i][j] = rusdraw_.rhpchped[i][j];           
	  mftndof[i][j] = rusdraw_.mftndof[i][j];
	  mip[i][j] = rusdraw_.mip[i][j];
	  mftchi2[i][j] = rusdraw_.mftchi2[i][j];
	  mftp[i][j].resize(4);
	  mftpe[i][j].resize(4);
	  for (Int_t k = 0; k < (Int_t)mftp[i][j].size(); k++)
	    {
	      mftp[i][j][k] = rusdraw_.mftp[i][j][k];
	      mftpe[i][j][k] = rusdraw_.mftpe[i][j][k];
	    }
	}
    }
}
void rusdraw_class::loadToDST()
{
  rusdraw_.event_num  = event_num;
  rusdraw_.event_code = event_code ;
  rusdraw_.site = site;
  for (Int_t i  = 0; i < (Int_t)run_id.size(); i++)
    {
      rusdraw_.run_id[i] = run_id[i];
      rusdraw_.trig_id[i] = trig_id[i];
    }
  rusdraw_.errcode = errcode;
  rusdraw_.yymmdd = yymmdd;
  rusdraw_.hhmmss = hhmmss;
  rusdraw_.usec = usec;
  rusdraw_.monyymmdd = monyymmdd;
  rusdraw_.monhhmmss = monhhmmss;
  rusdraw_.nofwf = nofwf;
  for (Int_t i = 0; i < (Int_t)nretry.size(); i++)
    {
      rusdraw_.nretry[i] = nretry[i];
      rusdraw_.wf_id[i] = wf_id[i];
      rusdraw_.trig_code[i] = trig_code[i];
      rusdraw_.xxyy[i] = xxyy[i];
      rusdraw_.clkcnt[i] = clkcnt[i];
      rusdraw_.mclkcnt[i] = mclkcnt[i];
      for (Int_t j = 0; j < (Int_t)fadcti[i].size(); j++)
	{
	  rusdraw_.fadcti[i][j] = fadcti[i][j];
	  rusdraw_.fadcav[i][j] = fadcav[i][j];
	  for (Int_t k  = 0; k < (Int_t)fadc[i][j].size(); k++)
	    rusdraw_.fadc[i][j][k] = fadc[i][j][k];
	  rusdraw_.pchmip[i][j] = pchmip[i][j];
	  rusdraw_.pchped[i][j] = pchped[i][j];
	  rusdraw_.lhpchmip[i][j] = lhpchmip[i][j];
	  rusdraw_.lhpchped[i][j] = lhpchped[i][j];
	  rusdraw_.rhpchmip[i][j] = rhpchmip[i][j];
	  rusdraw_.rhpchped[i][j] = rhpchped[i][j];
	  rusdraw_.mftndof[i][j] = mftndof[i][j];
	  rusdraw_.mip[i][j] = mip[i][j];
	  rusdraw_.mftchi2[i][j] = mftchi2[i][j];
	  for (Int_t k = 0; k < (Int_t)mftp[i][j].size(); k++)
	    {
	      rusdraw_.mftp[i][j][k] = mftp[i][j][k];
	      rusdraw_.mftpe[i][j][k] = mftpe[i][j][k];
	    }
	}
    }
}
void rusdraw_class::clearOutDST()
{
  memset(&rusdraw_,0,sizeof(rusdraw_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(rusdraw);
#endif

ClassImp (rusdmc_class)

#ifndef RUSDMC_BANKID
_dstbank_not_implemented_(rusdmc);
#else
rusdmc_class::rusdmc_class() : dstbank_class(RUSDMC_BANKID,RUSDMC_BANKVERSION) {;}
rusdmc_class::~rusdmc_class() {;}
void rusdmc_class::loadFromDST()
{
  event_num = rusdmc_.event_num;
  parttype = rusdmc_.parttype;
  corecounter = rusdmc_.corecounter;
  tc = rusdmc_.tc;
  energy = rusdmc_.energy;
  height = rusdmc_.height;
  theta = rusdmc_.theta;
  phi = rusdmc_.phi;
  corexyz.resize(3);
  for (Int_t i = 0; i < (Int_t)corexyz.size(); i++)
    corexyz[i] = rusdmc_.corexyz[i];
}
void rusdmc_class::loadToDST()
{
  rusdmc_.event_num = event_num;
  rusdmc_.parttype = parttype;
  rusdmc_.corecounter = corecounter;
  rusdmc_.tc = tc;
  rusdmc_.energy = energy;
  rusdmc_.height = height;
  rusdmc_.theta = theta;
  rusdmc_.phi = phi;
  for (Int_t i = 0; i < (Int_t)corexyz.size(); i++)
    rusdmc_.corexyz[i] = corexyz[i];
}
void rusdmc_class::clearOutDST()
{
  memset(&rusdmc_,0,sizeof(rusdmc_dst_common));
  loadFromDST();
}
#endif

ClassImp (rusdmc1_class)

#ifndef RUSDMC1_BANKID
_dstbank_not_implemented_(rusdmc1);
#else
rusdmc1_class::rusdmc1_class() : dstbank_class(RUSDMC1_BANKID,RUSDMC1_BANKVERSION) {;}
rusdmc1_class::~rusdmc1_class() {;}
void rusdmc1_class::loadFromDST()
{
  xcore = rusdmc1_.xcore;
  ycore = rusdmc1_.ycore;
  t0  = rusdmc1_.t0;
  bdist = rusdmc1_.bdist;
  tdistbr = rusdmc1_.tdistbr;
  tdistlr = rusdmc1_.tdistlr;
  tdistsk = rusdmc1_.tdistsk;
  tdist = rusdmc1_.tdist;
}
void rusdmc1_class::loadToDST()
{
  rusdmc1_.xcore = xcore;
  rusdmc1_.ycore = ycore;
  rusdmc1_.t0  = t0;
  rusdmc1_.bdist = bdist;
  rusdmc1_.tdistbr = tdistbr;
  rusdmc1_.tdistlr = tdistlr;
  rusdmc1_.tdistsk = tdistsk;
  rusdmc1_.tdist = tdist;
}
void rusdmc1_class::clearOutDST()
{
  memset(&rusdmc1_,0,sizeof(rusdmc1_dst_common));
  loadFromDST();
}
#endif

ClassImp (showlib_class)

#ifdef SHOWLIB_BANKID
showlib_class::showlib_class() : dstbank_class(SHOWLIB_BANKID,SHOWLIB_BANKVERSION)
{
  ;
}
showlib_class::~showlib_class()
{
  ;
}
void showlib_class::loadFromDST()
{
  code              =   showlib_.code;
  number            =   showlib_.number;
  angle             =   showlib_.angle;
  particle          =   showlib_.particle;
  energy            =   showlib_.energy;
  first             =   showlib_.first;
  nmax              =   showlib_.nmax;
  x0                =   showlib_.x0;
  xmax              =   showlib_.xmax;
  lambda            =   showlib_.lambda;
  chi2              =   showlib_.chi2;
}
void showlib_class::loadToDST()
{
  showlib_.code     =   code;
  showlib_.number   =   number;
  showlib_.angle    =   angle;
  showlib_.particle =   particle;
  showlib_.energy   =   energy;
  showlib_.first    =   first;
  showlib_.nmax     =   nmax;
  showlib_.x0       =   x0;
  showlib_.xmax     =   xmax;
  showlib_.lambda   =   lambda;
  showlib_.chi2     =   chi2;
}
void showlib_class::clearOutDST()
{
  memset(&showlib_,0,sizeof(showlib_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(showlib);
#endif

ClassImp (bsdinfo_class)


#ifdef BSDINFO_BANKID
bsdinfo_class::bsdinfo_class() : dstbank_class(BSDINFO_BANKID,BSDINFO_BANKVERSION)  {;}
bsdinfo_class::~bsdinfo_class() {;}
void bsdinfo_class::loadFromDST()
{
  yymmdd = bsdinfo_.yymmdd;
  hhmmss = bsdinfo_.hhmmss;
  usec = bsdinfo_.usec;
  nbsds = bsdinfo_.nbsds;
  xxyy.resize(nbsds);
  bitf.resize(nbsds);
  for (Int_t i=0; i<(Int_t)bitf.size(); i++)
    {
       xxyy[i] = bsdinfo_.xxyy[i];
       bitf[i] = bsdinfo_.bitf[i];
    }
#if BSDINFO_BANKVERSION >= 1
  nsdsout = bsdinfo_.nsdsout;
  xxyyout.resize(nsdsout);
  bitfout.resize(nsdsout);
  for (Int_t i=0; i<(Int_t)xxyyout.size(); i++)
    xxyyout[i] = bsdinfo_.xxyyout[i];
#if BSDINFO_BANKVERSION >= 2
  for (Int_t i=0; i<(Int_t)bitfout.size(); i++)
    bitfout[i] = bsdinfo_.bitfout[i];
#else
  for (Int_t i=0; i<(Int_t)bitfout.size(); i++)
    bitfout[i] = 0xFFFF;
#endif
#else
  nsdsout = 0;
  xxyyout.clear();
  bitfout.clear();
#endif



}

void bsdinfo_class::loadToDST()
{
  bsdinfo_.yymmdd = yymmdd;
  bsdinfo_.hhmmss = hhmmss;
  bsdinfo_.usec = usec;
  bsdinfo_.nbsds = nbsds;
  for (Int_t i=0; i<(Int_t)bitf.size(); i++)
    {
      bsdinfo_.xxyy[i] = xxyy[i];
      bsdinfo_.bitf[i] = bitf[i];
    }
#if BSDINFO_BANKVERSION >= 1
  bsdinfo_.nsdsout = nsdsout;
  for (Int_t i=0; i<(Int_t)xxyyout.size(); i++)
    bsdinfo_.xxyyout[i] = xxyyout[i];
#endif
#if BSDINFO_BANKVERSION >= 2
  for (Int_t i=0; i<(Int_t)bitfout.size(); i++)
    bsdinfo_.bitfout[i] = bitfout[i];
#endif
}

void bsdinfo_class::clearOutDST()
{
  memset(&bsdinfo_,0,sizeof(bsdinfo_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(bsdinfo);
#endif

ClassImp(sdtrgbk_class)

#ifndef SDTRGBK_BANKID
_dstbank_not_implemented_(sdtrgbk);
#else
sdtrgbk_class::sdtrgbk_class() : dstbank_class(SDTRGBK_BANKID,SDTRGBK_BANKVERSION) {;}
sdtrgbk_class::~sdtrgbk_class() {;}
void sdtrgbk_class::loadFromDST()
{
  raw_bankid = sdtrgbk_.raw_bankid;
  nsd = sdtrgbk_.nsd;
  n_bad_ped = sdtrgbk_.n_bad_ped;
  n_spat_cont = sdtrgbk_.n_spat_cont;
  n_isol = sdtrgbk_.n_isol;
  n_pot_st_cont = sdtrgbk_.n_pot_st_cont;
  n_l1_tg = sdtrgbk_.n_l1_tg;
  dec_ped = sdtrgbk_.dec_ped;
  inc_ped = sdtrgbk_.inc_ped;
  il2sd.resize(3);
  il2sd_sig.resize(3);
  for (Int_t i = 0; i < (Int_t)il2sd.size(); i++)
    {
      il2sd[i] = sdtrgbk_.il2sd[i];
      il2sd_sig[i] = sdtrgbk_.il2sd_sig[i];
    }
  trigp = sdtrgbk_.trigp;
  igevent = sdtrgbk_.igevent;
  secf.resize(nsd);
  tlim.resize(nsd);
  ich.resize(nsd);
  q.resize(nsd);
  l1sig_wfindex.resize(nsd);
  xxyy.resize(nsd);
  wfindex_cal.resize(nsd);
  nl1.resize(nsd);
  ig.resize(nsd);
  for (Int_t i = 0; i < (Int_t)secf.size(); i++)
    {
      nl1[i] = sdtrgbk_.nl1[i];
      secf[i].resize(nl1[i]);
      tlim[i].resize(2);
      for (Int_t j = 0; j < (Int_t)tlim[i].size(); j++)
        tlim[i][j] = sdtrgbk_.tlim[i][j];
      ich[i].resize(nl1[i]);
      q[i].resize(nl1[i]);
      l1sig_wfindex[i].resize(nl1[i]);
      xxyy[i] = sdtrgbk_.xxyy[i];
      wfindex_cal[i] = sdtrgbk_.wfindex_cal[i];
      ig[i] = sdtrgbk_.ig[i];
      for (Int_t j = 0; j < (Int_t)secf[i].size(); j++)
	{
	  secf[i][j] = sdtrgbk_.secf[i][j];
	  ich[i][j] = sdtrgbk_.ich[i][j];
	  q[i][j].resize(2);
	  for (Int_t k = 0; k < (Int_t)q[i][j].size(); k++)
	    q[i][j][k] = sdtrgbk_.q[i][j][k];
	  l1sig_wfindex[i][j] = sdtrgbk_.l1sig_wfindex[i][j];
	}
    }
}

void sdtrgbk_class::loadToDST()
{
  sdtrgbk_.raw_bankid = raw_bankid;
  sdtrgbk_.nsd = nsd;
  sdtrgbk_.n_bad_ped = n_bad_ped;
  sdtrgbk_.n_spat_cont = n_spat_cont;
  sdtrgbk_.n_isol = n_isol;
  sdtrgbk_.n_pot_st_cont = n_pot_st_cont;
  sdtrgbk_.n_l1_tg = n_l1_tg;
  sdtrgbk_.dec_ped = dec_ped;
  sdtrgbk_.inc_ped = inc_ped;
  for (Int_t i = 0; i < (Int_t)il2sd.size(); i++)
    {
      sdtrgbk_.il2sd[i] = il2sd[i];
      sdtrgbk_.il2sd_sig[i] = il2sd_sig[i];
    }
  sdtrgbk_.trigp = trigp;
  sdtrgbk_.igevent = igevent;
  for (Int_t i = 0; i < (Int_t)secf.size(); i++)
    {
      sdtrgbk_.nl1[i] = nl1[i];
      for (Int_t j = 0; j < (Int_t)tlim[i].size(); j++)
        sdtrgbk_.tlim[i][j] = tlim[i][j];
      sdtrgbk_.xxyy[i] = xxyy[i];
      sdtrgbk_.wfindex_cal[i] = wfindex_cal[i];
      sdtrgbk_.ig[i] = ig[i];
      for (Int_t j = 0; j < (Int_t)secf[i].size(); j++)
	{
	  sdtrgbk_.secf[i][j] = secf[i][j];
	  sdtrgbk_.ich[i][j] = ich[i][j];
	  for (Int_t k = 0; k < (Int_t)q[i][j].size(); k++)
	    sdtrgbk_.q[i][j][k] = q[i][j][k];
	  sdtrgbk_.l1sig_wfindex[i][j] = l1sig_wfindex[i][j];
	}
    }
}
void sdtrgbk_class::clearOutDST()
{
  memset(&sdtrgbk_,0,sizeof(sdtrgbk_dst_common));
  loadFromDST();
}
#endif

ClassImp(SDEventSubData_class)
SDEventSubData_class::SDEventSubData_class()
{
  ;
}
SDEventSubData_class::~SDEventSubData_class()
{
  ;
}
#ifdef TASDEVENT_BANKID
void SDEventSubData_class::loadFromDST(Int_t iwf)
{
  if (iwf  < 0 || iwf >= tasdevent_.num_trgwf)
    {
      fprintf(stderr, "SDEventSubData_class::loadFromDST: ");
      fprintf(stderr, "iwf must be in 0 .. %d range\n", tasdevent_.num_trgwf-1);
      return;
    }
  clock = tasdevent_.sub[iwf].clock;
  max_clock = tasdevent_.sub[iwf].max_clock;
  lid = tasdevent_.sub[iwf].lid;
  usum = tasdevent_.sub[iwf].usum;
  lsum = tasdevent_.sub[iwf].lsum;
  uavr = tasdevent_.sub[iwf].uavr;
  lavr = tasdevent_.sub[iwf].lavr;
  wf_id = tasdevent_.sub[iwf].wf_id;
  num_trgwf = tasdevent_.sub[iwf].num_trgwf;
  bank = tasdevent_.sub[iwf].bank;
  num_retry = tasdevent_.sub[iwf].num_retry;
  trig_code = tasdevent_.sub[iwf].trig_code;
  wf_error = tasdevent_.sub[iwf].wf_error;
  uwf.resize(tasdevent_nfadc);
  lwf.resize(tasdevent_nfadc);
  for (Int_t i = 0; i < (Int_t)uwf.size(); i++)
    {
      uwf[i] = tasdevent_.sub[iwf].uwf[i];
      lwf[i] = tasdevent_.sub[iwf].lwf[i];
    }
}
void SDEventSubData_class::loadToDST(Int_t iwf)
{
  if (iwf  < 0 || iwf >= tasdevent_.num_trgwf)
    {
      fprintf(stderr, "SDEventSubData_class::loadToDST: ");
      fprintf(stderr, "iwf must be in 0 .. %d range\n", tasdevent_.num_trgwf-1);
      return;
    }
  tasdevent_.sub[iwf].clock = clock;
  tasdevent_.sub[iwf].max_clock = max_clock;
  tasdevent_.sub[iwf].lid = lid;
  tasdevent_.sub[iwf].usum = usum;
  tasdevent_.sub[iwf].lsum = lsum;
  tasdevent_.sub[iwf].uavr = uavr;
  tasdevent_.sub[iwf].lavr = lavr;
  tasdevent_.sub[iwf].wf_id = wf_id;
  tasdevent_.sub[iwf].num_trgwf = num_trgwf;
  tasdevent_.sub[iwf].bank = bank;
  tasdevent_.sub[iwf].num_retry = num_retry;
  tasdevent_.sub[iwf].trig_code = trig_code;
  tasdevent_.sub[iwf].wf_error = wf_error;
  for (Int_t i = 0; i < (Int_t)uwf.size(); i++)
    {
      tasdevent_.sub[iwf].uwf[i] = uwf[i];
      tasdevent_.sub[iwf].lwf[i] = lwf[i];
    }  
}
#else
void SDEventSubData_class::loadFromDST(Int_t iwf)
{
}
void SDEventSubData_class::loadToDST(Int_t iwf)
{
}
#endif

ClassImp(tasdevent_class)


#ifdef TASDEVENT_BANKID
tasdevent_class::tasdevent_class() : dstbank_class(TASDEVENT_BANKID,TASDEVENT_BANKVERSION)
{
  event_code = 0;
  site = 0;
  date = 0;
  time = 0;
  num_trgwf = 0;
  num_wf = 0;
}
tasdevent_class::~tasdevent_class()
{
  ;
}
void tasdevent_class::loadFromDST()
{
  event_code = tasdevent_.event_code;
  run_id = tasdevent_.run_id;
  site = tasdevent_.site;
  trig_id = tasdevent_.trig_id;
  trig_code = tasdevent_.trig_code;
  code = tasdevent_.code;
  num_trgwf = tasdevent_.num_trgwf;
  num_wf = tasdevent_.num_wf;
  bank = tasdevent_.bank;
  date = tasdevent_.date;
  time = tasdevent_.time;
  date_org = tasdevent_.date_org;
  time_org = tasdevent_.time_org;
  usec = tasdevent_.usec;
  gps_error = tasdevent_.gps_error;
  pos = tasdevent_.pos;
  pattern.resize(16);
  for (Int_t i = 0; i < (Int_t)pattern.size(); i++)
    pattern[i] = tasdevent_.pattern[i];
  sub.resize(num_trgwf);
  for (Int_t i = 0; i < (Int_t)sub.size(); i++)
    sub[i].loadFromDST(i);
}
void tasdevent_class::loadToDST()
{
  tasdevent_.event_code = event_code;
  tasdevent_.run_id = run_id;
  tasdevent_.site = site;
  tasdevent_.trig_id = trig_id;
  tasdevent_.trig_code = trig_code;
  tasdevent_.code = code;
  tasdevent_.num_trgwf = num_trgwf;
  tasdevent_.num_wf = num_wf;
  tasdevent_.bank = bank;
  tasdevent_.date = date;
  tasdevent_.time = time;
  tasdevent_.date_org = date_org;
  tasdevent_.time_org = time_org;
  tasdevent_.usec = usec;
  tasdevent_.gps_error = gps_error;
  tasdevent_.pos = pos;
  for (Int_t i = 0; i < (Int_t)pattern.size(); i++)
    tasdevent_.pattern[i] = pattern[i];
  for (Int_t i  = 0; i < (Int_t) sub.size(); i++)
    sub[i].loadToDST(i);
}
void tasdevent_class::clearOutDST()
{
  memset(&tasdevent_,0,sizeof(tasdevent_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(tasdevent);
#endif

void tasdevent_class::trigp2xxyyt(Int_t trigp, Int_t *trigp_xxyy, Int_t *trigp_usec)
{
  (*trigp_xxyy) = (((trigp>>20)&0x3f)+(100*((trigp>>26)&0x3f)));
  (*trigp_usec) = (trigp&0xfffff);
}
void tasdevent_class::itrigp2xxyyt(Int_t itrigp, Int_t *trigp_xxyy, Int_t *trigp_usec)
{
  if (itrigp < 0 || itrigp > 15)
    {
      fprintf(stderr,"itrigp must be in 0 - 15 range\n");
      (*trigp_xxyy) = 0;
      (*trigp_usec) = 0;
      return;
    }
  trigp2xxyyt(pattern[itrigp],trigp_xxyy,trigp_usec);
}

// ADDED 20101023 - DI


ClassImp(SDCalibHostData_class)

SDCalibHostData_class::SDCalibHostData_class()
{
}
SDCalibHostData_class::~SDCalibHostData_class()
{
}

#ifdef TASDCALIB_BANKID

void SDCalibHostData_class::loadFromDST(Int_t ihost)
{
  SDCalibHostData* h = &tasdcalib_.host[ihost];
  site                     = h->site;
  numTrg                   = h->numTrg;
  trgBank.resize(numTrg);
  trgSec.resize(numTrg);
  trgPos.resize(numTrg);
  daqMode.resize(numTrg);
  for(Int_t i = 0; i < (Int_t)trgBank.size() ; i++)
    {
      trgBank[i]           = h->trgBank[i];
      trgSec[i]            = h->trgSec[i];
      trgPos[i]            = h->trgPos[i];
      daqMode[i]           = h->daqMode[i];
    }
  miss.resize(600);
  run_id.resize(600);
  for (Int_t i = 0; i < (Int_t)miss.size(); i++)
    {
      miss[i]              = h->miss[i];
      run_id[i]            = h->run_id[i];
    }
}

void SDCalibHostData_class::loadToDST(Int_t ihost)
{
  SDCalibHostData* h = &tasdcalib_.host[ihost];
  h->site                  = site;
  h->numTrg                = numTrg;
  for(Int_t i = 0; i<(Int_t)trgBank.size(); i++)
    {
      h->trgBank[i]        = trgBank[i];
      h->trgSec[i]         = trgSec[i];
      h->trgPos[i]         = trgPos[i];
      h->daqMode[i]        = daqMode[i];
    }
  for (Int_t i = 0; i < (Int_t)miss.size(); i++)
    {
      h->miss[i]           = miss[i];
      h->run_id[i]         = run_id[i];
    }
}

#else

void SDCalibHostData_class::loadFromDST(Int_t ihost)
{
}
void SDCalibHostData_class::loadToDST(Int_t ihost)
{ 
}

#endif




ClassImp(SDCalibSubData_class)

SDCalibSubData_class::SDCalibSubData_class()
{
}
SDCalibSubData_class::~SDCalibSubData_class()
{
}

#ifdef TASDCALIB_BANKID

void SDCalibSubData_class::loadFromDST(Int_t idet)
{
  SDCalibSubData* sub = &tasdcalib_.sub[idet]; 
  site                     = sub->site;
  lid                      = sub->lid;
  livetime                 = sub->livetime;
  warning                  = sub->warning;
  dontUse                  = sub->dontUse;
  dataQuality              = sub->dataQuality;
  gpsRunMode               = sub->gpsRunMode;
  miss.resize(75);
  for (Int_t i = 0; i < (Int_t)miss.size(); i++)
    miss[i]                = sub->miss[i];
  clockFreq                = sub->clockFreq;
  clockChirp               = sub->clockChirp;
  clockError               = sub->clockError;
  upedAvr                  = sub->upedAvr;
  lpedAvr                  = sub->lpedAvr;
  upedStdev                = sub->upedStdev;
  lpedStdev                = sub->lpedStdev;
  upedChisq                = sub->upedChisq;
  lpedChisq                = sub->lpedChisq;
  umipNonuni               = sub->umipNonuni;
  lmipNonuni               = sub->lmipNonuni;
  umipMev2cnt              = sub->umipMev2cnt;
  lmipMev2cnt              = sub->lmipMev2cnt;
  umipMev2pe               = sub->umipMev2pe;
  lmipMev2pe               = sub->lmipMev2pe;
  umipChisq                = sub->umipChisq;
  lmipChisq                = sub->lmipChisq;
  lvl0Rate                 = sub->lvl0Rate;
  lvl1Rate                 = sub->lvl1Rate;
  scinti_temp              = sub->scinti_temp;
  pchmip.resize(2);
  pchped.resize(2);
  lhpchmip.resize(2);
  lhpchped.resize(2);
  rhpchmip.resize(2);
  rhpchped.resize(2);
  mftndof.resize(2);
  mip.resize(2);
  mftchi2.resize(2);
  mftp.resize(2);
  mftpe.resize(2);
  for (Int_t i=0; i < (Int_t)pchmip.size(); i++)
    {
      pchmip[i]            = sub->pchmip[i];
      pchped[i]            = sub->pchped[i];
      lhpchmip[i]          = sub->lhpchmip[i];
      lhpchped[i]          = sub->lhpchped[i];
      rhpchmip[i]          = sub->rhpchmip[i];
      rhpchped[i]          = sub->rhpchped[i];
      mftndof[i]           = sub->mftndof[i];
      mip[i]               = sub->mip[i];
      mftchi2[i]           = sub->mftchi2[i];
      mftp[i].resize(4);
      mftpe[i].resize(4);
      for (Int_t j = 0; j < (Int_t)mftp[i].size(); j++)
	{
	  mftp[i][j]       = sub->mftp[i][j];
	  mftpe[i][j]      = sub->mftpe[i][j];
	}
    } 
}
void SDCalibSubData_class::loadToDST(Int_t idet)
{
  SDCalibSubData* sub = &tasdcalib_.sub[idet]; 
  sub->site                = site;
  sub->lid                 = lid;
  sub->livetime            = livetime;
  sub->warning             = warning;
  sub->dontUse             = dontUse;
  sub->dataQuality         = dataQuality;
  sub->gpsRunMode          = gpsRunMode;
  for(Int_t i = 0; i < (Int_t)miss.size(); i++)
    sub->miss[i]           = miss[i];
  sub->clockFreq           = clockFreq;
  sub->clockChirp          = clockChirp;
  sub->clockError          = clockError;
  sub->upedAvr             = upedAvr;
  sub->lpedAvr             = lpedAvr;
  sub->upedStdev           = upedStdev;
  sub->lpedStdev           = lpedStdev;
  sub->upedChisq           = upedChisq;
  sub->lpedChisq           = lpedChisq;
  sub->umipNonuni          = umipNonuni;
  sub->lmipNonuni          = lmipNonuni;
  sub->umipMev2cnt         = umipMev2cnt;
  sub->lmipMev2cnt         = lmipMev2cnt;
  sub->umipMev2pe          = umipMev2pe;
  sub->lmipMev2pe          = lmipMev2pe;
  sub->umipChisq           = umipChisq;
  sub->lmipChisq           = lmipChisq;
  sub->lvl0Rate            = lvl0Rate;
  sub->lvl1Rate            = lvl1Rate;
  sub->scinti_temp         = scinti_temp;
  for(Int_t i  = 0; i < (Int_t)pchmip.size(); i++)
    {
      sub->pchmip[i]       = pchmip[i];
      sub->pchped[i]       = pchped[i];
      sub->lhpchmip[i]     = lhpchmip[i];
      sub->lhpchped[i]     = lhpchped[i];
      sub->rhpchmip[i]     = rhpchmip[i];
      sub->rhpchped[i]     = rhpchped[i];
      sub->mftndof[i]      = mftndof[i];
      sub->mip[i] = mip[i];
      sub->mftchi2[i]      = mftchi2[i];
      for (Int_t j = 0; j < (Int_t)mftp[i].size(); j++)
	{
	  sub->mftp[i][j]  = mftp[i][j];
	  sub->mftpe[i][j] = mftpe[i][j];
	}
    }
}
#else
void SDCalibSubData_class::loadFromDST(Int_t idet)
{
}
void SDCalibSubData_class::loadToDST(Int_t idet)
{
}
#endif






ClassImp(SDCalibWeatherData_class)

SDCalibWeatherData_class::SDCalibWeatherData_class()
{
}
SDCalibWeatherData_class::~SDCalibWeatherData_class()
{
}

#ifdef TASDCALIB_BANKID

void SDCalibWeatherData_class::loadFromDST(Int_t iweat)
{
  SDCalibWeatherData* w    =  &tasdcalib_.weather[iweat]; 
  site                     =  w->site;
  averageWindSpeed         =  w->averageWindSpeed;
  maximumWindSpeed         =  w->maximumWindSpeed;
  windDirection            =  w->windDirection;
  atmosphericPressure      =  w->atmosphericPressure;
  temperature              =  w->temperature;
  humidity                 =  w->humidity;
  rainfall                 =  w->rainfall;
  numberOfHails            =  w->numberOfHails;
}
void SDCalibWeatherData_class::loadToDST(Int_t iweat)
{
  SDCalibWeatherData* w    =  &tasdcalib_.weather[iweat]; 
  w->site                  =  site;
  w->averageWindSpeed      =  averageWindSpeed;
  w->maximumWindSpeed      =  maximumWindSpeed;
  w->windDirection         =  windDirection;
  w->atmosphericPressure   =  atmosphericPressure;
  w->temperature           =  temperature;
  w->humidity              =  humidity;
  w->rainfall              =  rainfall;
  w->numberOfHails         =  numberOfHails;
}

#else

void SDCalibWeatherData_class::loadFromDST(Int_t iweat)
{
}
void SDCalibWeatherData_class::loadToDST(Int_t iweat)
{
}
  
#endif





ClassImp(tasdcalib_class)


#ifdef TASDCALIB_BANKID

tasdcalib_class::tasdcalib_class() : dstbank_class(TASDCALIB_BANKID,TASDCALIB_BANKVERSION)
{
  ;
}
tasdcalib_class::~tasdcalib_class()
{
  ;
}

void tasdcalib_class::loadFromDST()
{
  num_host                 = tasdcalib_.num_host;
  num_det                  = tasdcalib_.num_det;
  num_weather              = tasdcalib_.num_weather;
  date                     = tasdcalib_.date;
  time                     = tasdcalib_.time;
  trgMode.resize(600);
  for (Int_t i = 0; i < (Int_t)trgMode.size(); i++)
    trgMode[i]             = tasdcalib_.trgMode[i];
  host.resize(num_host);
  sub.resize(num_det);
  weather.resize(num_weather);
  for (Int_t i = 0; i < (Int_t)host.size(); i++)
    host[i].loadFromDST(i);
  for (Int_t i = 0; i < (Int_t)sub.size(); i++)
    sub[i].loadFromDST(i);
  for (Int_t i = 0; i < (Int_t)weather.size(); i++)
    weather[i].loadFromDST(i);
  footer                   = tasdcalib_.footer;
}

void tasdcalib_class::loadToDST()
{
  tasdcalib_.num_host      = num_host;
  tasdcalib_.num_det       = num_det;
  tasdcalib_.num_weather   = num_weather;
  tasdcalib_.date          = date;
  tasdcalib_.time          = time;
  for (Int_t i = 0; i < (Int_t)trgMode.size(); i++)
    tasdcalib_.trgMode[i]  = trgMode[i];
  for (Int_t i = 0; i < (Int_t)host.size(); i++)
    host[i].loadToDST(i);
  for (Int_t i = 0; i < (Int_t)sub.size(); i++)
    sub[i].loadToDST(i);
  for (Int_t i = 0; i < (Int_t)weather.size(); i++)
    weather[i].loadToDST(i);
  tasdcalib_.footer        = footer;
}

void tasdcalib_class::clearOutDST()
{
  memset(&tasdcalib_,0,sizeof(tasdcalib_dst_common));
  loadFromDST();
}

#else
_dstbank_not_implemented_(tasdcalib);
#endif

///////////////// Waveform Information ///////////////
ClassImp(SDCalibevData_class)

SDCalibevData_class::SDCalibevData_class()
{
}
SDCalibevData_class::~SDCalibevData_class()
{
}
#ifdef TASDCALIBEV_BANKID
void SDCalibevData_class::loadFromDST(Int_t iwf)
{
  if (iwf < 0 || iwf >= tasdcalibev_ndmax)
    return;
  const SDCalibevData& s = tasdcalibev_.sub[iwf];  
  site = s.site;
  lid = s.lid;
  clock = s.clock;
  maxClock = s.maxClock;
  wfId = s.wfId;
  numTrgwf = s.numTrgwf;
  trgCode = s.trgCode;
  wfError = s.wfError;
  uwf.resize(tasdcalibev_nfadc);
  lwf.resize(tasdcalibev_nfadc);
  for (Int_t i = 0; i < (Int_t)uwf.size(); i++)
    {
      uwf[i] = s.uwf[i];
      lwf[i] = s.lwf[i];
    }
  clockError = s.clockError;
  upedAvr = s.upedAvr;
  lpedAvr = s.lpedAvr;
  upedStdev = s.upedStdev;
  lpedStdev = s.lpedStdev;
  umipNonuni = s.umipNonuni;
  lmipNonuni = s.lmipNonuni;
  umipMev2cnt = s.umipMev2cnt;
  lmipMev2cnt = s.lmipMev2cnt;
  umipMev2pe = s.umipMev2pe;
  lmipMev2pe = s.lmipMev2pe;
  lvl0Rate = s.lvl0Rate;
  lvl1Rate = s.lvl1Rate;
  scintiTemp = s.scintiTemp;
  warning = s.warning;
  dontUse = s.dontUse;
  dataQuality = s.dataQuality;
  trgMode0 = s.trgMode0;
  trgMode1 = s.trgMode1;
  gpsRunMode = s.gpsRunMode;
  uthreLvl0 = s.uthreLvl0;
  lthreLvl0 = s.lthreLvl0;
  uthreLvl1 = s.uthreLvl1;
  lthreLvl1 = s.lthreLvl1;
  posX = s.posX;
  posY = s.posY;
  posZ = s.posZ;
  delayns = s.delayns;
  ppsofs = s.ppsofs;
  ppsflu = s.ppsflu;
  lonmas = s.lonmas;
  latmas = s.latmas;
  heicm = s.heicm;
  udec5pled = s.udec5pled;
  ldec5pled = s.ldec5pled;
  udec5pmip = s.udec5pmip;
  ldec5pmip = s.ldec5pmip;
  pchmip.resize(2);
  pchped.resize(2);
  lhpchmip.resize(2);
  lhpchped.resize(2);
  rhpchmip.resize(2);
  rhpchped.resize(2);
  mftndof.resize(2);
  mip.resize(2);
  mftchi2.resize(2);
  mftp.resize(2);
  mftpe.resize(2);
  for (Int_t i = 0; i < (Int_t)pchmip.size(); i++)
    {
      pchmip[i] = s.pchmip[i];
      pchped[i] = s.pchped[i];
      lhpchmip[i] = s.lhpchmip[i];
      lhpchped[i] = s.lhpchped[i];
      rhpchmip[i] = s.rhpchmip[i];
      rhpchped[i] = s.rhpchped[i];
      mftndof[i] = s.mftndof[i];
      mip[i] = s.mip[i];
      mftchi2[i] = s.mftchi2[i];
      mftp[i].resize(4);
      mftpe[i].resize(4);
      for (Int_t j = 0; j < (Int_t)mftp[i].size(); j++)
	{
	  mftp[i][j] = s.mftp[i][j];
	  mftpe[i][j] = s.mftpe[i][j];
	}
    }
}
void SDCalibevData_class::loadToDST(Int_t iwf)
{
  if (iwf < 0 || iwf >= tasdcalibev_ndmax)
    return;
  SDCalibevData& s = tasdcalibev_.sub[iwf];
  s.site = site;
  s.lid = lid;
  s.clock = clock;
  s.maxClock = maxClock;
  s.wfId = wfId;
  s.numTrgwf = numTrgwf;
  s.trgCode = trgCode;
  s.wfError = wfError;
  for (Int_t i = 0; i < (Int_t)uwf.size(); i++)
    {
      s.uwf[i] = uwf[i];
      s.lwf[i] = lwf[i];
    }
  s.clockError = clockError;
  s.upedAvr = upedAvr;
  s.lpedAvr = lpedAvr;
  s.upedStdev = upedStdev;
  s.lpedStdev = lpedStdev;
  s.umipNonuni = umipNonuni;
  s.lmipNonuni = lmipNonuni;
  s.umipMev2cnt = umipMev2cnt;
  s.lmipMev2cnt = lmipMev2cnt;
  s.umipMev2pe = umipMev2pe;
  s.lmipMev2pe = lmipMev2pe;
  s.lvl0Rate = lvl0Rate;
  s.lvl1Rate = lvl1Rate;
  s.scintiTemp = scintiTemp;
  s.warning = warning;
  s.dontUse = dontUse;
  s.dataQuality = dataQuality;
  s.trgMode0 = trgMode0;
  s.trgMode1 = trgMode1;
  s.gpsRunMode = gpsRunMode;
  s.uthreLvl0 = uthreLvl0;
  s.lthreLvl0 = lthreLvl0;
  s.uthreLvl1 = uthreLvl1;
  s.lthreLvl1 = lthreLvl1;
  s.posX = posX;
  s.posY = posY;
  s.posZ = posZ;
  s.delayns = delayns;
  s.ppsofs = ppsofs;
  s.ppsflu = ppsflu;
  s.lonmas = lonmas;
  s.latmas = latmas;
  s.heicm = heicm;
  s.udec5pled = udec5pled;
  s.ldec5pled = ldec5pled;
  s.udec5pmip = udec5pmip;
  s.ldec5pmip = ldec5pmip;  
  for (Int_t i = 0; i < (Int_t)pchmip.size(); i++)
    {
      s.pchmip[i] = pchmip[i];
      s.pchped[i] = pchped[i];
      s.lhpchmip[i] = lhpchmip[i];
      s.lhpchped[i] = lhpchped[i];
      s.rhpchmip[i] = rhpchmip[i];
      s.rhpchped[i] = rhpchped[i];
      s.mftndof[i] = mftndof[i];
      s.mip[i] = mip[i];
      s.mftchi2[i] = mftchi2[i];
      for (Int_t j = 0; j < (Int_t)mftp[i].size(); j++)
	{
	  s.mftp[i][j] = mftp[i][j];
	  s.mftpe[i][j] = mftpe[i][j];
	}
    }
}
#else
void SDCalibevData_class::loadFromDST(Int_t iwf)
{
}
void SDCalibevData_class::loadToDST(Int_t iwf)
{
}
#endif

///////////////// Weather Information ///////////////
ClassImp(SDCalibevWeatherData_class)

SDCalibevWeatherData_class::SDCalibevWeatherData_class()
{
}

SDCalibevWeatherData_class::~SDCalibevWeatherData_class()
{
}
#ifdef TASDCALIBEV_BANKID
void SDCalibevWeatherData_class::loadFromDST(Int_t iweat)
{
  if (iweat < 0 || iweat >= tasdcalibev_nwmax)
    return;
  const SDCalibevWeatherData& w = tasdcalibev_.weather[iweat];
  site                  =  w.site;
  atmosphericPressure   =  w.atmosphericPressure;
  temperature           =  w.temperature;
  humidity              =  w.humidity;
  rainfall              =  w.rainfall;
  numberOfHails         =  w.numberOfHails;
}
void SDCalibevWeatherData_class::loadToDST(Int_t iweat)
{
  if (iweat < 0 || iweat >= tasdcalibev_nwmax)
    return;
  SDCalibevWeatherData& w = tasdcalibev_.weather[iweat];
  w.site                =  site;
  w.atmosphericPressure =  atmosphericPressure;
  w.temperature         =  temperature;
  w.humidity            =  humidity;
  w.rainfall            =  rainfall;
  w.numberOfHails       =  numberOfHails;
}
#else
void SDCalibevWeatherData_class::loadFromDST(Int_t iweat)
{
}
void SDCalibevWeatherData_class::loadToDST(Int_t iweat)
{
}
#endif


///////////////// MC Information ///////////////
ClassImp(SDCalibevSimInfo_class)

SDCalibevSimInfo_class::SDCalibevSimInfo_class()
{
}
SDCalibevSimInfo_class::~SDCalibevSimInfo_class()
{
}
#ifdef TASDCALIBEV_BANKID
void SDCalibevSimInfo_class::loadFromDST()
{
  const SDCalibevSimInfo& s = tasdcalibev_.sim;
  interactionModel.resize(24);
  for(Int_t i = 0; i < (Int_t)interactionModel.size(); i++)
    interactionModel[i] = s.interactionModel[i];
  for(Int_t i = 0; i < (Int_t)primaryParticleType.size(); i++)
    primaryParticleType[i] = s.primaryParticleType[i];
  primaryEnergy                =   s.primaryEnergy;
  primaryCosZenith             =   s.primaryCosZenith;
  primaryAzimuth               =   s.primaryAzimuth;
  primaryFirstIntDepth         =   s.primaryFirstIntDepth;
  primaryArrivalTimeFromPps    =   s.primaryArrivalTimeFromPps;
  primaryCorePosX              =   s.primaryCorePosX;
  primaryCorePosY              =   s.primaryCorePosY;
  primaryCorePosZ              =   s.primaryCorePosZ;
  thinRatio                    =   s.thinRatio;
  maxWeight                    =   s.maxWeight;
  trgCode                      =   s.trgCode;
  userInfo                     =   s.userInfo;
  detailUserInfo.resize(10);
  for (Int_t i = 0; i < (Int_t)detailUserInfo.size(); i++)
    detailUserInfo[i] = s.detailUserInfo[i];
}
void SDCalibevSimInfo_class::loadToDST()
{  
  SDCalibevSimInfo& s = tasdcalibev_.sim;
  for(Int_t i = 0; i < (Int_t)interactionModel.size(); i++)
    s.interactionModel[i] = interactionModel[i];
  for(Int_t i = 0; i < (Int_t)primaryParticleType.size(); i++)
    s.primaryParticleType[i] = primaryParticleType[i];
  s.primaryEnergy             =   primaryEnergy;
  s.primaryCosZenith          =   primaryCosZenith;
  s.primaryAzimuth            =   primaryAzimuth;
  s.primaryFirstIntDepth      =   primaryFirstIntDepth;
  s.primaryArrivalTimeFromPps =   primaryArrivalTimeFromPps;
  s.primaryCorePosX           =   primaryCorePosX;
  s.primaryCorePosY           =   primaryCorePosY;
  s.primaryCorePosZ           =   primaryCorePosZ;
  s.thinRatio                 =   thinRatio;
  s.maxWeight                 =   maxWeight;
  s.trgCode                   =   trgCode;
  s.userInfo                  =   userInfo;
  for (Int_t i = 0; i < (Int_t)detailUserInfo.size(); i++)
    s.detailUserInfo[i] = detailUserInfo[i];
}
#else
void SDCalibevSimInfo_class::loadFromDST()
{
}
void SDCalibevSimInfo_class::loadToDST()
{
}
#endif


///////////////// TASDCALIBEV CLASS ///////////////////
ClassImp(tasdcalibev_class)

#ifdef TASDCALIBEV_BANKID
tasdcalibev_class::tasdcalibev_class() : dstbank_class(TASDCALIBEV_BANKID,TASDCALIBEV_BANKVERSION)
{
  numTrgwf   = 0;
  numWeather = 0;
  numAlive   = 0;
  numDead    = 0;
}
tasdcalibev_class::~tasdcalibev_class()
{
  numTrgwf   = 0;
  numWeather = 0;
  numAlive   = 0;
  numDead    = 0;
  sub.clear();
  weather.clear();
  aliveDetLid.clear();
  aliveDetSite.clear();
  aliveDetPosX.clear();
  aliveDetPosY.clear();
  aliveDetPosZ.clear();
  deadDetLid.clear();
  deadDetSite.clear();
  deadDetPosX.clear();
  deadDetPosY.clear();
  deadDetPosZ.clear();
}
void tasdcalibev_class::loadFromDST()
{
  const tasdcalibev_dst_common& t = tasdcalibev_;
  eventCode = t.eventCode;
  date = t.date;
  time = t.time;
  usec = t.usec;
  trgBank = t.trgBank;
  trgPos = t.trgPos;
  trgMode = t.trgMode;
  daqMode = t.daqMode;
  numWf = t.numWf;
  numTrgwf = t.numTrgwf;
  numWeather = t.numWeather;
  numAlive = t.numAlive;
  numDead = t.numDead;
  runId.resize(tasdcalibev_nhmax);
  daqMiss.resize(tasdcalibev_nhmax);
  for (Int_t i = 0; i < (Int_t)runId.size(); i++)
    {
      runId[i]   = t.runId[i];
      daqMiss[i] = t.daqMiss[i];
    }
  sub.resize(numTrgwf);
  for (Int_t i=0; i < (Int_t)sub.size(); i++)
    sub[i].loadFromDST(i);
  weather.resize(numWeather);
  for (Int_t i=0; i < (Int_t)weather.size(); i++)
    weather[i].loadFromDST(i);
  sim.loadFromDST();  
  aliveDetLid.resize(numAlive);
  aliveDetSite.resize(numAlive);
  aliveDetPosX.resize(numAlive);
  aliveDetPosY.resize(numAlive);
  aliveDetPosZ.resize(numAlive);  
  for (Int_t i=0; i < (Int_t)aliveDetLid.size(); i++)
    {
      aliveDetLid[i]  = t.aliveDetLid[i];
      aliveDetSite[i] = t.aliveDetSite[i];
      aliveDetPosX[i] = t.aliveDetPosX[i];
      aliveDetPosY[i] = t.aliveDetPosY[i];
      aliveDetPosZ[i] = t.aliveDetPosZ[i];
    }  
  deadDetLid.resize(numDead);
  deadDetSite.resize(numDead);
  deadDetPosX.resize(numDead);
  deadDetPosY.resize(numDead);
  deadDetPosZ.resize(numDead);  
  for (Int_t i=0; i < (Int_t)deadDetLid.size(); i++)
    {
      deadDetLid[i]  = t.deadDetLid[i];
      deadDetSite[i] = t.deadDetSite[i];
      deadDetPosX[i] = t.deadDetPosX[i];
      deadDetPosY[i] = t.deadDetPosY[i];
      deadDetPosZ[i] = t.deadDetPosZ[i];
    }
}
void tasdcalibev_class::loadToDST()
{
  tasdcalibev_dst_common& t = tasdcalibev_;  
  t.eventCode = eventCode;
  t.date = date;
  t.time = time;
  t.usec = usec;
  t.trgBank = trgBank;
  t.trgPos = trgPos;
  t.trgMode = trgMode;
  t.daqMode = daqMode;
  t.numWf = numWf;
  t.numTrgwf = numTrgwf;
  t.numWeather = numWeather;
  t.numAlive = numAlive;
  t.numDead = numDead;
  for (Int_t i = 0; i < (Int_t)runId.size(); i++)
    {
      t.runId[i]   = runId[i];
      t.daqMiss[i] = daqMiss[i];
    }
  for (Int_t i=0; i < (Int_t)sub.size(); i++)
    sub[i].loadToDST(i);
  for (Int_t i=0; i < (Int_t)weather.size(); i++)
    weather[i].loadToDST(i);
  sim.loadToDST();
  for (Int_t i=0; i < (Int_t)aliveDetLid.size(); i++)
    {
      t.aliveDetLid[i]  = aliveDetLid[i];
      t.aliveDetSite[i] = aliveDetSite[i];
      t.aliveDetPosX[i] = aliveDetPosX[i];
      t.aliveDetPosY[i] = aliveDetPosY[i];
      t.aliveDetPosZ[i] = aliveDetPosZ[i];
    }  
  for (Int_t i=0; i < (Int_t) deadDetLid.size(); i++)
    {
      t.deadDetLid[i]  = deadDetLid[i];
      t.deadDetSite[i] = deadDetSite[i];
      t.deadDetPosX[i] = deadDetPosX[i];
      t.deadDetPosY[i] = deadDetPosY[i];
      t.deadDetPosZ[i] = deadDetPosZ[i];
    }  
}
void tasdcalibev_class::clearOutDST()
{
  memset(&tasdcalibev_,0,sizeof(tasdcalibev_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(tasdcalibev);
#endif

using namespace std;

ClassImp (rufptn_class)

#ifndef RUFPTN_BANKID
_dstbank_not_implemented_(rufptn);
#else 
rufptn_class::rufptn_class() : dstbank_class(RUFPTN_BANKID,RUFPTN_BANKVERSION) {;}
rufptn_class::~rufptn_class() {;}
void rufptn_class::loadFromDST()
{ 
  nhits = rufptn_.nhits;
  nsclust = rufptn_.nsclust;
  nstclust = rufptn_.nstclust;
  nborder = rufptn_.nborder;   
  isgood.resize(nhits);
  wfindex.resize(nhits);
  xxyy.resize(nhits);
  nfold.resize(nhits);
  sstart.resize(nhits);
  sstop.resize(nhits);
  lderiv.resize(nhits);
  zderiv.resize(nhits);
  xyzclf.resize(nhits);
  tyro_cdist.resize(3);
  for (Int_t k = 0; k < (Int_t)tyro_cdist.size(); k++)
    {
      tyro_cdist[k].resize(nhits);
      for (Int_t i = 0; i < (Int_t)tyro_cdist[k].size(); i++)
	tyro_cdist[k][i]  = rufptn_.tyro_cdist[k][i];
    }
  reltime.resize(nhits);
  timeerr.resize(nhits);
  fadcpa.resize(nhits);
  fadcpaerr.resize(nhits);
  ped.resize(nhits);
  pederr.resize(nhits);
  pulsa.resize(nhits);
  pulsaerr.resize(nhits);
  vem.resize(nhits);
  vemerr.resize(nhits);
  for (Int_t i = 0; i < (Int_t)isgood.size(); i++)
    {
      isgood[i]  = rufptn_.isgood[i];
      wfindex[i] = rufptn_.wfindex[i];
      xxyy[i] = rufptn_.xxyy[i];
      nfold[i] = rufptn_.nfold[i];
      sstart[i].resize(2);
      sstop[i].resize(2);
      lderiv[i].resize(2);
      zderiv[i].resize(2);
      xyzclf[i].resize(3);
      for (Int_t j = 0; j < (Int_t)xyzclf[i].size(); j++)
	xyzclf[i][j] = rufptn_.xyzclf[i][j];
      reltime[i].resize(2);
      timeerr[i].resize(2);
      fadcpa[i].resize(2);
      fadcpaerr[i].resize(2);
      pulsa[i].resize(2);
      pulsaerr[i].resize(2);
      ped[i].resize(2);
      pederr[i].resize(2);
      vem[i].resize(2);
      vemerr[i].resize(2);
      for (Int_t k = 0; k < (Int_t)sstart[i].size(); k++)
	{
	  sstart[i][k] = rufptn_.sstart[i][k];
	  sstop[i][k] = rufptn_.sstop[i][k];
	  lderiv[i][k] = rufptn_.lderiv[i][k];
	  zderiv[i][k] = rufptn_.zderiv[i][k];
	  reltime[i][k] = rufptn_.reltime[i][k];
	  timeerr[i][k] = rufptn_.timeerr[i][k];
	  fadcpa[i][k] = rufptn_.fadcpa[i][k];
	  fadcpaerr[i][k] = rufptn_.fadcpaerr[i][k];
	  pulsa[i][k] = rufptn_.pulsa[i][k];
	  pulsaerr[i][k] = rufptn_.pulsaerr[i][k];
	  ped[i][k] = rufptn_.ped[i][k];
	  pederr[i][k] = rufptn_.pederr[i][k];
	  vem[i][k] = rufptn_.vem[i][k];
	  vemerr[i][k] = rufptn_.vemerr[i][k];
	}
    }
  qtot.resize(2);
  tearliest.resize(2);
  for (Int_t k = 0; k < (Int_t)qtot.size(); k++)
    {
      qtot[k] = rufptn_.qtot[k];
      tearliest[k] = rufptn_.tearliest[k];  
    }
  tyro_xymoments.resize(3);
  tyro_xypmoments.resize(3);
  tyro_u.resize(3);
  tyro_v.resize(3);
  tyro_tfitpars.resize(3);
  for (Int_t k = 0; k < (Int_t)tyro_xypmoments.size(); k++)
    {
      tyro_xymoments[k].resize(5);
      for (Int_t j = 0; j < (Int_t)tyro_xymoments[k].size(); j++)
	tyro_xymoments[k][j] = rufptn_.tyro_xymoments[k][j];
      tyro_xypmoments[k].resize(2);
      tyro_u[k].resize(2);
      tyro_v[k].resize(2);
      tyro_tfitpars[k].resize(2);
      for (Int_t j = 0; j < (Int_t)tyro_xypmoments[k].size(); j++)
	{
	  tyro_xypmoments[k][j] = rufptn_.tyro_xypmoments[k][j];
	  tyro_u[k][j] = rufptn_.tyro_u[k][j];
	  tyro_v[k][j] = rufptn_.tyro_v[k][j];
	  tyro_tfitpars[k][j] = rufptn_.tyro_tfitpars[k][j];
	}
    }
  tyro_chi2.resize(3);
  tyro_ndof.resize(3);
  tyro_theta.resize(3);
  tyro_phi.resize(3);
  for (Int_t k = 0; k < (Int_t)tyro_chi2.size(); k++)
    {
      tyro_chi2[k] = rufptn_.tyro_chi2[k];
      tyro_ndof[k] = rufptn_.tyro_ndof[k];
      tyro_theta[k] = rufptn_.tyro_theta[k];
      tyro_phi[k] = rufptn_.tyro_phi[k];
    }
}


void rufptn_class::loadToDST()
{
  rufptn_.nhits = nhits;
  rufptn_.nsclust = nsclust;
  rufptn_.nstclust = nstclust;
  rufptn_.nborder = nborder;   
  for (Int_t k = 0; k < (Int_t)tyro_cdist.size(); k++)
    for (Int_t i = 0; i < (Int_t)tyro_cdist[k].size(); i++)
      rufptn_.tyro_cdist[k][i] = tyro_cdist[k][i];
  for (Int_t i = 0; i < (Int_t)isgood.size(); i++)
    {
      rufptn_.isgood[i] = isgood[i];
      rufptn_.wfindex[i] = wfindex[i];
      rufptn_.xxyy[i] = xxyy[i];
      rufptn_.nfold[i] = nfold[i];
      for (Int_t j = 0; j < (Int_t)xyzclf[i].size(); j++)
	rufptn_.xyzclf[i][j] = xyzclf[i][j];
      for (Int_t k = 0; k < (Int_t)sstart[i].size(); k++)
	{
	  rufptn_.sstart[i][k] = sstart[i][k];
	  rufptn_.sstop[i][k] = sstop[i][k];
	  rufptn_.lderiv[i][k] = lderiv[i][k];
	  rufptn_.zderiv[i][k] = zderiv[i][k];
	  rufptn_.reltime[i][k] = reltime[i][k];
	  rufptn_.timeerr[i][k] = timeerr[i][k];
	  rufptn_.fadcpa[i][k] = fadcpa[i][k];
	  rufptn_.fadcpaerr[i][k] = fadcpaerr[i][k];
	  rufptn_.pulsa[i][k] = pulsa[i][k];
	  rufptn_.pulsaerr[i][k] = pulsaerr[i][k];
	  rufptn_.ped[i][k] = ped[i][k];
	  rufptn_.pederr[i][k] = pederr[i][k];
	  rufptn_.vem[i][k] = vem[i][k];
	  rufptn_.vemerr[i][k] = vemerr[i][k];
	}
    }
  for (Int_t k = 0; k < (Int_t)qtot.size(); k++)
    {
      rufptn_.qtot[k] = qtot[k];
      rufptn_.tearliest[k] = tearliest[k];  
    }
  for (Int_t k = 0; k < (Int_t)tyro_xypmoments.size(); k++)
    {
      for (Int_t j = 0; j < (Int_t)tyro_xymoments[k].size(); j++)
	rufptn_.tyro_xymoments[k][j] = tyro_xymoments[k][j];
      for (Int_t j = 0; j < (Int_t)tyro_xypmoments[k].size(); j++)
	{
	  rufptn_.tyro_xypmoments[k][j] = tyro_xypmoments[k][j];
	  rufptn_.tyro_u[k][j] = tyro_u[k][j];
	  rufptn_.tyro_v[k][j] = tyro_v[k][j];
	  rufptn_.tyro_tfitpars[k][j] = tyro_tfitpars[k][j];
	}
    }
  for (Int_t k = 0; k < (Int_t)tyro_chi2.size(); k++)
    {
      rufptn_.tyro_chi2[k] = tyro_chi2[k];
      rufptn_.tyro_ndof[k] = tyro_ndof[k];
      rufptn_.tyro_theta[k] = tyro_theta[k];
      rufptn_.tyro_phi[k] = tyro_phi[k];
    }
}

void rufptn_class::clearOutDST()
{
  memset(&rufptn_,0,sizeof(rufptn_dst_common));
  loadFromDST();
}
#endif

ClassImp (rusdgeom_class)

#ifndef RUSDGEOM_BANKID
_dstbank_not_implemented_(rusdgeom);
#else
rusdgeom_class::rusdgeom_class() : dstbank_class(RUSDGEOM_BANKID,RUSDGEOM_BANKVERSION) {;}
rusdgeom_class::~rusdgeom_class() {;}
void rusdgeom_class::loadFromDST()
{
  nsds=rusdgeom_.nsds;
  nsig.resize(nsds);
  sdsigq.resize(nsds);
  sdsigt.resize(nsds);
  sdsigte.resize(nsds);
  igsig.resize(nsds);
  irufptn.resize(nsds);
  xyzclf.resize(nsds);
  pulsa.resize(nsds);
  sdtime.resize(nsds);
  sdterr.resize(nsds);
  igsd.resize(nsds);
  xxyy.resize(nsds);
  sdirufptn.resize(nsds);
  for (Int_t i = 0; i < (Int_t)nsig.size(); i++)
    {
      nsig[i] = rusdgeom_.nsig[i];
      sdsigq[i].resize(nsig[i]);
      sdsigt[i].resize(nsig[i]);
      sdsigte[i].resize(nsig[i]);
      igsig[i].resize(nsig[i]);
      irufptn[i].resize(nsig[i]);
      for (Int_t j = 0; j < (Int_t)sdsigq[i].size(); j++)
	{
	  sdsigq[i][j] = rusdgeom_.sdsigq[i][j];
	  sdsigt[i][j] = rusdgeom_.sdsigt[i][j];
	  sdsigte[i][j] = rusdgeom_.sdsigte[i][j];
	  igsig[i][j] = rusdgeom_.igsig[i][j];
	  irufptn[i][j] = rusdgeom_.irufptn[i][j];
	}
      xyzclf[i].resize(3);
      for (Int_t j = 0; j < (Int_t)xyzclf[i].size(); j++)
	xyzclf[i][j] = rusdgeom_.xyzclf[i][j];
      pulsa[i] = rusdgeom_.pulsa[i];
      sdtime[i] = rusdgeom_.sdtime[i];
      sdterr[i] = rusdgeom_.sdterr[i];
      igsd[i] = rusdgeom_.igsd[i];
      xxyy[i] = rusdgeom_.xxyy[i];
      sdirufptn[i] = rusdgeom_.sdirufptn[i];
    }
  xcore.resize(3);
  dxcore.resize(3);
  ycore.resize(3);
  dycore.resize(3);
  t0.resize(3);
  dt0.resize(3);
  theta.resize(3);
  dtheta.resize(3);
  phi.resize(3);
  dphi.resize(3);
  chi2.resize(3);
  ndof.resize(3);
  for (Int_t i = 0; i < (Int_t) xcore.size(); i++)
    {
      xcore[i] = rusdgeom_.xcore[i];
      dxcore[i] = rusdgeom_.dxcore[i];
      ycore[i] = rusdgeom_.ycore[i];
      dycore[i] = rusdgeom_.dycore[i];
      t0[i] = rusdgeom_.t0[i];
      dt0[i] = rusdgeom_.dt0[i];
      theta[i] = rusdgeom_.theta[i];
      dtheta[i] = rusdgeom_.dtheta[i];
      phi[i] = rusdgeom_.phi[i];
      dphi[i] = rusdgeom_.dphi[i];
      chi2[i] = rusdgeom_.chi2[i];
      ndof[i] = rusdgeom_.ndof[i];
    }
  a  = rusdgeom_.a;
  da = rusdgeom_.da; 
  tearliest = rusdgeom_.tearliest;
}


void rusdgeom_class::loadToDST()
{
  rusdgeom_.nsds = nsds;
  for (Int_t i = 0; i < (Int_t)nsig.size(); i++)
    {
      rusdgeom_.nsig[i] = nsig[i];
      for (Int_t j = 0; j < (Int_t)sdsigq[i].size(); j++)
	{
	  rusdgeom_.sdsigq[i][j] = sdsigq[i][j];
	  rusdgeom_.sdsigt[i][j] = sdsigt[i][j];
	  rusdgeom_.sdsigte[i][j] = sdsigte[i][j];
	  rusdgeom_.igsig[i][j] = igsig[i][j];
	  rusdgeom_.irufptn[i][j] = irufptn[i][j];
	}
      for (Int_t j = 0; j < (Int_t)xyzclf[i].size(); j++)
	rusdgeom_.xyzclf[i][j] = xyzclf[i][j];
      rusdgeom_.pulsa[i] = pulsa[i];
      rusdgeom_.sdtime[i] = sdtime[i];
      rusdgeom_.sdterr[i] = sdterr[i];
      rusdgeom_.igsd[i] = igsd[i];
      rusdgeom_.xxyy[i] = xxyy[i];
      rusdgeom_.sdirufptn[i] = sdirufptn[i];
    }
  for (Int_t i = 0; i < (Int_t) xcore.size(); i++)
    {
      rusdgeom_.xcore[i] = xcore[i];
      rusdgeom_.dxcore[i] = dxcore[i];
      rusdgeom_.ycore[i] = ycore[i];
      rusdgeom_.dycore[i] = dycore[i];
      rusdgeom_.t0[i] = t0[i];
      rusdgeom_.dt0[i] = dt0[i];
      rusdgeom_.theta[i] = theta[i];
      rusdgeom_.dtheta[i] = dtheta[i];
      rusdgeom_.phi[i] = phi[i];
      rusdgeom_.dphi[i] = dphi[i];
      rusdgeom_.chi2[i] = chi2[i];
      rusdgeom_.ndof[i] = ndof[i];
    }
  rusdgeom_.a = a;
  rusdgeom_.da = da; 
  rusdgeom_.tearliest = tearliest;
}
void rusdgeom_class::clearOutDST()
{
  memset(&rusdgeom_,0,sizeof(rusdgeom_dst_common));
  loadFromDST();
}
#endif

ClassImp (rufldf_class)

rufldf_class::rufldf_class() : dstbank_class(RUFLDF_BANKID,RUFLDF_BANKVERSION)
{
}

rufldf_class::~rufldf_class()
{
}

void rufldf_class::loadFromDST()
{
  xcore.resize(2);
  dxcore.resize(2);
  ycore.resize(2);
  dycore.resize(2);
  sc.resize(2);
  dsc.resize(2);
  s600.resize(2);
  s600_0.resize(2);
  s800.resize(2);
  s800_0.resize(2);
  aenergy.resize(2);
  energy.resize(2);
  atmcor.resize(2);
  chi2.resize(2);
  ndof.resize(2);
  for (Int_t k = 0; k < (Int_t)xcore.size(); k++)
    {
      xcore[k] = rufldf_.xcore[k];
      dxcore[k] = rufldf_.dxcore[k];
      ycore[k] = rufldf_.ycore[k];
      dycore[k] = rufldf_.dycore[k];
      sc[k] = rufldf_.sc[k];
      dsc[k] = rufldf_.dsc[k];
      s600[k] = rufldf_.s600[k];
      s600_0[k] = rufldf_.s600_0[k];
      s800[k] = rufldf_.s800[k];
      s800_0[k] = rufldf_.s800_0[k];
      aenergy[k] = rufldf_.aenergy[k];
      energy[k] = rufldf_.energy[k];
#if RUFLDF_BANKVERSION >= 1
      atmcor[k] = rufldf_.atmcor[k];
#else
      atmcor[k] = 0.0;
#endif
      chi2[k] = rufldf_.chi2[k];
      ndof[k] = rufldf_.ndof[k];
    }
  theta=rufldf_.theta;
  dtheta=rufldf_.dtheta;
  phi=rufldf_.phi;
  dphi=rufldf_.dphi;
  t0=rufldf_.t0;
  dt0=rufldf_.dt0;
  bdist = rufldf_.bdist;
  tdistbr = rufldf_.tdistbr;
  tdistlr = rufldf_.tdistlr;
  tdistsk = rufldf_.tdistsk;
  tdist = rufldf_.tdist;
}
void rufldf_class::loadToDST()
{
  for (Int_t k = 0; k < (Int_t)xcore.size(); k++)
    {
      rufldf_.xcore[k] = xcore[k];
      rufldf_.dxcore[k] = dxcore[k];
      rufldf_.ycore[k] = ycore[k];
      rufldf_.dycore[k] = dycore[k];
      rufldf_.sc[k] = sc[k];
      rufldf_.dsc[k] = dsc[k];
      rufldf_.s600[k] = s600[k];
      rufldf_.s600_0[k] = s600_0[k];
      rufldf_.s800[k] = s800[k];
      rufldf_.s800_0[k] = s800_0[k];
      rufldf_.aenergy[k] = aenergy[k];
      rufldf_.energy[k] = energy[k];
#if RUFLDF_BANKVERSION >= 1
      rufldf_.atmcor[k] = atmcor[k];
#endif
      rufldf_.chi2[k] = chi2[k];
      rufldf_.ndof[k] = ndof[k];
    }
  
  rufldf_.theta=theta;
  rufldf_.dtheta=dtheta;
  rufldf_.phi=phi;
  rufldf_.dphi=dphi;
  rufldf_.t0=t0;
  rufldf_.dt0=dt0;
  rufldf_.bdist = bdist;
  rufldf_.tdistbr = tdistbr;
  rufldf_.tdistlr = tdistlr;
  rufldf_.tdistsk = tdistsk;
  rufldf_.tdist = tdist;
  
}

void rufldf_class::clearOutDST()
{
  memset(&rufldf_,0,sizeof(rufldf_dst_common));
  loadFromDST();
}
/**
   Root tree class for etrack DST bank.
   Last modified: Jan 15, 2020
   Dmitri Ivanov <dmiivanov@gmail.com>
**/


ClassImp (etrack_class)

#ifdef ETRACK_BANKID
etrack_class::etrack_class() : dstbank_class(ETRACK_BANKID,ETRACK_BANKVERSION) {;}
etrack_class::~etrack_class() {;}
void etrack_class::loadFromDST()
{
  energy = etrack_.energy;
  xmax = etrack_.xmax;
  theta = etrack_.theta;
  phi = etrack_.phi;
  t0 = etrack_.t0;
  xycore[0] = etrack_.xycore[0];
  xycore[1] = etrack_.xycore[1];
  nudata = etrack_.nudata;
  if(nudata < 0 || nudata > ETRACK_NUDATA)
    {
      fprintf(stderr,"^^^^^ warning: etrack_class::loadFromDST(): wrong nudata value %d, must be in 0-%d range\n",
              nudata,ETRACK_NUDATA);
      if(nudata < 0)
        nudata = 0;
      else
        nudata = ETRACK_NUDATA;
    }
  udata.resize(nudata);
  for (int i=0; i<(int)udata.size(); i++)
    udata[i] = etrack_.udata[i];
  yymmdd = etrack_.yymmdd;
  hhmmss = etrack_.hhmmss;
  qualct = etrack_.qualct;
}
void etrack_class::loadToDST()
{
  etrack_.energy = energy;
  etrack_.xmax = xmax;
  etrack_.theta = theta;
  etrack_.phi = phi;
  etrack_.t0 = t0;
  etrack_.xycore[0] = xycore[0];
  etrack_.xycore[1] = xycore[1];
  if(nudata < 0 || nudata > ETRACK_NUDATA)
    {
      fprintf(stderr,"^^^^^ warning: etrack_class::loadToDST(): wrong nudata value %d, must be in 0-%d range\n",
              nudata,ETRACK_NUDATA);
      if(nudata < 0)
        nudata = 0;
      else
        nudata = ETRACK_NUDATA;
    }
  etrack_.nudata = nudata;
  if((int)udata.size() < nudata)
    {
      fprintf(stderr,"^^^^^ warning: etrack_class::loadToDST(): udata.size()=%d is smaller than nudata=%d\n",
              (int)udata.size(),nudata);
      nudata = (int)udata.size();
    }
  for (int i=0; i < nudata; i++)
      etrack_.udata[i] = udata[i];

  etrack_.yymmdd = yymmdd;
  etrack_.hhmmss = hhmmss;
  etrack_.qualct = qualct;
}
void etrack_class::clearOutDST()
{
  memset(&etrack_,0,sizeof(etrack_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(etrack);
#endif

ClassImp (atmpar_class)

#ifdef ATMPAR_BANKID
atmpar_class::atmpar_class() : dstbank_class(ATMPAR_BANKID,ATMPAR_BANKVERSION)  {;}
atmpar_class::~atmpar_class() {;}
void atmpar_class::loadFromDST()
{  
  dateFrom = atmpar_.dateFrom;
  dateTo = atmpar_.dateTo;
  modelid = atmpar_.modelid;
  nh = atmpar_.nh; 
  h.resize(nh);
  a.resize(nh);
  b.resize(nh);
  c.resize(nh);
  for (Int_t i=0; i < (Int_t)h.size(); i++)
    {
      h[i] = atmpar_.h[i];
      a[i] = atmpar_.a[i];
      b[i] = atmpar_.b[i];
      c[i] = atmpar_.c[i];
    }
  chi2 = atmpar_.chi2;
  ndof = atmpar_.ndof;
}

void atmpar_class::loadToDST()
{
  atmpar_.dateFrom = dateFrom;
  atmpar_.dateTo = dateTo;
  atmpar_.modelid = modelid;
  for (Int_t i=0; i < (Int_t)h.size(); i++)
    {
      atmpar_.h[i] = h[i];
      atmpar_.a[i] = a[i];
      atmpar_.b[i] = b[i];
      atmpar_.c[i] = c[i];
    }
  atmpar_.chi2 = chi2;
  atmpar_.ndof = ndof;
}

void atmpar_class::clearOutDST()
{
  memset(&atmpar_,0,sizeof(atmpar_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(atmpar);
#endif
