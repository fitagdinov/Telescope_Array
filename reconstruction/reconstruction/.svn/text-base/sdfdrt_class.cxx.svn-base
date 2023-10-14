#include "sdfdrt_class.h"

ClassImp(dstbank_class);

dstbank_class::dstbank_class(): dstbank_id(0), dstbank_version(0)
{
  ;
}
dstbank_class::dstbank_class(Int_t dstbank_id_val, Int_t dstbank_version_val): 
  dstbank_id(dstbank_id_val), dstbank_version(dstbank_version_val)
{
  ;
}
dstbank_class::~dstbank_class()
{
  ;
}
void dstbank_class::loadFromDST()
{
  ;
}
void dstbank_class::loadToDST()
{
  ;
}
void dstbank_class::clearOutDST()
{
  ;
}
void dstbank_class::DumpBank(FILE *fp, Int_t format)
{
  if(!dstbank_id)
    return;
  loadToDST();
  Int_t bank_list = newBankList(2);
  addBankList(bank_list,dstbank_id);
  eventSetDumpFormat(bank_list,format);
  eventDumpf(fp,bank_list);
  clrBankList(bank_list);
  delBankList(bank_list);
}

const char* dstbank_class::GetDSTDIR()
{
#ifndef __DSTDIR__
#define __DSTDIR__ "UNKNOWN"
#endif
  return __DSTDIR__;
}
