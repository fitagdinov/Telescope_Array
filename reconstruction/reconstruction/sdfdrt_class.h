#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "event.h"
#include "sdanalysis_icc_settings.h"
#include "TNamed.h"
#ifndef _dstbank_class_h_
#define _dstbank_class_h_



// Macro for creating empty constructor and destructor implementations
// for DST bank classes
#define _dstbank_empty_constructor_destructor_(_dstbank_)	        \
  _dstbank_##_class::_dstbank_##_class() {;}				\
  _dstbank_##_class::~_dstbank_##_class() {;}
// Macro for creating functions that print message about the DST bank
// being not implemented
#define _dstbank_empty_functions_(_dstbank_)				\
  void _dstbank_##_class::loadFromDST()					\
  {									\
    fprintf(stderr,"^^^ WARNING: ");					\
    fprintf(stderr,"%s_class::loadFromDST(): ",#_dstbank_);		\
    fprintf(stderr,"dst2k-ta used in this build (%s) doesn't have ",	\
	    GetDSTDIR());						\
    fprintf(stderr,"%s source code\n",#_dstbank_);			\
  }									\
  void _dstbank_##_class::loadToDST()					\
  {									\
    fprintf(stderr,"^^^ WARNING: ");					\
    fprintf(stderr, "%s_class::loadToDST(): ",#_dstbank_);		\
    fprintf(stderr,"dst2k-ta used in this build (%s) doesn't have ",	\
	    GetDSTDIR());						\
    fprintf(stderr,"%s source code\n",#_dstbank_);			\
  }									\
  void _dstbank_##_class::clearOutDST()					\
  {									\
    fprintf(stderr,"^^^ WARNING: ");					\
    fprintf(stderr,"%s_class::clearOutDST(): ",#_dstbank_);		\
    fprintf(stderr,"dst2k-ta used in this build (%s) doesn't have ",	\
	    GetDSTDIR());						\
    fprintf(stderr,"%s source code\n",#_dstbank_);			\
  }
// Macro that provides functions for the DST classes for which corresponding
// DST banks are not available in the dst2k-ta that was used for linking
#define _dstbank_not_implemented_(_dstbank_)				\
  _dstbank_empty_constructor_destructor_(_dstbank_)			\
  _dstbank_empty_functions_(_dstbank_)

class dstbank_class : public TNamed
{
public:
  virtual ~dstbank_class();
  // classes that inherit from this class
  // will provide their data here
  virtual void loadFromDST();  // load the dst class from DST bank
  virtual void loadToDST();    // load the dst class into DST bank
  virtual void clearOutDST();  // clear out the class variables and the DST bank
  Int_t get_bank_id() const { return dstbank_id; }
  Int_t get_bank_version() const { return dstbank_version; }
  TString get_bank_name() const 
  { 
    return 
      TString(ClassName()).EndsWith("_class") ? 
      TString(ClassName(),strlen(ClassName())-6) : 
      TString(ClassName()) ;
  }
  void DumpBank(FILE *fp = stdout, Int_t format = 1); // dump the contents of the DST class
  static const char *GetDSTDIR(); // return the dst2k-ta directory against which this class was compiled
protected:
  Int_t dstbank_id;      // ID of the DST bank
  Int_t dstbank_version; // Version of the DST bank
  dstbank_class();
  dstbank_class(Int_t dstbank_id_val, Int_t dstbank_version_val);
  ClassDef(dstbank_class,3);
};


#endif
