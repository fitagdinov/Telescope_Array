//
// C++ PREPROCESSOR MACRO FOR dealing with DST CLASS BRANCHES IN pass1plot.h and pass1plot.cxx
//


// (1) IF INSIDE OF THE HEADER FILE (_PASS1PLOT_IMPLEMENTATION_ not defined)
//     DECLARING ROOT TREE BRANCHES FOR VARIOUS DST CLASSES
//     [branch]_class *[branch]; - declaring the class object for the dst class
//     Bool_t have_[branch];     - a boolean that tells if the DST branch is present
//     Note tha event if the branches are not present, they still get
//     allocated, although their arrays (mostly stl vectors) 
//     will contain no elements so they won't use much memory.
//     This is so that every dst class variable can be safely used by 
//     ROOT macros.
// (2) IF INSIDE OF THE IMPLEMENTATION FILE (_PASS1PLOT_IMPLEMENTATION_ defined)
//     Set the root tree branch addresses for the branches tha were found in the ROOT tree
//     and set the appropriate flags to true for the branches that were found

#ifndef _PASS1PLOT_IMPLEMENTATION_
#define PASS1PLOT_DST_BRANCH(branch)			\
  branch##_class *branch;				\
  Bool_t have_##branch;
#else
#define PASS1PLOT_DST_BRANCH(branch)			\
 have_##branch = false;					\
 branch = new branch##_class;				\
 if(pass1tree->GetBranch(#branch))			\
   {							\
     have_##branch = true;				\
     pass1tree->SetBranchAddress(#branch, &branch);	\
     dst_branches.push_back((dstbank_class**)&branch);	\
   }							\
 else							\
   branch->clearOutDST();
#endif

// (1) IF INSIDE OF THE HEADER FILE (_PASS1PLOT_IMPLEMENTATION_ not defined)
//     DECLARING THE BRANCH DST CLASS SELECTOR POINTER
// (2) IF INSIDE OF THE IMPLEMENTATION FILE (_PASS1PLOT_IMPLEMENTATION_ defined)
//     INITIALIZE THE BRANCH DST CLASS SELECTOR POINTER TO ZERO
#ifndef _PASS1PLOT_IMPLEMENTATION_
#define PASS1PLOT_DST_BRANCH_SELECTOR(selectorvariable)	\
  selectorvariable##_class *selectorvariable;
#else
#define PASS1PLOT_DST_BRANCH_SELECTOR(selectorvariable)	\
  selectorvariable = 0;
#endif

// include the DST branch lists, which
// will result in either declaration or implementation
// depending on whether _PASS1PLOT_IMPLEMENTATION_ is defined
// at the inclusion of pass1plot_branch_macro.h
#include "pass1plot_dst_branches.h"


#undef PASS1PLOT_DST_BRANCH
#undef PASS1PLOT_DST_BRANCH_SELECTOR
