PASS1PLOT_DST_BRANCH(talex00);            // talex00 (pass0) variables from a tree
PASS1PLOT_DST_BRANCH(rusdraw);            // rusdraw (pass0) variables from a tree
PASS1PLOT_DST_BRANCH(bsdinfo);            // information on counters that didn't work correctly
PASS1PLOT_DST_BRANCH(tasdcalibev);        // ICRR calibrated event bank
PASS1PLOT_DST_BRANCH(sdtrgbk);            // SD trigger backup variables from a tree
PASS1PLOT_DST_BRANCH(rusdmc);             // Monte-Carlo variables
PASS1PLOT_DST_BRANCH(rusdmc1);            // Calculated Monte-Carlo variables
PASS1PLOT_DST_BRANCH(rufptn);             // rufptn (pass1) variables from a tree
PASS1PLOT_DST_BRANCH(rusdgeom);           // rusdgeom (pass1) variables from a tree
PASS1PLOT_DST_BRANCH(rufldf);             // pass2 variables from the root tree
PASS1PLOT_DST_BRANCH_SELECTOR(fdplane);   // BR LR geometry generic for manipulation
PASS1PLOT_DST_BRANCH(brplane);            // BR FD geometry variables
PASS1PLOT_DST_BRANCH(lrplane);            // LR FD geometry variables
PASS1PLOT_DST_BRANCH_SELECTOR(fdprofile); // BR LR FD profile generic (for manipulations)
PASS1PLOT_DST_BRANCH(brprofile);          // BR FD profile variables
PASS1PLOT_DST_BRANCH(lrprofile);          // LR FD profile variables
PASS1PLOT_DST_BRANCH(hraw1);              // MD raw
PASS1PLOT_DST_BRANCH(mcraw);              // MD mc raw
PASS1PLOT_DST_BRANCH(mc04);               // MD mc thrown
PASS1PLOT_DST_BRANCH(hbar);               // MD calibration
PASS1PLOT_DST_BRANCH(stps2);              // MD pass2 class
PASS1PLOT_DST_BRANCH(stpln);              // MD plane fit
PASS1PLOT_DST_BRANCH(hctim);              // MD time fit
PASS1PLOT_DST_BRANCH(hcbin);              // MD profile fit bins
PASS1PLOT_DST_BRANCH(prfc);               // MD profile fit results

// special case variables that are also related to the presence of ROOT tree branches
#ifndef _PASS1PLOT_IMPLEMENTATION_
Bool_t haveMC;                            // Indicate the presence of both rusdmc and rusdmc1
Bool_t have_fdplane[2];                   // Indicate the presence of fdplane branches   ( [0] BR, [1] LR )
Bool_t have_fdprofile[2];                 // Indicate the presence of fdprofile branches ( [0] BR, [1] LR )
#else
haveMC = (have_rusdmc && have_rusdmc1);   // Indicates whether all needed SD MC branches are present
have_fdplane[0] = have_brplane;           // Indicates whether a particular (BR=0 or LR=1) geometry branch is present
have_fdplane[1] = have_lrplane;           // Indicates whether a particular (BR=0 or LR=1) geometry branch is present
have_fdprofile[0] = have_brprofile;       // Indicates whether a particular (BR=0 or LR=1) profile branch is present
have_fdprofile[1] = have_lrprofile;       // Indicates whether a particular (BR=0 or LR=1) profile branch is present
#endif
