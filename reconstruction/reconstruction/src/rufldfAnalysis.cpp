#include "rufldfAnalysis.h"
#include "sdparamborder.h"
#include "sdenergy.h"

using namespace TMath;


rufldfAnalysis::rufldfAnalysis(listOfOpt& passed_opt) : opt(passed_opt)
  {
    p2geomfitter = new p2geomfitter_class();
    p2ldffitter = new p2ldffitter_class();
    p2gldffitter = new p2gldffitter_class();
    sdcoorclf = new sdxyzclf_class();
  }

rufldfAnalysis::~rufldfAnalysis()
  {
  }

/////////////////////// All analysis routines called from here ///////////////////////
void rufldfAnalysis::analyzeEvent()
  {
    // If (applicable to MC) this is a no-trigger event,
    // then writes zeros to geom. and ldf banks.
    if (rusdraw_.nofwf == 0 || rufptn_.nhits == 0)
      {
        memset(&rufptn_, 0, sizeof(rufptn_));
        memset(&rusdgeom_, 0, sizeof(rusdgeom_));
        memset(&rufldf_, 0, sizeof(rufldf_));
      }
    else
      {
        redo_pass1geom(); // refit pass1 geometry
        ldfFit(); // fit LDF alone
        gldfFit(); // fit Geom/LDF combined
        addCoreBorderInfo(); // Computes reconstructed core distances from the border and records them into pass2 DST.
      }
    

  }

void rufldfAnalysis::redo_pass1geom()
  {
    refit_pass1geom();
    changeStclustAgr();
    put2rusdgeomAgr();
  }

bool rufldfAnalysis::refit_pass1geom()
  {
    integer4 i;

    // Load variables into the fitter
    p2geomfitter->loadVariables_rufptn();

    // Clean space-time cluster using  modified Linsley's fit.
    p2geomfitter->cleanClust(P2GEOM_DCHI2);

    // Don't do any further fitting if # of good points is less than 3
    if (p2geomfitter->ngpts < 3)
      {
        for (i=0; i<3; i++)
          {
            rusdgeom_.xcore[i] = 0.0;
            rusdgeom_.dxcore[i] = 0.0;

            rusdgeom_.ycore[i] = 0.0;
            rusdgeom_.dycore[i] = 0.0;

            rusdgeom_.t0[i] = 0.0;
            rusdgeom_.dt0[i] = 0.0;

            rusdgeom_.theta[i] = 0.0;
            rusdgeom_.dtheta[i] = 0.0;

            rusdgeom_.phi[i] = 0.0;
            rusdgeom_.dphi[i] = 0.0;

            rusdgeom_.chi2[i] = 1.e6;
            rusdgeom_.ndof[i] = p2geomfitter->ndof;
          }
        return false;
      }

    // Fitting into modified Linsley
    p2geomfitter->doFit(1);
    if (p2geomfitter->chi2 > 1.e6)
      p2geomfitter->chi2 = 1.e6;
    rusdgeom_.ndof[1] = p2geomfitter->ndof;

    rusdgeom_.xcore[1] = p2geomfitter->R[0]; // Core X position
    rusdgeom_.dxcore[1] = p2geomfitter->dR[0];

    rusdgeom_.ycore[1] = p2geomfitter->R[1]; // Core Y position
    rusdgeom_.dycore[1] = p2geomfitter->dR[1];

    rusdgeom_.t0[1] = p2geomfitter->T0; // Time of the core hit
    rusdgeom_.dt0[1] = p2geomfitter->dT0;

    rusdgeom_.theta[1] = p2geomfitter->theta; // Zenith angle
    rusdgeom_.dtheta[1] = p2geomfitter->dtheta;

    rusdgeom_.phi[1] = p2geomfitter->phi; // Azimuthal angle
    rusdgeom_.dphi[1] = p2geomfitter->dphi;

    rusdgeom_.chi2[1] = p2geomfitter->chi2; // Chi2
    rusdgeom_.ndof[1] = p2geomfitter->ndof; // # of degrees of freedom
    if(!p2geomfitter->hasConverged())
      rusdgeom_.chi2[1]=1.e6;

    // Plane fitting.
    p2geomfitter->doFit(0);
    if (p2geomfitter->chi2 > 1.e6)
      p2geomfitter->chi2 = 1.e6;
    rusdgeom_.xcore[0] = p2geomfitter->R[0]; // Core X position
    rusdgeom_.dxcore[0] = p2geomfitter->dR[0];

    rusdgeom_.ycore[0] = p2geomfitter->R[1]; // Core Y position
    rusdgeom_.dycore[0] = p2geomfitter->dR[1];

    rusdgeom_.t0[0] = p2geomfitter->T0; // Time of the core hit
    rusdgeom_.dt0[0] = p2geomfitter->dT0;

    rusdgeom_.theta[0] = p2geomfitter->theta; // Zenith angle
    rusdgeom_.dtheta[0] = p2geomfitter->dtheta;

    rusdgeom_.phi[0] = p2geomfitter->phi; // Azimuthal angle
    rusdgeom_.dphi[0] = p2geomfitter->dphi;

    rusdgeom_.chi2[0] = p2geomfitter->chi2; // Chi2
    rusdgeom_.ndof[0] = p2geomfitter->ndof; // # of degrees of freedom

    if(!p2geomfitter->hasConverged())
      rusdgeom_.chi2[0]=1.e6;
    // Final fit variables (Linsley's with variable curvature)
    
    p2geomfitter->doFit(3);
    if (p2geomfitter->chi2 > 1.e6)
      p2geomfitter->chi2 = 1.e6;
    rusdgeom_.xcore[2] = p2geomfitter->R[0]; // Core X position
    rusdgeom_.dxcore[2] = p2geomfitter->dR[0];

    rusdgeom_.ycore[2] = p2geomfitter->R[1]; // Core Y position
    rusdgeom_.dycore[2] = p2geomfitter->dR[1];

    rusdgeom_.t0[2] = p2geomfitter->T0; // Time of the core hit
    rusdgeom_.dt0[2] = p2geomfitter->dT0;

    rusdgeom_.theta[2] = p2geomfitter->theta; // Zenith angle
    rusdgeom_.dtheta[2] = p2geomfitter->dtheta;

    rusdgeom_.phi[2] = p2geomfitter->phi; // Azimuthal angle
    rusdgeom_.dphi[2] = p2geomfitter->dphi;

    rusdgeom_.a = p2geomfitter->a;         // Curvature parameter
    rusdgeom_.da = p2geomfitter->da;
    
    rusdgeom_.chi2[2] = p2geomfitter->chi2; // Chi2
    rusdgeom_.ndof[2] = p2geomfitter->ndof; // # of degrees of freedom
    if(!p2geomfitter->hasConverged())
      rusdgeom_.chi2[2]=1.e6;
    return true;
  }

void rufldfAnalysis::changeStclustAgr()
  {
    integer4 ipoint;
    integer4 ihit;

    // To keep track of signals that are saturated. We don't
    // want to re-do the saturation analysis here.
    integer4 old_isgood[RUFPTNMH];

    memcpy(old_isgood, rufptn_.isgood, RUFPTNMH*sizeof(integer4));
    for (ihit=0; ihit < rufptn_.nhits; ihit++)
      {
        if (rufptn_.isgood[ihit] > 3)
          {
            rufptn_.isgood[ihit] = 3;
            rufptn_.nstclust -= 1;
          }

      }
    for (ipoint=0; ipoint<p2geomfitter->ngpts; ipoint++)
      {
        ihit=p2geomfitter->goodpts[ipoint];
        rufptn_.isgood[ihit] = 4;
        rufptn_.nstclust += 1;
      }

    // Label the saturated counters (this information is stored in a special array)
    for (ihit=0; ihit<rufptn_.nhits; ihit++)
      {
        if ((rufptn_.isgood[ihit] == 4) && (old_isgood[ihit] == 5))
          rufptn_.isgood[ihit] = 5;
      }

  }

void rufldfAnalysis::put2rusdgeomAgr()
  {
    integer4 ic, ih, is;

    ic = 0;
    for (ih=0; ih<rufptn_.nhits; ih++)
      {
        if ((ic>0) && (rufptn_.xxyy[ih]==rusdgeom_.xxyy[ic-1]))
          {
            rusdgeom_.irufptn[ic-1][rusdgeom_.nsig[ic-1]]=ih;
            rusdgeom_.sdsigq[ic-1][rusdgeom_.nsig[ic-1]]= 0.5
                *(rufptn_.pulsa[ih][0]+rufptn_.pulsa[ih][1]);
            rusdgeom_.sdsigt[ic-1][rusdgeom_.nsig[ic-1]]= 0.5
                *(rufptn_.reltime[ih][0]+rufptn_.reltime[ih][1]);
            rusdgeom_.sdsigte[ic-1][rusdgeom_.nsig[ic-1]]= 0.5
                *sqrt(rufptn_.timeerr[ih][0]*rufptn_.timeerr[ih][0]
                    + rufptn_.timeerr[ih][1]*rufptn_.timeerr[ih][1]);
            rusdgeom_.igsig[ic-1][rusdgeom_.nsig[ic-1]]=rufptn_.isgood[ih];
            rusdgeom_.nsig[ic-1]++;
          }
        else
          {
            rusdgeom_.igsd[ic]=1;
            rusdgeom_.xxyy[ic]=rufptn_.xxyy[ih];
            memcpy(rusdgeom_.xyzclf[ic], rufptn_.xyzclf[ih], (integer4)(3
                *sizeof(real8)));
            rusdgeom_.irufptn[ic][0]=ih;
            rusdgeom_.sdsigq[ic][0]= 0.5*(rufptn_.pulsa[ih][0]
                +rufptn_.pulsa[ih][1]);
            rusdgeom_.sdsigt[ic][0]= 0.5*(rufptn_.reltime[ih][0]
                +rufptn_.reltime[ih][1]);
            rusdgeom_.sdsigte[ic][0]= 0.5*sqrt(rufptn_.timeerr[ih][0]
                *rufptn_.timeerr[ih][0]+ rufptn_.timeerr[ih][1]
                *rufptn_.timeerr[ih][1]);
            rusdgeom_.igsig[ic][0]=rufptn_.isgood[ih];
            rusdgeom_.nsig[ic]=1;
            ic++;
          }
        // Hit is a part of space-time cluster, and passed the chi2-cleaning procedure
        if (rufptn_.isgood[ih]==4)
          {
            rusdgeom_.igsd[ic-1] = 2;
          }
        // Label the saturated counters
        if (rufptn_.isgood[ih]==5)
          {
            rusdgeom_.igsd[ic-1] = 3;
          }
        // Bad counter
        if (rufptn_.isgood[ih]==0)
          {
            rusdgeom_.igsd[ic-1] = 0;
          }
      }

    // Use good signal information for counters
    rusdgeom_.nsds=ic;
    for (ic=0; ic<rusdgeom_.nsds; ic++)
      {
        rusdgeom_.pulsa[ic]=rusdgeom_.sdsigq[ic][0];
        rusdgeom_.sdtime[ic]=rusdgeom_.sdsigt[ic][0];
        rusdgeom_.sdterr[ic]=rusdgeom_.sdsigte[ic][0];
        rusdgeom_.sdirufptn[ic]=rusdgeom_.irufptn[ic][0];
        for (is=0; is<rusdgeom_.nsig[ic]; is++)
          {
            if (rusdgeom_.igsig[ic][is]>=4)
              {
                rusdgeom_.pulsa[ic]=rusdgeom_.sdsigq[ic][is];
                rusdgeom_.sdtime[ic]=rusdgeom_.sdsigt[ic][is];
                rusdgeom_.sdterr[ic]=rusdgeom_.sdsigte[ic][is];
                rusdgeom_.sdirufptn[ic] = rusdgeom_.irufptn[ic][is];
                break;
              }
          }
      }

    // Earliest time in the event readout
    rusdgeom_.tearliest = 0.5*(rufptn_.tearliest[0]+rufptn_.tearliest[1]);
  }

void rufldfAnalysis::ldfFit()
  {
    p2ldffitter->loadVariables();

    // Fit with core poisiton allowed to vary
    p2ldffitter->doFit(false);
    if (p2ldffitter->chi2 > 1.e6)
      p2ldffitter->chi2 = 1.e6;
    // Put variables into LDF DST bank  
    rufldf_.xcore[0] = p2ldffitter->R[0];
    rufldf_.dxcore[0] = p2ldffitter->dR[0];
    rufldf_.ycore[0] = p2ldffitter->R[1];
    rufldf_.dycore[0] = p2ldffitter->dR[1];
    rufldf_.sc[0] = p2ldffitter->S;
    rufldf_.dsc[0] = p2ldffitter->dS;
    rufldf_.s600[0] = p2ldffitter->S600;
    rufldf_.s600_0[0] = p2ldffitter->S600_0;
    rufldf_.s800[0] = p2ldffitter->S800;
    rufldf_.s800_0[0] = 0.0; // Not available yet
    rufldf_.aenergy[0] = p2ldffitter->energy;
    
    rufldf_.energy[0] = rusdenergy(rufldf_.s800[0],rusdgeom_.theta[2]);
    rufldf_.atmcor[0] = 1.0; // atmospheric corrections done by a different program */
    
    rufldf_.chi2[0] = p2ldffitter->chi2;
    rufldf_.ndof[0] = p2ldffitter->ndof;
    
    if(!p2ldffitter->hasConverged())
      rufldf_.chi2[0]=1.e6;

  }

void rufldfAnalysis::gldfFit()
  {
    p2gldffitter->loadVariables();
    //p2gldffitter->clean(P2GLDF_DCHI2);
    p2gldffitter->doFit();
    if (p2gldffitter->chi2 > 1.e6)
      p2gldffitter->chi2 = 1.e6;
    
    
    // Put variables into LDF DST bank  
    rufldf_.xcore[1] = p2gldffitter->R[0];
    rufldf_.dxcore[1] = p2gldffitter->dR[0];
    rufldf_.ycore[1] = p2gldffitter->R[1];
    rufldf_.dycore[1] = p2gldffitter->dR[1];
    rufldf_.sc[1] = p2gldffitter->S;
    rufldf_.dsc[1] = p2gldffitter->dS;
    rufldf_.s600[1] = p2gldffitter->s600;
    rufldf_.s600_0[1] = p2gldffitter->s600_0;
    rufldf_.s800[1] = p2gldffitter->s800;
    rufldf_.s800_0[1] = 0.0; // Not available yet
    rufldf_.aenergy[1] = p2gldffitter->energy;
    rufldf_.energy[1] = rusdenergy(rufldf_.s800[1],p2gldffitter->theta);
    rufldf_.atmcor[1] = 1.0; // atmospheric corrections done by a different program */
    rufldf_.chi2[1] = p2gldffitter->chi2;

    // Geometry variables that come out only from combined geom/ldf fit:
    rufldf_.theta = p2gldffitter->theta;
    rufldf_.dtheta = p2gldffitter->dtheta;
    rufldf_.phi = p2gldffitter->phi;
    rufldf_.dphi = p2gldffitter->dphi;
    rufldf_.t0 = p2gldffitter->T0;
    rufldf_.dt0 = p2gldffitter->dT0;

    rufldf_.ndof[1] = p2gldffitter->ndof;
    if(!p2gldffitter->hasConverged())
      rufldf_.chi2[1]=1.e6;
  }

void rufldfAnalysis::addCoreBorderInfo()
  {
    double xcore, ycore;
    double bdist, v[2];
    double tdistbr, vbr[2];
    double tdistlr, vlr[2];
    double tdistsk, vsk[2];
    double tdist;
    xcore=rusdgeom_.xcore[2];
    ycore=rusdgeom_.ycore[2];

    sdbdist(xcore, ycore, &v[0], &bdist, &vbr[0], &tdistbr, &vlr[0], &tdistlr,
        &vsk[0], &tdistsk);
    rufldf_.bdist = bdist;
    rufldf_.tdistbr = tdistbr;
    rufldf_.tdistlr = tdistlr;
    rufldf_.tdistsk = tdistsk;

    // Pick out the actual T-shape boundary distance for whatever subarray
    tdist = tdistbr;
    if (tdistlr> tdist)
      tdist = tdistlr;
    if (tdistsk> tdist)
      tdist = tdistsk;
    rufldf_.tdist = tdist;

  }
