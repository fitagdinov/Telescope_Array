
using namespace TMath;



// FD Site locations in SD coordinate system
#define BR_X 26.433588 
#define BR_Y 6.403746
#define BR_Z -0.010079

#define LR_X -3.071120 
#define LR_Y  8.221666
#define LR_Z  0.114935

#define MD_X 6.153441
#define MD_Y 32.720698
#define MD_Z 0.153190




#define secIn1200m 4.0027691424e-6 // Seconds in a [1200m] unit

void parseAABBCC(Int_t AABBCC, Int_t *aa, Int_t *bb, Int_t *cc)
{
  (*aa) = AABBCC / 10000;
  (*bb) = (AABBCC % 10000) / 100;
  (*cc) = (AABBCC % 100); 
}


void writePdAscii(const char *fname = "sd4lscott.txt")
{
  Int_t year;
  Int_t month;
  Int_t day;
  Int_t hour;
  Int_t minute;
  Int_t second;
  Double_t theta;
  Double_t phi;
  Double_t dphi;
  Double_t dtheta;
  Double_t chi2;
  Int_t ndof;
  Int_t nsds;

  Double_t s800;
  Double_t ldfchi2;
  Int_t ldfndof;
  
  FILE *fp;
  Int_t nevents;
  if(!(fp = fopen(fname,"w")))
    {
      fprintf(stderr,"Can't start %s",fname);
      return;
    }  
  fprintf(stdout,"Writing pointing direction ascii file ...\n");  
  nevents = (Int_t)p1.GetEntries();
  for (Int_t i = 0; i < nevents; i++)
    {
      p1.GetEntry(i);
      parseAABBCC(p1.rusdraw->yymmdd,&year,&month,&day);
      parseAABBCC(p1.rusdraw->hhmmss,&hour,&minute,&second);
      theta = p1.rusdgeom->theta[2];
      dtheta = p1.rusdgeom->dtheta[2];
      phi = p1.rusdgeom->phi[2]+180.0;
      if(phi > 360.0) phi -= 360.0;
      dphi = p1.rusdgeom->dphi[2];
      chi2 = p1.rusdgeom->chi2[2];
      ndof = p1.rusdgeom->ndof[2];
      nsds = p1.rufptn->nstclust;

      
      // If were not able to load the variables (not enough data points)
      if(!ldf->loadVariables(p1.rusdraw,p1.rufptn,p1.rusdgeom))
	continue;
      
      // If were not able to fit (not enough data points)
      if(!ldf->Ifit(false,false))
	continue;

      s800    = (ldf->S800);
      ldfchi2 = (ldf->chi2);
      ldfndof = (ldf->ndof);


      fprintf(fp,"%d %d %d %d %d %d %f %f %f %f %f %d %d %f %f %d\n",
	      year,month,day,hour,minute,second,
	      theta,dtheta,phi,dphi,chi2,ndof,nsds,
	      s800,ldfchi2,ldfndof);

      fprintf(stdout,"Completed: %.0f%c\r", (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout); 
    }
  fprintf(stdout,"\n");
  fclose(fp);
}



// Write an ascii file for comparing with FD
void writeFdCompAscii(Int_t fdsite=0,
		      const char* asciiFile="fileSD.txt")
{
  Int_t i;
  Int_t nevents;
  Int_t eventsWritten;
  
  Int_t    yyyymmdd;
  Int_t    hhmmss;
  Int_t    usec;
  Double_t xcore;
  Double_t ycore;
  Double_t s800;
  Double_t theta;
  Double_t phi;
  Double_t rp;
  Double_t psi;
  Double_t energy;


  Double_t sdo[2];

  Double_t rxyz[3];   // Vector in the direction of shower propagation
  
  Double_t fd_XYZ[3]; // Position of the FD detector in SD coordinates
  Double_t cosPsi;    // Cosine of FD psi angle
  Double_t dFd[3];    // Distance to shower core from FD
  Double_t dFdMag;    // Magnitude of the distance to shower core from FD

  Int_t irec;           // for looping over various reconstructions
  Double_t chi2pdof[5]; // for making cuts on chi2/dof, for 5 reconstructions

  char fdSiteName[0x100];
  FILE *asciifl;


  if (fdsite < 0 || fdsite > 2)
    {
      fprintf (stderr, "Choose FD site: BR=0,LR=1,MD=2 \n");
      return;
    }

  switch (fdsite)
    {

    case 0:
      fd_XYZ[0] = BR_X;
      fd_XYZ[1] = BR_Y;
      fd_XYZ[2] = BR_Z;
      sprintf(fdSiteName,"BR");
      break;
    case 1:
      fd_XYZ[0] = LR_X;
      fd_XYZ[1] = LR_Y;
      fd_XYZ[2] = LR_Z;
      sprintf(fdSiteName,"LR");
      break;
    case 2:
      fd_XYZ[0] = MD_X;
      fd_XYZ[1] = MD_Y;
      fd_XYZ[2] = MD_Z;
      sprintf(fdSiteName,"MD");
      break;
    default:
      fprintf(stderr, "Error: incorrect FD site\n");
      return;
      break;
    }
  
  
  memcpy(&sdo[0],&sdgeom->sdorigin_xy[0],2*sizeof(Double_t));
  
  if ( (asciifl = fopen(asciiFile,"w")) == 0 )
    {
      fprintf (stderr, "Failed to start: %s\n",asciiFile);
      return;
    }
  
  eventsWritten = 0;
  nevents = p1.GetEntries();
  fprintf(stdout,"nevents = %d\n",nevents);
  
  fprintf(stdout,"Writing ASCII file for FD-%s comparison ...\n",fdSiteName);
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);
      
      // Cut on number of counters in the time fit
      if (p1.rufptn->nstclust < 5)
	continue;
      
      // Cut out events that are on the border
      if (p1.rufldf->bdist<1.)
	continue;
      
      if (p1.rufldf->tdist<1.)
	continue;

      
      // chi2/dof or just chi2 if ndof < 1:
      
      // 3 geom. fits:
      for (irec = 0; irec < 3; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rusdgeom->ndof[irec] > 1) 
	       ? 
	       (p1.rusdgeom->chi2[irec]/(double)p1.rusdgeom->ndof[irec])
	       :
	       (p1.rusdgeom->chi2[irec])
		);
	}
      // 2 ldf fits
      for (irec = 3; irec < 5; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rufldf->ndof[irec-3] > 1) 
	       ? 
	       (p1.rufldf->chi2[irec-3]/(double)p1.rufldf->ndof[irec-3])
	       :
	       (p1.rufldf->chi2[irec-3])
		);
	}
      

      // chi2 cuts:

      // geometry
      if (chi2pdof[2] > 4.0 )
	continue;

      // ldf
      if (chi2pdof[3] > 4.0 )
	continue;
            
      // pointing direction resolution cut
      if (pdErr(p1.rusdgeom->theta[2],p1.rusdgeom->dtheta[2],p1.rusdgeom->dphi[2]) > 5.0)
      	continue;

      // cut on s800
      if (p1.rufldf->dsc[0] / p1.rufldf->sc[0] > 0.25)
      	continue;
      
      // zenith angle cut
      if (p1.rusdgeom->theta[2] > 45.0)
	continue;
      
      energy = p1.rufldf->energy[0];
      
      // cut on energy
      if (energy <1.0)
	continue;
      
      yyyymmdd  = p1.rusdraw->yymmdd+20000000;
      hhmmss    = p1.rusdraw->hhmmss;
      usec      = p1.rusdraw->usec;
      xcore     = p1.rufldf->xcore[0];
      ycore     = p1.rufldf->ycore[0];
      s800      = p1.rufldf->s800[0];
      theta     = p1.rusdgeom->theta[2]+0.5;
      phi       = p1.rusdgeom->phi[2];
      // Make the phi angle point back to the source (as it is for FD reconstruction)
      phi += 180.;
      while (phi > 360.)
	phi -= 360.;
      while (phi < 0.) 
	phi += 360.;
      // Shower propagation vector ( pointing along the momentum of the primary):
      rxyz[0]  = -Sin (DegToRad() * theta) * Cos (DegToRad() * phi);
      rxyz[1]  = -Sin (DegToRad() * theta) * Sin (DegToRad() * phi);
      rxyz[2]  = -Cos (DegToRad() * theta);

      
      
      dFd[0]    = xcore - fd_XYZ[0];
      dFd[1]    = ycore - fd_XYZ[1];
      dFd[2]    = 0.0   - fd_XYZ[2];
      dFdMag = sqrt (dFd[0]*dFd[0]+dFd[1]*dFd[1]+dFd[2]*dFd[2]); 
      cosPsi=(rxyz[0]*dFd[0]+rxyz[1]*dFd[1]+rxyz[2]*dFd[2])/dFdMag;
      psi = RadToDeg() * ACos (cosPsi);
      rp  =  dFdMag * sqrt( 1.0 - cosPsi * cosPsi );
      
      
      // Print SD reconstruction information into the ASCII file
      fprintf(asciifl,
	      " %d, %06d.%06d, %8.3f, %8.3f, %8.2f, %8.2f, %8.2f, %8.3f, %8.2f, %8.2f\n",
	      yyyymmdd,hhmmss,usec,
	      1.2*(xcore+sdo[0]),1.2*(ycore+sdo[1]),
	      s800,theta,phi,1.2*rp,psi,energy
	      );
      
      eventsWritten ++;
      
    }
  fprintf(stdout,"%d events written\n",eventsWritten);
  fclose (asciifl);
}




// Write an ascii file for comparing with FD for each of the
// 3 towers separately ( applicable for separated DS1 only)
void writeFdCompAscii_tower(Int_t fdsite=0, 
			    const char *asciiFileBR="fileBR.txt", 
			    const char *asciiFileLR="fileLR.txt", 
			    const char *asciiFileSK="fileSK.txt")
{
  Int_t i;

  Int_t sdsite;
  Int_t nevents;
  Int_t eventsWritten;
  
  Int_t    yyyymmdd;
  Int_t    hhmmss;
  Int_t    usec;
  Double_t xcore;
  Double_t ycore;
  Double_t s800;
  Double_t theta;
  Double_t phi;
  Double_t rp;
  Double_t psi;
  Double_t energy;


  Double_t sdo[2];

  Double_t rxyz[3];   // Vector in the direction of shower propagation
  
  Double_t fd_XYZ[3]; // Position of the FD detector in SD coordinates
  Double_t cosPsi;    // Cosine of FD psi angle
  Double_t dFd[3];    // Distance to shower core from FD
  Double_t dFdMag;    // Magnitude of the distance to shower core from FD

  Int_t irec;           // for looping over various reconstructions
  Double_t chi2pdof[5]; // for making cuts on chi2/dof, for 5 reconstructions

  char fdSiteName[0x100];


  if (fdsite < 0 || fdsite > 2)
    {
      fprintf (stderr, "Choose FD site: BR=0,LR=1,MD=2 \n");
      return;
    } 

  switch (fdsite)
    {

    case 0:
      fd_XYZ[0] = BR_X;
      fd_XYZ[1] = BR_Y;
      fd_XYZ[2] = BR_Z;
      sprintf(fdSiteName,"BR");
      break;
    case 1:
      fd_XYZ[0] = LR_X;
      fd_XYZ[1] = LR_Y;
      fd_XYZ[2] = LR_Z;
      sprintf(fdSiteName,"LR");
      break;
    case 2:
      fd_XYZ[0] = MD_X;
      fd_XYZ[1] = MD_Y;
      fd_XYZ[2] = MD_Z;
      sprintf(fdSiteName,"MD");
      break;
    default:
      fprintf(stderr, "Error: incorrect FD site\n");
      return;
      break;
    }
  
  
  memcpy(&sdo[0],&sdgeom->sdorigin_xy[0],2*sizeof(Double_t));
  
  FILE *fl[3];
  fl[0] = fopen(asciiFileBR,"w");
  fl[1] = fopen(asciiFileLR,"w");
  fl[2] = fopen(asciiFileSK,"w");
  

  eventsWritten = 0;
  nevents = p1.GetEntries();
  fprintf(stdout,"nevents = %d\n",nevents);
  
  fprintf(stdout,"Writing ASCII file for FD-%s comparison ...\n",fdSiteName);
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);

      // Choosing the SD site
      sdsite=p1.rusdraw->site;
      if (sdsite < 0 || sdsite > 3)
	{
	  fprintf (stderr, "Multi-SD-site event, not recording for now\n");
	  continue;
	}
      
      
      // Cut on number of counters in the time fit
      if (p1.rufptn->nstclust < 4)
	continue;
      
      // Cut out events that are on the border
      if (p1.rufldf->bdist<1.)
	continue;
      
      if (p1.rufldf->tdist<1.)
	continue;

      
      // chi2/dof or just chi2 if ndof < 1:
      
      // 3 geom. fits:
      for (irec = 0; irec < 3; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rusdgeom->ndof[irec] > 1) 
	       ? 
	       (p1.rusdgeom->chi2[irec]/(double)p1.rusdgeom->ndof[irec])
	       :
	       (p1.rusdgeom->chi2[irec])
		);
	}
      // 2 ldf fits
      for (irec = 3; irec < 5; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rufldf->ndof[irec-3] > 1) 
	       ? 
	       (p1.rufldf->chi2[irec-3]/(double)p1.rufldf->ndof[irec-3])
	       :
	       (p1.rufldf->chi2[irec-3])
		);
	}
      
      if (chi2pdof[1] > 10.0)
	continue;
      if (chi2pdof[3] > 10.0)
	continue;
            
      if (pdErr(p1.rusdgeom->theta[1],p1.rusdgeom->dtheta[1],p1.rusdgeom->dphi[1]) > 10.0)
      	continue;
      
      if (corErr(p1.rufldf->dxcore[0],p1.rufldf->dycore[0]) > 0.25)
	continue;
      
      if (p1.rufldf->dsc[0] / p1.rufldf->sc[0] > 0.25)
      	continue;
      
      if (p1.rusdgeom->theta[1] > 45.0)
	continue;
      
      energy = p1.rufldf->energy[0];
      
      if (energy <1.0)
	continue;
      
      yyyymmdd  = p1.rusdraw->yymmdd+20000000;
      hhmmss    = p1.rusdraw->hhmmss;
      usec      = p1.rusdraw->usec;
      xcore     = p1.rufldf->xcore[0];
      ycore     = p1.rufldf->ycore[0];
      s800      = p1.rufldf->s800[0];
      theta     = p1.rusdgeom->theta[1];
      phi       = p1.rusdgeom->phi[1];
      // Make the phi angle point back to the source (as it is for FD reconstruction)
      phi += 180.;
      while (phi > 360.)
	phi -= 360.;
      while (phi < 0.) 
	phi += 360.;
      // Shower propagation vector ( pointing along the momentum of the primary):
      rxyz[0]  = -Sin (DegToRad() * theta) * Cos (DegToRad() * phi);
      rxyz[1]  = -Sin (DegToRad() * theta) * Sin (DegToRad() * phi);
      rxyz[2]  = -Cos (DegToRad() * theta);

      
      
      dFd[0]    = xcore - fd_XYZ[0];
      dFd[1]    = ycore - fd_XYZ[1];
      dFd[2]    = 0.0   - fd_XYZ[2];
      dFdMag = sqrt (dFd[0]*dFd[0]+dFd[1]*dFd[1]+dFd[2]*dFd[2]); 
      cosPsi=(rxyz[0]*dFd[0]+rxyz[1]*dFd[1]+rxyz[2]*dFd[2])/dFdMag;
      psi = RadToDeg() * ACos (cosPsi);
      rp  =  dFdMag * sqrt( 1.0 - cosPsi * cosPsi );
      
      
      
      fprintf(fl[sdsite],
	      " %d, %06d.%06d, %8.3f, %8.3f, %8.2f, %8.2f, %8.2f, %8.3f, %8.2f, %8.2f\n",
	      yyyymmdd,hhmmss,usec,
	      1.2*(xcore+sdo[0]),1.2*(ycore+sdo[1]),
	      s800,theta,phi,1.2*rp,psi,energy
	      );
      
      
      eventsWritten ++;
      
      
    }
  
  for (sdsite=0;sdsite<3;sdsite++)
    fclose(fl[sdsite]);
  fprintf(stdout,"%d events written\n",eventsWritten);
}


// Write an ascii file for radar coincidence searches
void writeAscii4radar(char *asciiFile)
{
  Int_t i;
  Int_t nevents;
  Int_t eventsWritten;
  Double_t energy;
  Double_t log10en;
  Double_t sdo[2];
  Double_t tt;
  Double_t theta;
  Double_t phi;
  Double_t s600;
  Double_t s800;
  Double_t xcore;
  Double_t ycore;
  Int_t irec;           // for looping over various reconstructions
  Double_t chi2pdof[5]; // for making cuts on chi2/dof, for 5 reconstructions
  bool chi2cutflag;
  Int_t pqcuts;
  
  memcpy(&sdo[0],&sdgeom->sdorigin_xy[0],2*sizeof(Double_t));
  
  FILE *fl = fopen(asciiFile,"w");

  eventsWritten = 0;
  nevents = p1.GetEntries();
  fprintf(stdout,"nevents = %d\n",nevents);
  
  fprintf(stdout,"Writing ASCII file for Radar comparison ...\n");
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      

      // Number of counters must be at least 3 in order
      // for events to make it into ascii file.
      if (p1.rufptn->nstclust < 3)
	continue;
      
      pqcuts = 1;
      
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);
      
      // Cut on number of counters in the time fit
      if (p1.rufptn->nstclust < 4)
	pqcuts=0;
      
      // Cut out events that are on the border
      if (p1.rufldf->bdist<1.)
	pqcuts=0;
      
      if (p1.rufldf->tdist<1.)
	pqcuts=0;

      
      // chi2/dof or just chi2 if ndof < 1:

      // 3 geom. fits:
      for (irec = 0; irec < 3; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rusdgeom->ndof[irec] > 1) 
	       ? 
	       (p1.rusdgeom->chi2[irec]/(double)p1.rusdgeom->ndof[irec])
	       :
	       (p1.rusdgeom->chi2[irec])
		);
	}
      // 2 ldf fits
      for (irec = 3; irec < 5; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rufldf->ndof[irec-3] > 1) 
	       ? 
	       (p1.rufldf->chi2[irec-3]/(double)p1.rufldf->ndof[irec-3])
	       :
	       (p1.rufldf->chi2[irec-3])
		);
	}
      chi2cutflag = false;
      for (irec=0; irec < 5; irec ++)
	{
	  if ( chi2pdof[irec] > 4. )
	    {
	      chi2cutflag = true;
	      break;
	    }
	}
      if (chi2cutflag)
	pqcuts=0;

      
      if (pdErr(p1.rusdgeom->theta[1],p1.rusdgeom->dtheta[1],p1.rusdgeom->dphi[1]) > 5.0)
      	pqcuts=0;
      
      if (corErr(p1.rufldf->dxcore[0],p1.rufldf->dycore[0]) > 0.25)
	pqcuts=0;
      
      if (p1.rufldf->dsc[0] / p1.rufldf->sc[0] > 0.25)
	pqcuts=0;
      
      if (p1.rusdgeom->theta[1] > 45.0)
	pqcuts=0;
      
      energy = p1.rufldf->energy[0];
      
      if (energy < 1.0)
       	pqcuts=0;
  
      
      tt = 
	3600.0 * (Double_t)(p1.rusdraw->hhmmss / 10000) +
	60.0 * (Double_t)((p1.rusdraw->hhmmss % 10000)/100)  +
	(Double_t)(p1.rusdraw->hhmmss % 100) +
	((Double_t)p1.rusdraw->usec) / 1e6;


      if (chi2cutflag)
	{
	  xcore=1.2e3*(sdo[0]+p1.rufptn->tyro_xymoments[2][0]);
	  ycore=1.2e3*(sdo[1]+p1.rufptn->tyro_xymoments[2][1]);
	  theta=0.0;
	  phi=0.0;
	  s600=0.0;
	  s800=0.0;
	  log10en=0.0;
	}
      else
	{
	  xcore=1.2e3 * (p1.rufldf->xcore[0]+sdo[0]);
	  ycore=1.2e3 * (p1.rufldf->ycore[0]+sdo[1]);
	  theta=p1.rusdgeom->theta[1];
	  phi=p1.rusdgeom->phi[1];
	  s600=p1.rufldf->s600[0];
	  s800=p1.rufldf->s800[0];
	  log10en = 18.0+Log10(energy);
	}    
      
      fprintf (fl,
	       "%d %06d %06d %02d %lf %lf %lf %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
	       p1.rusdraw->site,p1.rusdraw->yymmdd,p1.rusdraw->hhmmss,
	       p1.rufptn->nstclust,
	       tt,
	       0.5*(p1.rufptn->tearliest[0]+p1.rufptn->tearliest[1]),
	       0.5*(p1.rufptn->tearliest[0]+p1.rufptn->tearliest[1]) +
	       secIn1200m * p1.rusdgeom->t0[1],
	       xcore,
	       ycore,
	       theta,
	       phi,
	       s600,
	       s800,
	       log10en
	       );
      eventsWritten ++;
    }
  
  fclose(fl);
  fprintf(stdout,"%d events written\n",eventsWritten);
}




// Write an ascii file for comparing with FD
void writeFdDebebugAscii(Int_t fdsite=0, 
			 const char *asciiFileBR="fileBR.txt", 
			 const char *asciiFileLR="fileLR.txt", 
			 const char *asciiFileSK="fileSK.txt")
{
  Int_t i;

  Int_t sdsite;
  Int_t nevents;
  Int_t eventsWritten;
  
  Int_t    yyyymmdd;
  Int_t    hhmmss;
  Int_t    usec;
  Double_t xcore;
  Double_t ycore;
  Double_t s800;
  Double_t theta;
  Double_t phi;
  Double_t rp;
  Double_t psi;
  Double_t energy;


  Double_t sdo[2];

  Double_t rxyz[3];   // Vector in the direction of shower propagation
  
  Double_t fd_XYZ[3]; // Position of the FD detector in SD coordinates
  Double_t cosPsi;    // Cosine of FD psi angle
  Double_t dFd[3];    // Distance to shower core from FD
  Double_t dFdMag;    // Magnitude of the distance to shower core from FD

  Int_t irec;           // for looping over various reconstructions
  Double_t chi2pdof[5]; // for making cuts on chi2/dof, for 5 reconstructions

  char fdSiteName[0x100];

  Int_t nstclust;
  Int_t corecounter;

  if (fdsite < 0 || fdsite > 2)
    {
      fprintf (stderr, "Choose FD site: BR=0,LR=1,MD=2 \n");
      return;
    } 

  switch (fdsite)
    {

    case 0:
      fd_XYZ[0] = BR_X;
      fd_XYZ[1] = BR_Y;
      fd_XYZ[2] = BR_Z;
      sprintf(fdSiteName,"BR");
      break;
    case 1:
      fd_XYZ[0] = LR_X;
      fd_XYZ[1] = LR_Y;
      fd_XYZ[2] = LR_Z;
      sprintf(fdSiteName,"LR");
      break;
    case 2:
      fd_XYZ[0] = MD_X;
      fd_XYZ[1] = MD_Y;
      fd_XYZ[2] = MD_Z;
      sprintf(fdSiteName,"MD");
      break;
    default:
      fprintf(stderr, "Error: incorrect FD site\n");
      return;
      break;
    }
  
  
  memcpy(&sdo[0],&sdgeom->sdorigin_xy[0],2*sizeof(Double_t));
  
  FILE *fl[3];
  fl[0] = fopen(asciiFileBR,"w");
  fl[1] = fopen(asciiFileLR,"w");
  fl[2] = fopen(asciiFileSK,"w");
  

  eventsWritten = 0;
  nevents = p1.GetEntries();
  fprintf(stdout,"nevents = %d\n",nevents);
  
  fprintf(stdout,"Writing ASCII file for FD-%s comparison ...\n",fdSiteName);
  for (i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);

      // Choosing the SD site
      sdsite=p1.rusdraw->site;
      if (sdsite < 0 || sdsite > 3)
	{
	  fprintf (stderr, "Multi-SD-site event, not recording for now\n");
	  continue;
	}
      
 
      
      
      // chi2/dof or just chi2 if ndof < 1:
      
      // 3 geom. fits:
      for (irec = 0; irec < 3; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rusdgeom->ndof[irec] > 1) 
	       ? 
	       (p1.rusdgeom->chi2[irec]/(double)p1.rusdgeom->ndof[irec])
	       :
	       (p1.rusdgeom->chi2[irec])
		);
	}
      // 2 ldf fits
      for (irec = 3; irec < 5; irec ++)
	{
	  chi2pdof[irec] 
	    = ( 
	       (p1.rufldf->ndof[irec-3] > 1) 
	       ? 
	       (p1.rufldf->chi2[irec-3]/(double)p1.rufldf->ndof[irec-3])
	       :
	       (p1.rufldf->chi2[irec-3])
		);
	}
      
           
      // Cut on number of counters in the time fit
      if (p1.rufptn->nstclust < 9)
	continue;
      if (chi2pdof[1] > 10.0)
	continue;
      if (chi2pdof[3] > 10.0)
	continue;
      
      energy = p1.rufldf->energy[0];
      
      yyyymmdd  = p1.rusdraw->yymmdd+20000000;
      hhmmss    = p1.rusdraw->hhmmss;
      usec      = p1.rusdraw->usec;
      xcore     = p1.rufldf->xcore[0];
      ycore     = p1.rufldf->ycore[0];
      s800      = p1.rufldf->s800[0];
      theta     = p1.rusdgeom->theta[1];
      phi       = p1.rusdgeom->phi[1];
      nstclust  = p1.rufptn->nstclust;
      corecounter= ((Int_t)Floor(xcore+0.5))*100 +((Int_t)Floor(ycore+0.5));
      // Make the phi angle point back to the source (as it is for FD reconstruction)
      phi += 180.;
      while (phi > 360.)
	phi -= 360.;
      while (phi < 0.) 
	phi += 360.;
      // Shower propagation vector ( pointing along the momentum of the primary):
      rxyz[0]  = -Sin (DegToRad() * theta) * Cos (DegToRad() * phi);
      rxyz[1]  = -Sin (DegToRad() * theta) * Sin (DegToRad() * phi);
      rxyz[2]  =  Cos (DegToRad() * theta);

      
      
      dFd[0]    = xcore - fd_XYZ[0];
      dFd[1]    = ycore - fd_XYZ[1];
      dFd[2]    = 0.0   - fd_XYZ[2];
      dFdMag = sqrt (dFd[0]*dFd[0]+dFd[1]*dFd[1]+dFd[2]*dFd[2]); 
      cosPsi=(rxyz[0]*dFd[0]+rxyz[1]*dFd[1]+rxyz[2]*dFd[2])/dFdMag;
      psi = RadToDeg() * ACos (cosPsi);
      rp  =  dFdMag * sqrt( 1.0 - cosPsi * cosPsi );
      
      
      
      fprintf(fl[sdsite],
	      " %d %06d.%06d %8.3f %8.3f %8.04d %8.2f %8.2f %8.3f %8.2f %8d\n",
	      yyyymmdd,hhmmss,usec,
	      1.2*(xcore+sdo[0]),1.2*(ycore+sdo[1]),
	      corecounter,theta,phi,1.2*rp,psi,nstclust
	      );
      
      
      eventsWritten ++;
      
      
    }
  
  for (sdsite=0;sdsite<3;sdsite++)
    fclose(fl[sdsite]);
  fprintf(stdout,"%d events written\n",eventsWritten);
}
