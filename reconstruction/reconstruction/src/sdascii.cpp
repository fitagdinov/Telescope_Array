#include "sdascii.h"
#include "tacoortrans.h"
#include "sdenergy.h"
#include "sdxyzclf_class.h"
#include "TMath.h"

static sdxyzclf_class sdascii_sdxyzclf;

using namespace TMath;

// scale theta, phi fitting uncertainties 
// (for the fit that comes from the final curvature + development fit)
// so that they represent the true 68% confidence limits
// this has been found from the MC
#define dtheta_scaling_const 0.58
#define dphi_scaling_const   0.56


int main(int argc, char **argv)
{
  listOfOpt opt;
  sddstio_class dstio;
  char *infile;
  FILE *outfl   = 0;   // output file for event reconstruction information only
  FILE *sfoutfl = 0;   // output file for event reconstruction and shower front structure studies 

  integer4 reqBanks;        // list of required banks
  integer4 addBanks;        // banks that may be added to the output dst file

  /////// output variables /////////////
  int yyyymmdd,hhmmss,usec; // date and time, UTC
  double jday;              // julian date, days
  double lmst;              // local mean sidereal time, radians, 0..2*PI range
  double energy;            // energy, EeV
  double theta,dtheta;      // event zenith angle with error, radians
  double phi,dphi;          // event azimuthal angle with error, radians (X = East), 0..2*PI range
  double ra,dec;            // ra,dec in radians, ra in 0..2*PI range
  double l,b;               // galactic latitude/longitude, radians
  double sgl,sgb;           // supergalactic longitude/latitude, radians
  
  
  // additional variables for TA-Wiki
  double s800;              // S800 [VEM/m^2]
  double xcore;             // core X position in CLF frame, [m]
  double ycore;             // core Y position in CLF frame, [m]
  
  // dummy variables  
  int year,month,day,hour,minute,second;
  double second_since_midnight;
  double ha;                // hour angle, radians, 0 .. PI
  double lat,lon;           // latitude, longitude of TA CLF, radians
  int events_read;
  int events_written;
  

  // additional variables needed for making cuts
  double gfchi2pdof;        // Goeom. chi2 / dof
  double ldfchi2pdof;       // LDF chi2 / dof
  double pderr;             // pointing direction error, degree
  double ds800os800;        // sigma_S800 / S800

  
  // additional variables needed for styding shower front structure
  double geom_xcore, geom_ycore; // x and y core from geometry reconstruction (not LDF fit)
  double geom_tcore;             // time of the shower core [s] with respect to the GPS integer second
  double sdx, sdy, sdz;          // position of each individual hit SD [m] in CLF frame
  double tsd;                    // time of each individual hit SD [s] with respect to GPS integer second
  double tref;                   // event reference second fraction with respect to GPS integer second

  // MC variables needed for the shower front structure studies
  double mctheta, mcphi;
  double mcxcore;             // MC thrown core X position in CLF frame [m]                        
  double mcycore;             // MC thrown core Y position in CLF frame [m]  
  double mctcore;             // MC thrown core time [s] with respect to GPS integer second
  double mcenergy;            // MC thrown energy [EeV]

  // CLF latitude and longitude, radians
  lat = tacoortrans_CLF_Latitude  * DegToRad();
  lon = tacoortrans_CLF_Longitude * DegToRad();
  
  
  if(!opt.getFromCmdLine(argc,argv)) 
    return 2;
  
  opt.printOpts();
  
  // output file pointer initialized to stdout if one uses
  // such option or if one did not specify the output file name
  // if one specified the output file name, initialize it to 0 first
  // and it will be opened later when it's needed for writing events
  outfl = ((opt.stdout_opt || (!opt.outfile[0])) ? stdout : 0);
  
  events_read = 0;
  events_written = 0;

  reqBanks = newBankList(10);
  addBankList(reqBanks,RUSDRAW_BANKID);
  addBankList(reqBanks,RUFPTN_BANKID);
  addBankList(reqBanks,RUSDGEOM_BANKID);
  addBankList(reqBanks,RUFLDF_BANKID);
  if(opt.tb_opt)
    addBankList(reqBanks,SDTRGBK_BANKID);
  
  if(opt.brd_cut == 1 || opt.brd_cut == 2) 
    addBankList(reqBanks,BSDINFO_BANKID);

  // event track information bank may be filled out and added to the output event
  // dst banks, if certain options are used
  addBanks = newBankList(10);
  addBankList(addBanks,ETRACK_BANKID);
  
  while((infile=pullFile()))
    {
      
      // Open DST files
      if(!dstio.openDSTinFile(infile)) 
	return 2;
      
      while(dstio.readEvent())
	{

	  events_read ++;

	  // check the required banks
	  if(!dstio.haveBanks(reqBanks,opt.bank_warning_opt))
	    continue;
	  
	  ///////////////////// CUTS (BELOW) /////////////////////////
	  
	  // TRIGGER BACKUP CUT (IF THIS OPTION IS REQUESTED ON THE COMMAND LINE)
	  if(opt.tb_opt)
	    {     
	      
	      // event doesn't pass trigger backup in any case, 
	      // even when the pedestals are lowered to maximum possible amount
	      if(sdtrgbk_.igevent == 0)
		continue;
	      
	      // usual trigger backup cut, 
	      // without raising or lowering the pedestal
	      if(opt.tb_delta_ped == 0)
		{ 
		  if(sdtrgbk_.igevent < 2)
		    continue;
		}
	      
	      // if trigger backup cut with lowering the pedestal is allowed
	      if(opt.tb_delta_ped < 0)
		{
		  // check if event passes SD trigger backup with pedestals lowered by at most requested amount
		  if(sdtrgbk_.igevent==1 && (int)sdtrgbk_.dec_ped > Abs(opt.tb_delta_ped))
		    continue; 
		}
	      
	      // trigger backup cut with raised pedestal option
	      if(opt.tb_delta_ped > 0)
		{
		  // if event doesn't pass the trigger backup with raised pedestals
		  if(sdtrgbk_.igevent < 3)
		    continue;

		  // if the maximum pedestal increase that still results in a trigger
		  // is less than the one requested on the command line
		  if(sdtrgbk_.igevent == 3 && (int)sdtrgbk_.inc_ped < opt.tb_delta_ped)
		    continue; 
		}	 
	      
	    }

	  
	  // number of good SDs
	  if (rufptn_.nstclust < 5)
	    continue;
	  
	  // usual border cuts (1200m from the edge and the T-shape boundary (for DS1))
	  if(opt.brd_cut == 0)
	    {
	      if ((rufldf_.bdist < 1.0) || ((rusdraw_.yymmdd <= 81110) && (rufldf_.tdist < 1.0)))
		continue;
	    }
	  
	  // Another version of border cut: checking that the largest counter is surrounded by
	  // 4 working SDs
	  else if(opt.brd_cut == 1 || opt.brd_cut == 2)
	    {
	      integer4 event_surroundedness = sdascii_sdxyzclf.get_event_surroundedness(&rusdraw_,&rusdgeom_,&bsdinfo_);
	      // largest signal counter that's part of the event surrounded by 4 working counters
	      if(opt.brd_cut == 1)
		{
		  if(event_surroundedness < 1)
		    continue;
		}
	      // largest signal counter that's part of the event surrounded by 4 working counters that are
	      // immediate neighbors of the largest signal
	      if(opt.brd_cut == 2)
		{
		  if(event_surroundedness < 2)
		    continue;
		}
	    }
	 
	  // the cut flag is not supported; can't have that
	  else
	    {
	      fprintf(stderr,"Internal Error: %s(%d): borded_cut = %d not supported\n",
		      __FILE__,__LINE__,opt.brd_cut);
	      return 2;
	    }
	  
	  // chi2 / dof cuts (geometry, LDF)
	  gfchi2pdof = (rusdgeom_.ndof[2] < 1 ? rusdgeom_.chi2[2] : rusdgeom_.chi2[2]/(double)rusdgeom_.ndof[2]);
	  ldfchi2pdof = (rufldf_.ndof[0] < 1 ? rufldf_.chi2[0] : rufldf_.chi2[0] / (double) rufldf_.ndof[0]);
	  if (gfchi2pdof > 4.0 || ldfchi2pdof > 4.0)
	    continue;
	  
	  // sigma_s800 / s800
	  ds800os800 = rufldf_.dsc[0] / rufldf_.sc[0];
	  if (ds800os800 > 0.25)
	    continue;
	  
	  
	  // pointing direction error
	  pderr = sqrt(sin(rusdgeom_.theta[2] * DegToRad()) * sin(rusdgeom_.theta[2] * DegToRad()) * rusdgeom_.dphi[2]
		       * rusdgeom_.dphi[2] + rusdgeom_.dtheta[2] * rusdgeom_.dtheta[2]);
	  if (pderr > 5.0)
	    continue;
	  
	  // zentith angle
	  if ((rusdgeom_.theta[2] +0.5) > opt.za_cut)
	    continue;
	  
	  // energy, use energy >= minimum energy cut (option) AFTER the energy scale has
	  // been applied
	  // SD energy scale by default is set to FD ( default value of opt->enscale)
	  energy = rusdenergy(rufldf_.s800[0],rusdgeom_.theta[2]) * rufldf_.atmcor[0] * (opt.enscale);
	  if (energy < opt.emin)
	    continue;
	  
	  ///////////////////// CUTS (ABOVE) /////////////////////////
	  
	  
	  // date and time
	  yyyymmdd = 20000000+rusdraw_.yymmdd;
	  hhmmss   = rusdraw_.hhmmss;
	  usec     = rusdraw_.usec;
	  year   = yyyymmdd/10000;
	  month  = (yyyymmdd % 10000)/100;
	  day    = yyyymmdd % 100;
	  hour   = (hhmmss / 10000);
	  minute = (hhmmss % 10000)/100;
	  second = hhmmss % 100;
	  usec   = rusdraw_.usec;
	  second_since_midnight = (double)(hour*3600+minute*60+second)+
	    ((double)usec)/1e6;
	  jday = tacoortrans::utc_to_jday(year,month,day,second_since_midnight);
	  lmst  = tacoortrans::jday_to_LMST(jday,lon);

	  // zenith angle (pointing back to the source)
	  theta  = (rusdgeom_.theta[2] + 0.5)  * DegToRad();
	  dtheta = rusdgeom_.dtheta[2] * DegToRad();
	  
	  // azimuthal angle, X=East, pointing back to the source
	  phi    = tacoortrans::range(rusdgeom_.phi[2] + 180.0,360.0);
	  phi *= DegToRad();
	  dphi = rusdgeom_.dphi[2] * DegToRad();
	  
	  // rescale zenith angle, azimuthal angle errors ( so that they are
	  // true 68% C.L ? )
	  if (opt.rescale_err_opt)
	    {
	      dtheta *= dtheta_scaling_const;
	      dphi   *= dphi_scaling_const;
	    }
	  
	  // hour angle
	  ha = tacoortrans::get_ha(theta,phi,lat);
	  if (ha < 0)
	    ha += TwoPi();
	  
	  // right ascension
	  ra = lmst - ha;
	  while(ra >= TwoPi())
	    ra -= TwoPi();
	  while(ra < 0.0)
	    ra += TwoPi();
	  
	  // declination
	  dec = tacoortrans::get_dec(theta,phi,lat);
	  
	  // galactic coordinates	  
	  l = tacoortrans::gall(ra,dec);
	  b = tacoortrans::galb(ra,dec);
	  
	  // supergalactic coordinates
	  sgl = tacoortrans::sgall(ra,dec);
	  sgb = tacoortrans::sgalb(ra,dec);

	  s800  = rufldf_.s800[0];
	  xcore = (1.2e3) * (rufldf_.xcore[0] + RUFPTN_ORIGIN_X_CLF); // want X,Y-core in CLF frame
	  ycore = (1.2e3) * (rufldf_.ycore[0] + RUFPTN_ORIGIN_Y_CLF);
	  
	  
	  // write out the event

	  if(!outfl)
	    {
	      if (!opt.fOverwrite)
		{
		  if ((outfl=fopen(opt.outfile,"r")))
		    {
		      fprintf(stderr,"error: %s exists; use -f option\n",
			      opt.outfile);
		      fclose(outfl);
		      return 2;
		    }
		}
	      if(!(outfl=fopen(opt.outfile,"w")))
		{
		  fprintf(stderr,"error: failed to start %s\n",opt.outfile);
		  return 2;
		}
	    }
	  
	  // prefix event lines with EVT if writing into stdout so that one can grep for the events
	  if(outfl == stdout)
	    fprintf(outfl,"EVT ");
	  
	  // prints out event information depending on what the format option is
	  // so far either anisotropy-full or ta-wiki formats are supported
	  switch (opt.format)
	    {
	      
	      // FULL-ANISOTROPY FORMAT
	    case 0:
	      {
		// Date and time
		fprintf(outfl,"%08d %06d.%06d %.12f %.6f",yyyymmdd,hhmmss,usec,jday,lmst);
		// energy
		fprintf(outfl," %.3f",energy);
		// theta,dtheta and phi,dphi
		fprintf(outfl," %.6f %.6f %.6f %.6f",theta,dtheta,phi,dphi);
		// ha,ra,dec,l,b,sgl,sgb
		fprintf(outfl," %.6f %.6f %.6f %.6f %.6f %.6f %.6f",ha,ra,dec,l,b,sgl,sgb);
		// end of line
		fprintf(outfl,"\n");
		break;
	      }
	      
	      // TA-WIKI FORMAT
	    case 1:
	      {
		fprintf(outfl," %d, %06d.%06d, %8.3f, %8.3f, %8.2f, %8.2f, %8.2f, %8.3f, %8.2f, %8.2f\n",
			yyyymmdd,hhmmss,usec,xcore/1e3,ycore/1e3,s800,
			RadToDeg()*theta,
			RadToDeg()*phi,
			0.0,0.0,
			energy);
		break;
	      }
	      
	    default:
	      fprintf(stderr,"fatal error: %s(%d): data format = %d is not supported\n",__FILE__,__LINE__,opt.format);
	      break;
	    }
	  
	  fflush(outfl);

	  // if user desires information for the shower front structure
	  if(opt.sfoutfile[0])
	    {
	      // open shower front ASCII file for writing if it hasn't been
	      // opened yet
	      if(!sfoutfl)
		{
		  // check if the file exists first if the force overwrite option is not used
		  if(!opt.fOverwrite)
		    {
		      if((sfoutfl=fopen(opt.sfoutfile,"r")))
			{
			  fprintf(stderr,"error: shower front output file %s exists; use -f to overwrite output files!\n",
				  opt.sfoutfile);
			  fclose(sfoutfl);
			  return 2;
			}
		    }
		  if(!(sfoutfl=fopen(opt.sfoutfile,"w")))
		    {
		      fprintf(stderr,"error: failed to open %s for writing!\n",opt.sfoutfile);
		      return 2;
		    }
		}
	      
	      // reference second fraction [s] with respect to the integer GPS second
	      tref =  (rusdgeom_.tearliest - (int)Floor(rusdgeom_.tearliest)); // reference second fraction [second]

	      // core position from the geometry fit (not from LDF fit)
	      geom_xcore = (1.2e3) * (rusdgeom_.xcore[2] + RUFPTN_ORIGIN_X_CLF);
	      geom_ycore = (1.2e3) * (rusdgeom_.ycore[2] + RUFPTN_ORIGIN_Y_CLF);
	  
	      // Fill out MC information only if MC banks are present (if this is a Monte Carlo event)
	      // otherwise, zero out the Monte Carlo information
	      if(dstio.haveBank(RUSDMC_BANKID) && dstio.haveBank(RUSDMC1_BANKID))
		{
		  // MC thrown core position
		  mcxcore= (1.2e3) * (rusdmc1_.xcore + RUFPTN_ORIGIN_X_CLF);
		  mcycore = (1.2e3) * (rusdmc1_.ycore + RUFPTN_ORIGIN_Y_CLF);

		  // MC thrown core time
		  mctcore =  tref + 1e-6*(rusdmc1_.t0/RUSDGEOM_TIMDIST);  // [ second ]
		  
		  // MC thrown direction
		  mctheta = rusdmc_.theta*RadToDeg();
		  mcphi = rusdmc_.phi*RadToDeg();
		  if(rusdmc_.phi*RadToDeg()<=180) mcphi+=180;
		  if(rusdmc_.phi*RadToDeg()>=180) mcphi-=180;
		  
		  // MC thrown energy
		  mcenergy = (opt.enscale) * rusdmc_.energy;
		}
	      else
		{
		  mcxcore  = 0;
		  mcycore  = 0;
		  mctcore  = 0;
		  mctheta  = 0;
		  mcphi    = 0;
		  mcenergy = 0;
		}
	      
	      // geometry fit core time
	      geom_tcore = tref + 1e-6*(rusdgeom_.t0[2]/RUSDGEOM_TIMDIST);  // [ second ]

	      // loop over all hits in rufptn DST bank
	      for (int ihit=0; ihit<rufptn_.nhits; ihit++)
		{
		  
		  // use good hit SDs; skip anything that has flag less than 4
		  if (rufptn_.isgood[ihit] < 4)
		    continue;
		  
		  // hit SD time
		  tsd = (rufptn_.reltime[ihit][0]+rufptn_.reltime[ihit][1] )/2.0  ;
		  tsd = tref + 1e-6*(tsd/RUSDGEOM_TIMDIST); // hit SD time [ s ]
		  
		  // hit SD position
		  sdx = 1.2e3 * rufptn_.xyzclf[ihit][0]; // hit SD coordinates in CLF frame [ m ]
		  sdy = 1.2e3 * rufptn_.xyzclf[ihit][1];
		  sdz = 1.2e3 * rufptn_.xyzclf[ihit][2];
		  
		  // printint event information into a designated ASCII output file
		  fprintf(sfoutfl,"%d %06d.%06d %04d %.3f %.3f %.3f %.3f %.3f %.2f %.2f %.2f %.4e %.3f %.8f %.8f ,",  //reconstruction
			  yyyymmdd,
			  hhmmss,usec,
			  rufptn_.xxyy[ihit], 
			  sdx, sdy, sdz,    // SD position in CLF frame in [ m ] 
			  geom_xcore,       // Geometry reconstructed core X position [ m ]
			  geom_ycore,       // Geometry reconstructed core Y position [ m ]
			  s800,             // LDF reconstructed S800 [ VEM/m^2 ]
			  RadToDeg()*theta, // zenith angle [ Degree ]
			  RadToDeg()*phi,   // azimuthal angle [ Degree]
			  1e18*energy,      // shower energy [eV]
			  (rufptn_.pulsa[ihit][0] + rufptn_.pulsa[ihit][1])/2, // hit SD signal [ vem ]
			  geom_tcore, tsd); // geom_tcore and thit in unit of [s]
		  fprintf(sfoutfl," %.3e %.3f %.3f %.3f %.3f %.8f\n", //MC result
			  1e18*mcenergy, mctheta, mcphi, mcxcore, mcycore, mctcore); 
		}
	      
	    }
	  
	  
	  // if one also wants to write out the selected events into a dst file
	  if(opt.dstoutfile[0])
	    {
	      if(!dstio.outFileOpen())
		{
		  if(!dstio.openDSToutFile(opt.dstoutfile,opt.fOverwrite))
		    return 2;
		}
	      // fill out the etrack dst bank if such option is used
	      if(opt.f_etrack)
		{
		  etrack_.energy    = (real4)energy;
		  etrack_.xmax      = 0.0;
		  etrack_.theta     = (real4)theta;
		  etrack_.phi       = (real4)phi;
		  // calculating core time with respect to GPS second, [uS]
		  // 1st piece is the SD reference time with respect to GPS second, [uS]
		  // 2nd piece is the SD core time with respect to SD reference time, [uS]
		  etrack_.t0        = 1e6 * (rusdgeom_.tearliest-floor(rusdgeom_.tearliest));
		  etrack_.t0       += rusdgeom_.t0[2]*RUSDGEOM_TIMDIST;
		  etrack_.xycore[0] = xcore;
		  etrack_.xycore[1] = ycore;
		  etrack_.nudata    = 0;
		  etrack_.yymmdd    = rusdraw_.yymmdd;
		  etrack_.hhmmss    = rusdraw_.hhmmss;
		  etrack_.qualct    = 1; // all events written to the DST file pass the quality cuts
		  dstio.writeEvent(addBanks,true); // write out the event w/ the additional bank
		}
	      else
		dstio.writeEvent(); // just write out the event as is w/o adding any new banks
	    }
	  
	  events_written ++;
	} // while(dstio.ReadEvent ...
      
      dstio.closeDSTinFile();
    
    }
  
  // finalize the output
  if((outfl != 0) && (outfl != stdout))
    fclose(outfl);
  
  if(dstio.outFileOpen())
    dstio.closeDSToutFile();
  
  fprintf(stdout,"events_read: %d events_written: %d zenith_cut: %.2f\n",
	  events_read,events_written,opt.za_cut);
  fprintf(stdout,"\n\nDone\n");
  fflush(stdout);
  return 0;
}
