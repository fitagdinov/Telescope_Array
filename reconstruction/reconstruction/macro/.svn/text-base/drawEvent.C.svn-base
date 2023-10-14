#include "sdborderlines.h"


void drawBorders(Double_t xlo, Double_t xup, Double_t ylo, Double_t yup)
{
  static bool prepborders = true;
  Double_t val,x1,x2,y1,y2;
  Double_t xc,yc,vx,vy,dist;
  Double_t dx,dy;
  Double_t xloi[2],xupi[2],yloi[2],yupi[2];
  Double_t len;
  Int_t iborder,jborder;
  Bool_t set2draw;
  if (prepborders)
    {
      prepSDborders();
      prepborders = false;
    }
  for (jborder = 0; 
       jborder <(NSDEDGELINES+NSDTSHAPELINESBR+NSDTSHAPELINESLR+NSDTSHAPELINESSK); 
       jborder++)
    {
      // For drawing SD edge borders
      if (jborder < NSDEDGELINES)
	{
	  iborder=jborder; 
	  x1 = sdedgelinesRAW[iborder][0];
	  x2 = sdedgelinesRAW[iborder][2];    
	  y1 = sdedgelinesRAW[iborder][1];
	  y2 = sdedgelinesRAW[iborder][3];
	  vx = sdedgelines[iborder][2];
	  vy = sdedgelines[iborder][3];
	}
      // For drawing BR T-shape borders
      else if (jborder < (NSDEDGELINES+NSDTSHAPELINESBR))
	{
	  iborder=jborder-NSDEDGELINES; 
	  x1 = sdtshapelinesRAWBR[iborder][0];
	  x2 = sdtshapelinesRAWBR[iborder][2];    
	  y1 = sdtshapelinesRAWBR[iborder][1];
	  y2 = sdtshapelinesRAWBR[iborder][3];
	  vx = sdtshapelinesBR[iborder][2];
	  vy = sdtshapelinesBR[iborder][3];
	}

      // For drawing LR T-shape borders
      else if (jborder < (NSDEDGELINES+NSDTSHAPELINESBR+NSDTSHAPELINESLR))
	{
	  iborder=jborder-NSDEDGELINES-NSDTSHAPELINESBR; 
	  x1 = sdtshapelinesRAWLR[iborder][0];
	  x2 = sdtshapelinesRAWLR[iborder][2];    
	  y1 = sdtshapelinesRAWLR[iborder][1];
	  y2 = sdtshapelinesRAWLR[iborder][3];
	  vx = sdtshapelinesLR[iborder][2];
	  vy = sdtshapelinesLR[iborder][3];
	}

      // For drawing SK T-shape borders
      else
	{
	  iborder=jborder-NSDEDGELINES-NSDTSHAPELINESBR-NSDTSHAPELINESLR; 
	  x1 = sdtshapelinesRAWSK[iborder][0];
	  x2 = sdtshapelinesRAWSK[iborder][2];    
	  y1 = sdtshapelinesRAWSK[iborder][1];
	  y2 = sdtshapelinesRAWSK[iborder][3];
	  vx = sdtshapelinesSK[iborder][2];
	  vy = sdtshapelinesSK[iborder][3];
	}
      

      set2draw = false;
      
      dx = x2-x1;
      dy = y2-y1;
      if (dx < 0.)
	{
	  val=x1; x1=x2; x2=val;
	  val=y1; y1=y2; y2=val;
	  dx *= -1.0;
	  dy *= -1.0;
	}
      
      xc = x1 + dx/2.;
      yc = y1 + dy/2.;

      if (x1 <= xlo && x2 <= xlo)
	continue;
      if (x1 >= xup && x2 >= xup)
	continue;
      if (y1 <= ylo && y2 <= ylo)
	continue;
      if (y1 >= yup && y2 >= yup)
	continue;
      if (
	  (((xlo-xc)*vx + (ylo-yc)*vy) > 0.0) &&
	  (((xlo-xc)*vx + (yup-yc)*vy) > 0.0) &&
	  (((xup-xc)*vx + (ylo-yc)*vy) > 0.0) &&
	  (((xup-xc)*vx + (yup-yc)*vy) > 0.0)
	  )
	continue;
      
      if (!set2draw && fabs(dy) < 1.e-3)
	{
	  x1=( x1<xlo ? xlo : x1);
	  x2=( x2>xup ? xup : x2);
	  set2draw = true;
	}      
      if (!set2draw && fabs(dx) < 1.e-3)
	{
	  if (dy > 0.)
	    {
	      y1 = ( y1<ylo ? ylo : y1);
	      y2 = ( y2>yup ? yup : y2);
	    }
	  else
	    {
	      y1 = ( y1>yup ? yup : y1);
	      y2 = ( y2<ylo ? ylo : y2);
	    }
	  set2draw = true;
	}
      
      if (!set2draw)
	{
	  if ( x1 < xlo) 
	    {
	      x1=xlo;
	      y1=yc+dy/dx*(x1-xc);
	    }
	  if (x2 > xup)
	    {
	      x2=xup;
	      y2=yc+dy/dx*(x2-xc);
	    }
	  
	  if (dy/dx > 0.)
	    {
	      if (y1<ylo)
		{
		  y1=ylo;
		  x1=xc+dx/dy*(y1-yc);
		}
	      if (y2>yup)
		{
		  y2=yup;
		  x2=xc+dx/dy*(y2-yc);
		}
	    }
	  else
	    {
	      
	      if (y1>yup)
		{
		  y1=yup;
		  x1=xc+dx/dy*(y1-yc);
		}
	      if (y2<ylo)
		{
		  y2=ylo;
		  x2=xc+dx/dy*(y2-yc);
		}
	    }
	}
      if (jborder < NSDEDGELINES)
	{
	  pass1plot_drawLine(x1,y1,x2,y2,3,2);
	}
      else
	{
	  pass1plot_drawLine(x1,y1,x2,y2,2,50);
	}
    }
  
}




// wCounter - what to use counter: 0-lower, 1-upper, 2-both
// hDraw - what 2D histogram to use
// Ecanvas - on what canvas to draw
// nIxxyys - number of XXYY intra-event indecies in the array
// iXXYYs - array of intra-event indecies for XXYY's
// *cXY - center of the display in SD units with respect to SD origin

void drawEvent ( int wCounter, TH2F *hDraw, TCanvas *Ecanvas,
		 int nIxxyys, int *iXXYYs,
		 Double_t *cXY,
		 Int_t nx = 11, 
		 Int_t ny = 11 )
{
  int i, j, k, ixxyy, xxyy;
  int nbx, nby, npts;
  int bxmin, bxmax, bymin, bymax;
  double dx, dy, DX, DY, xm, ym;
  double t1, t2;		// Earliest and Latest relative times
  double r;
  int xy[2];
  int eventReadNumber;
  char hTitle[0x100];
  double lcharge;		// for determining the largest number of muons
  int Xlimits[2];           // will pick the smallest for the minimum
  double Ylimits[2];
  
  double coreXY[2];           // core position with respect to SD origin
  
  char tower_name[3];     // to display the name of the tower

  int* circle_x = new Int_t[nIxxyys];
  int* circle_y = new Int_t[nIxxyys];
  double* circle_radii = new Double_t[nIxxyys];
  double* circle_times = new Double_t[nIxxyys];

 
  

  coreXY[0] = p1.rufptn->tyro_xymoments[wCounter][0];
  coreXY[1] = p1.rufptn->tyro_xymoments[wCounter][1];
  
		     
  

  if (nx>MAXEDSIZE  || ny > MAXEDSIZE) 
    {
      printf("Display size is too large, using the default size %d\n",11);
      nx = ny = 11;
    }

  if(nx < 1 || ny <1) 
    {
      printf("Display size is too small, using the default %d\n",11);
      nx = ny = 11;
    }


  // So that the axis labels don't get cluttered
  if(nx<19)  hDraw->GetXaxis()->SetLabelSize(0.04);
  if(ny<19)  hDraw->GetYaxis()->SetLabelSize(0.04);
  if(nx==19) hDraw->GetXaxis()->SetLabelSize(0.03);
  if(ny==19) hDraw->GetYaxis()->SetLabelSize(0.03);
  if(nx>19)  hDraw->GetXaxis()->SetLabelSize(0.02);
  if(ny>19)  hDraw->GetYaxis()->SetLabelSize(0.02);
  
  
  Xlimits[0] = (Double_t)TMath::Nint(cXY[0]) - (Double_t)nx/2;
  if(Xlimits[0] < 0) Xlimits[0] = 0;
  Xlimits[1] = Xlimits[0] + (Double_t)nx;

  Ylimits[0] = (Double_t)TMath::Nint(cXY[1]) - (Double_t)ny/2;
  if(Ylimits[0] < 0) Ylimits[0] = 0;
  Ylimits[1] = Ylimits[0] + (Double_t)ny;
  hDraw->GetXaxis()->SetRangeUser(Xlimits[0],Xlimits[1]);
  hDraw->GetYaxis()->SetRangeUser(Ylimits[0],Ylimits[1]);
  hDraw->GetXaxis()->SetNdivisions(nx+1);
  hDraw->GetYaxis()->SetNdivisions(ny+1);
  if (wCounter < 0 || wCounter > 2)
    {
      printf ("Counter label must be 0(lower),1(upper),2(both)\n");
      return;
    }
  Ecanvas->cd ();
  // event number in the tree
  eventReadNumber = p1.pass1tree->GetReadEvent ();
  printf ("Drawing event %d\n",eventReadNumber); 
  printf("Using ");
  switch (wCounter)
    {
    case 0:
      printf ("lower counters\n");
      break;
    case 1:
      printf ("upper counters\n");
      break;
    case 2:
      printf ("upper and lower counters\n");
      break;
    default:
      printf ("Illegal layer ID\n");
      return;
      break;
    }
  // Pulse height area of the largest hit
  lcharge = p1.lcharge[wCounter];
  printf("Largest charge: %.2f\n",lcharge);
  if (lcharge < 1e-2) lcharge = 1.0; // caution


  switch(p1.rusdraw->site)
    {
    case 0:
      sprintf(tower_name,"BR");
      break;
    case 1:
      sprintf(tower_name,"LR");
      break;
    case 2:
      sprintf(tower_name,"SK");
      break;
    case 3:
      sprintf(tower_name,"BRLR");
      break;
    case 4:
      sprintf(tower_name,"BRSK");
      break;
    case 5:
      sprintf(tower_name,"LRSK");
      break;
    case 6:
      sprintf(tower_name,"BRLRSK");
      break;
    default:
      sprintf(tower_name,"??");
      break;
    }
  
  // sprintf (hTitle, "Event %d, LCHARGE=%.2f#mu ", eventReadNumber, lcharge);

  sprintf (hTitle, "%s : %d/%02d/%02d %02d:%02d:%02d.%06d",
	   tower_name,
	   (2000+p1.rusdraw->yymmdd/10000),((p1.rusdraw->yymmdd/100)%100),(p1.rusdraw->yymmdd%100),
	   (p1.rusdraw->hhmmss/10000),((p1.rusdraw->hhmmss/100)%100),(p1.rusdraw->hhmmss%100),
	   (Int_t)Floor((p1.tEarliest[wCounter]-floor(p1.tEarliest[wCounter]))*1e6+0.5));
  
	   
  
//   if (wCounter == 0) strcat (hTitle, " (lower)");
//   if (wCounter == 1) strcat (hTitle, " (upper)");
//   if (wCounter == 2) strcat (hTitle, " (upper & lower)");
  fprintf (stdout,"%s%7s%7s%16s%18s\n", "hit","folds","XXYY","Charge","Rel. Time");
  fprintf (stdout,"%s%5s%10s%16s%18s\n","num","#","(pos.)","(in VEM)","(in 1200m)");
  t1 = 1e10;
  t2 = 0.0;			// initialize earliest and latest relative hit times
  npts = 0;

  for (i = 0; i < nIxxyys; i++)
    {
//       fprintf(stdout,"%04d %f %f %f\n",p1.rufptn->xxyy[i],
// 	      p1.rufptn->xyzclf[i][0],p1.rufptn->xyzclf[i][1],
// 	      p1.rufptn->xyzclf[i][2]);
//       continue
      ixxyy = iXXYYs[i];
      xxyy = p1.rufptn->xxyy[ixxyy];
      p1.xycoor(xxyy, xy);
      // use lower, upper, or upper and lower counters, depending on
      // whether wCounter is 0,1,or 2.
      // circle_radii[npts] = sqrt(sqrt(p1.charge[ixxyy][wCounter] / lcharge));
      circle_radii[npts] = 
	sqrt(TMath::Log10(p1.charge[ixxyy][wCounter])/TMath::Log10(lcharge));
      circle_times[npts] = p1.relTime[ixxyy][wCounter];
      fprintf (stdout,"%02d%7.02d%8.04d%10.2f%5s%7.2f%10.1f\n",
	       ixxyy, p1.rufptn->nfold[ixxyy],xxyy,p1.charge[ixxyy][wCounter],"+/-",
	       p1.chargeErr[ixxyy][wCounter],p1.relTime[ixxyy][wCounter]);
      
      // Get only the counters which fit into our window
      if(xy[0] < Xlimits[0] || xy[0] > Xlimits[1] ||
	 xy[1] < Ylimits[0] || xy[1] > Ylimits[1]) continue;
      circle_x[npts] = xy[0];
      circle_y[npts] = xy[1];
      
      // To determine the earliest and the latest relative hit times
      if (circle_times[npts] < t1) t1 = circle_times[npts];
      if (circle_times[npts] > t2) t2 = circle_times[npts];
      npts++;
    }				// for(i=0;i<nIxxyy;i++) 
  printf ("Number of points: %d\n", npts);
  // Also display the earlist event GPS time
  printf ("Earliest GPS time: ");
  printf ("%.6f\n", p1.tEarliest[wCounter]);

  hDraw->Reset();
  
  // set histogram minimum and maximum to the times requested,
  // so that only this range shows up on the zcol plot
  hDraw->SetMinimum ((1-1e-6)*t1);
  hDraw->SetMaximum (t2*(1.0 + 1e-6));	// set the maximum slightly bigger
  
  
  xm = hDraw->GetXaxis ()->GetXmin ();
  ym = hDraw->GetYaxis ()->GetXmin ();
  nbx = hDraw->GetNbinsX ();
  nby = hDraw->GetNbinsY ();
  DX = hDraw->GetXaxis ()->GetXmax () - xm;
  dx = DX / (double) nbx;
  DY = hDraw->GetYaxis ()->GetXmax () - ym;
  dy = DY / (double) nby;
  for (k = 0; k < npts; k++)
    {
      bxmin = (int) ((circle_x[k] - circle_radii[k] - xm) / dx) - 1;
      if (bxmin < 1) bxmin = 1;
      bxmax = (int) ((circle_x[k] + circle_radii[k] - xm) / dx) + 1;
      if (bxmax > nbx) bxmax = nbx;
      bymin = (int) ((circle_y[k] - circle_radii[k] - ym) / dy) - 1;
      if (bymin < 1) bymin = 1;
      bymax = (int) ((circle_y[k] + circle_radii[k] - ym) / dy) + 1;
      if (bymax > nby) bymax = nby;
      for (i = bxmin; i <= bxmax; i++)
	{
	  for (j = bymin; j <= bymax; j++)
	    {
	      r = sqrt ((circle_x[k] - xm - DX * (double) i / (double) nbx) *
			(circle_x[k] - xm - DX * (double) i / (double) nbx) +
			(circle_y[k] - ym - DY * (double) j / (double) nby) *
			(circle_y[k] - ym - DY * (double) j / (double) nby));
	      if (r < circle_radii[k])
		{
		  
		  // So that 0 times are also displayed
		  if (circle_times[k]*circle_times[k] < 1e-10) circle_times[k] = 1e-5;
		  hDraw->SetBinContent (i, j, circle_times[k]);
		}
	    }
	}
    }
  
  // Draw event and the detector sites
  hDraw->SetTitle(hTitle);
  
  hDraw->Draw ("zcol");
   // Addd label for time
  Double_t xmax,ymax;
  TPaveLabel *l = new TPaveLabel(Xlimits[1]-1.0,Ylimits[1],
				 Xlimits[1],Ylimits[1]+1.0,"Time,  [1200m]");
  l->SetTextSize(0.6);
  l->SetBorderSize(0);
  l->SetFillStyle(0);
  l->SetTextFont(62);
  l->Draw();
  // Draw the long and short axes.  Draw an arrow along the long axis
  if(Draw_Tyro_Arrow)
    {
      pass1plot_drawArrow (coreXY,&p1.rufptn->tyro_u[wCounter][0], 0.05, 6.0, 2);
      pass1plot_drawLine  (coreXY,&p1.rufptn->tyro_v[wCounter][0], 3.0, 2);
    }
  drawBorders(Xlimits[0],Xlimits[1],Ylimits[0],Ylimits[1]);

  // clean up
  delete[] circle_x;
  delete[] circle_y;
  delete[] circle_radii;
  delete[] circle_times;
  
}


// Will draw all the counters that were hit 
// wCounter - what to use counter: 0-lower, 1-upper, 2-both
// hDraw - what 2D histogram to use
// Ecanvas - on what canvas to draw
void drawEvent (int wCounter, TH2F * hDraw, TCanvas * Ecanvas, double *cXY,
		Int_t nx = 11, Int_t ny = 11)
{
  int *iXXYYs = new int[p1.rufptn->nhits];
  for (Int_t i = 0; i < p1.rufptn->nhits; i++)
    iXXYYs[i] = i;
  drawEvent (wCounter, hDraw, Ecanvas,p1.rufptn->nhits, iXXYYs,cXY,nx,ny);
  delete[] iXXYYs;
}

void drawEvent (int wCounter, TH2F * hDraw, TCanvas * Ecanvas, 
	   Double_t x = 15.0, 
	   Double_t y = 15.0,
	   Int_t nx = 11, Int_t ny = 11)
{
  Double_t cXY[2];
  cXY[0] = x; cXY[1] = y;
  drawEvent (wCounter, hDraw, Ecanvas,cXY,nx,ny);
}
