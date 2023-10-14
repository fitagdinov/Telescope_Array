
/************** DRAWING LINES ***********************/ 

// To draw a line of given width and color
void pass1plot_drawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
	      Short_t lwidth,Short_t lcolor){
  TLine *myline=new TLine(x1,y1,x2,y2);
  myline->SetLineWidth(lwidth);
  myline->SetLineColor(lcolor);
  myline->Draw();
}

// To draw a black line of a given width
void pass1plot_drawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
	      Short_t lwidth){
  pass1plot_drawLine(x1,y1,x2,y2,lwidth,1);
}
// Draw a thin red line
void pass1plot_drawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2){
  pass1plot_drawLine(x1,y1,x2,y2,1,2);
}

// To draw lines along the unit vector u; p0 is the offset
// draws lines of given width and color
void pass1plot_drawLine(const Double_t* p0, 
			const Double_t* u, Double_t length,
			Short_t lwidth,Short_t lcolor)
{
  Double_t x1 = p0[0] - length/2.0*u[0];
  Double_t y1 = p0[1] - length/2.0*u[1];
  Double_t x2 = p0[0] + length/2.0*u[0];
  Double_t y2 = p0[1] + length/2.0*u[1];
  pass1plot_drawLine(x1,y1,x2,y2,lwidth,lcolor);
}

// To draw lines along the unit vector u; p0 is the offset
// draws black lines of a given width
void pass1plot_drawLine(const Double_t* p0, const Double_t* u, Double_t length,Short_t lwidth)
{
  Double_t x1 = p0[0] - length/2.0*u[0];
  Double_t y1 = p0[1] - length/2.0*u[1];
  Double_t x2 = p0[0] + length/2.0*u[0];
  Double_t y2 = p0[1] + length/2.0*u[1];
  pass1plot_drawLine(p0,u,length,lwidth,1);
}
void pass1plot_drawLine(const Double_t* lpars, Double_t length, Short_t lwidth, Short_t lcolor)
{
  Double_t p0[2] = {0.0, lpars[0]};
  Double_t u[2]  = {1.0/sqrt(1+lpars[1]*lpars[1]), u[0]*lpars[1]};
  pass1plot_drawLine(p0,u,length,lwidth,lcolor);
}
void pass1plot_drawLine(const Double_t* lpars, Double_t length, Short_t lwidth)
{
  pass1plot_drawLine(lpars,length,lwidth,1);
}



/**************** DRAWING ARROWS *******************/


// To draw an arrow directed to x2,y2 of given 
// arrow head size, width and color.
void pass1plot_drawArrow(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
			 Double_t arrowsize = 0.05, 
			 Short_t lwidth = 2, 
			 Short_t lcolor=kBlack){
  TArrow *myarrow=new TArrow(x1,y1,x2,y2,(Float_t)arrowsize,">");
  myarrow->SetLineWidth(lwidth);
  myarrow->SetLineColor(lcolor);
  myarrow->Draw();
}
// To draw arrows along the unit vector u; p0 is the offset
void pass1plot_drawArrow(const Double_t* p0, 
			 const Double_t* u, 
			 Double_t arrowsize = 0.05,
			 Double_t length = 6.0,
			 Short_t lwidth = 2,
			 Short_t lcolor = kBlack)
{
  Double_t x1 = p0[0] - length/2.0 * u[0];
  Double_t y1 = p0[1] - length/2.0 * u[1];
  Double_t x2 = p0[0] + length/2.0 * u[0];
  Double_t y2 = p0[1] + length/2.0 * u[1];
  pass1plot_drawArrow(x1,y1,x2,y2,arrowsize,lwidth,lcolor);
}



/***************** DRAWING MARKS ****************/

void pass1plot_drawmark(Double_t x, Double_t y, Double_t msize, Int_t marker, Short_t mcolor){
  TMarker *mymarker = new TMarker (x,y,marker);
  mymarker->SetMarkerSize((Size_t)msize);
  mymarker->SetMarkerColor(mcolor);
  mymarker->Draw();
}

// Draw a simple mark of a given size and color
void pass1plot_drawmark(Double_t x, Double_t y,Double_t msize,Short_t mcolor){
  pass1plot_drawmark(x,y,msize,5,mcolor);
}
// Draw a simple mark of black color of a given size
void pass1plot_drawmark(Double_t x, Double_t y, Double_t msize){
  pass1plot_drawmark(x,y,msize,5,1);
}


// To plot a 2D scatter histogram with a profile plot on the top
void pass1plot_plotScat(TH2F *pass1plot_hScat, TProfile *pass1plot_pScat)
{
  pass1plot_hScat->Draw("box");
  pass1plot_pScat->SetLineWidth(3);
  pass1plot_pScat->SetLineColor(2);
  pass1plot_pScat->Draw("same");
}




