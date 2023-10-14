void plot_rsd()
{
  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4])","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0");
}


void plot_rsd_vs_height()
{
  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):1e5*gdas->height[]","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","box");


  
  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):1e5*gdas->height[]>>p(100,0,1e7)","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","profs,same");
}

void plot_rsd_vs_mo()
{
  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):100.0/98.0*gdas->pressure[]","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","box");
  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):100.0/98.0*gdas->pressure[]>>p(120,0,1200.0)","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","profs,same");
  
}


void plot_inv_rsd_vs_mo()
{
  taTree->Draw("1e5*gdas->height[]-mo2h(100.0/98.0*gdas->pressure[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):100.0/98.0*gdas->pressure[]","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","box");
  taTree->Draw("1e5*gdas->height[]-mo2h(100.0/98.0*gdas->pressure[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):100.0/98.0*gdas->pressure[]>>p(120,0,1200.0)","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","profs,same");
  
}

void plot_rsd_vs_day()
{
  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):(gdas->dateFrom-1210464000)/86400.0","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","box");

  taTree->Draw("Air_Density_g_cm3(100.0*gdas->pressure[],273.15+gdas->temperature[],273.15+gdas->dewPoint[])/rho(1e5*gdas->height[],atmpar->h[0],atmpar->h[1],atmpar->h[2],atmpar->h[3],atmpar->h[4],atmpar->a[0],atmpar->a[1],atmpar->a[2],atmpar->a[3],atmpar->a[4],atmpar->b[0],atmpar->b[1],atmpar->b[2],atmpar->b[3],atmpar->b[4],atmpar->c[0],atmpar->c[1],atmpar->c[2],atmpar->c[3],atmpar->c[4]):(gdas->dateFrom-1210464000)/86400.0>>p(3652,0,3652.0)","chi2/ndof<100&&(gdas->dateFrom-1210464000)/86400.0>0","profs,same");

}


