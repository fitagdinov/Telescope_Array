#include <cmath>
#include "sduti.h"
#include <assert.h>

using namespace std;

double SDGEN::linear_interpolation( int n, double *x, double *y, double t )
{
  // Simple linear interpolation taken from CERN Root TGraph code
  int low  = -1;
  int up  = -1;
  int low2 = -1;
  int up2 = -1;
  for (int i = 0; i < n; ++i) 
    {
      if (x[i] < t)
	{
	  if (low == -1 || x[i] > x[low])
	    {
	      low2 = low;
	      low = i;
	    } 
	  else if (low2 == -1) 
	    low2 = i;
	} 
      else if (x[i] > t) 
	{
	  if (up  == -1 || x[i] < x[up])  
	    {
	      up2 = up;
	      up = i;
	    } 
	  else if (up2 == -1) 
	    up2 = i;
	} 
      else
	return y[i]; // no interpolation needed
    }
  // treat cases when t is outside graph min mat abscissa
  if (up == -1)  
    {
      up  = low;
      low = low2;
    }
  if (low == -1) 
    {
      low = up;
      up  = up2;
    }
  assert(low != -1 && up != -1);
  if (x[low] == x[up]) 
    return y[low];
  return y[up]+(t-x[up])*(y[low]-y[up])/(x[low]-x[up]);
  
}
