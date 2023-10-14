#include "sdaborder_raw.h"
#include "event.h"

class sdaborder_class
{
public:
  sdaborder_class();
  virtual ~sdaborder_class();
  bool isOnBorder(int x, int y);
  bool isOnBorder(int xxyy);
  
private:
  bool sda[SDMON_X_MAX][SDMON_Y_MAX];
};



sdaborder_class::sdaborder_class()
{
  int i,j,x,y;
  for (i=0; i<SDMON_X_MAX;i++)
    {
      for (j=0; j<SDMON_Y_MAX; j++)
	{
	  sda[i][j] = false; 
	}
    } 

  for (i=0; i<nbrborder; i++)
    {
      x = BR_border[i] / 100;
      y = BR_border[i] % 100;
      sda[x-1][y-1] = true;
    }
  for(i=0; i<nlrborder;i++)
    {
      x = LR_border[i] / 100;
      y = LR_border[i] % 100;
      sda[x-1][y-1] = true; 
    }
  
  for(i=0; i<nskborder;i++)
    {
      x = SK_border[i] / 100;
      y = SK_border[i] % 100;
      sda[x-1][y-1] = true; 
    }
  
  
}


sdaborder_class::~sdaborder_class()
{
}



bool sdaborder_class::isOnBorder(int x, int y)
{
  
  if ((x < 1) || (x > SDMON_X_MAX) || 
      (y < 1) || (y > SDMON_Y_MAX))
    {
      fprintf(stderr,"isOnBorder: %d %d is an invalid SD location\n",x,y);
      return true;
    }

  if(sda[x-1][y-1])
    return true;

  return false;
  
}


bool sdaborder_class::isOnBorder(int xxyy)
{
  int x, y;
  x = xxyy/100;
  y = xxyy%100;
  return isOnBorder(x,y);
}



