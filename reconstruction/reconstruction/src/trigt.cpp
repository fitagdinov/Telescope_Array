#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void)
{
  char inbuf[0x100];
  unsigned int eno, etim;
  int n;
  while(fgets(inbuf,sizeof(inbuf),stdin))
    {
      n=sscanf(inbuf,"E %x %x", &eno,&etim);
      if (n != 2 )
	{
	  fprintf(stderr,"The E-line is corrupted\n");
	  continue;
	}
      
      fprintf(stdout,"E %08d %03d %06d\n",
	      eno,((etim>>20)&0xfff),(etim&0xfffff));
      fflush(stdout);
    }
  return 0;
}
