#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void)
{
  char inbuf[0x100];
  int pat[16];
  int i,n;
  int pos;
  while(fgets(inbuf,sizeof(inbuf),stdin))
    {
      n=sscanf(inbuf,
	       "#P %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x",
	       &pos,
	       &pat[0],
	       &pat[1],
	       &pat[2],
	       &pat[3],
	       &pat[4],
	       &pat[5],
	       &pat[6],
	       &pat[7],
	       &pat[8],
	       &pat[9],
	       &pat[10],
	       &pat[11],
	       &pat[12],
	       &pat[13],
	       &pat[14],
	       &pat[15]
	       );
      if (n <=1 )
	{
	  fprintf(stderr,"line must start with #P and have at least one hex value after #P\n");
	  continue;
	}
      fprintf(stdout,"#P %04d", ((pos&0x3f)+(100*((pos>>6)&0x3f))));
      for (i=0; i<(n-1); i++)
	fprintf(stdout," %04d:%06d", (((pat[i]>>20)&0x3f)+(100*((pat[i]>>26)&0x3f))),(pat[i]&0xfffff));
      fprintf(stdout,"\n");
      fflush(stdout);
    }
  return 0;
}
