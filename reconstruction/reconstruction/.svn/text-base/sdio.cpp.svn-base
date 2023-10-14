#include "sdanalysis_icc_settings.h"
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <vector>
#include <sys/poll.h>
#include "sduti.h"

using namespace std;


static const int nl1suf = 3;
static const int nl2suf = 3;
static const char l1_suf[3][10] =
  { ".dst", ".dst.gz", ".dst.bz2" };
static const char l2_suf[3][10] =
  { ".rusdraw", ".rufptn", ".rufldf" };

// Gets the suffix after the last '.'.   If file name doesn't contain '.' then returns 0.
// returns value is the position in the fname string of the last '.'
int SDIO::getSuffix(const char *fname, char *suffix)
{
  const char *ch = strrchr (fname, '.');
  if (!ch) 
    return 0;
  memcpy(suffix,ch,((int)strlen(fname) - (int)(ch-fname) + 1));
  return (int)(ch-fname);
}

// Gets the DST suffix of a DST file into the string suffix
// returns 0 if fails, 1 if success
int SDIO::getDSTsuffix(const char *fname, char *suffix)
  {
    char inSuffix[0x100];
    for (int i = 0; i < nl1suf; i++)
      {
        const char* ch = fname - 1;
        do
          {
            ch = strstr(ch + 1, l1_suf[i]);
          } while (ch && (strstr(ch + 1, l1_suf[i])));
        if (!ch || (*(ch + strlen(l1_suf[i]))))
          continue;
        for (int j = 0; j < nl2suf; j++)
          {
            sprintf(inSuffix, "%s%s", l2_suf[j], l1_suf[i]);
            ch = fname - 1;
            do
              {
                ch = strstr(ch + 1, inSuffix);
              } while (ch && (strstr(ch + 1, inSuffix)));
            if (!ch || (*(ch + strlen(inSuffix))))
              continue;
            memcpy(suffix, inSuffix, strlen(inSuffix) + 1);
            return 1;
          }
        memcpy(suffix, l1_suf[i], strlen(l1_suf[i]) + 1);
        return 1;
      }
    suffix[0] = '\0';
    return 0;
  }


// This is for the case when we expect an input suffix:
// Replace the input suffix with the output suffix
// and obtain the full path for the output file
int SDIO::GetOutFileName (const char *inFile, const char *outDir,
			  const char *inSuffix, const char *outSuffix, 
			  char *outFile)
{

  char* outDir1 = (char*) malloc(strlen(outDir)+2); // extra byte for a possible '/' character
  if(!outDir1)
    {
      printErr("ERROR: failed to allocated memory");
      exit(2);
    }
  memcpy(outDir1,outDir,strlen(outDir)+1);

  // Make sure that the output directory ends wiht '/'
  int l = (int)strlen(outDir1);
  if(outDir1[l-1]!='/')
    {
      outDir1[l] = '/';
      outDir1[l+1] = '\0';
    }

  const char *ch = strstr (inFile, inSuffix);
  if (ch != NULL)
    {
      while ((strstr (ch + 1, inSuffix)) != NULL)
	{
	  ch = strstr (ch + 1, inSuffix);
	}
    }
  
  // Making sure that the file name ends with the expected suffix
  if (!ch || (*(ch + strlen (inSuffix))))
    {
      printErr ("%s: expected suffix is '%s' ; can't process",
		inFile, inSuffix);
      return 0;
    }
  // Input file name must be greater than the input suffix
  else
    {				
      ch = strrchr (inFile, '/');
      if (!ch)
	ch = inFile - 1;
      l = strlen (inFile) - (ch + 1 - inFile) - strlen (inSuffix);
      if (l == 0)
	{
	  printErr("%s: bad input file name, need more characters",
		   inFile);
	  return 0;
	}
      memcpy (outFile, outDir1, strlen (outDir1) + 1);
      memcpy (outFile + strlen (outDir1), ch + 1, l);
      *(outFile + strlen (outDir1) + l) = '\0';
      memcpy (outFile + strlen (outFile), outSuffix, strlen (outSuffix) + 1);
    }
  free(outDir1);
  return 1;
}

// This is for the case when we don't expect an input suffix
// Just add the output suffix to the file name
int SDIO::GetOutFileName (const char *inFile, const char *outDir, const char *outSuffix, 
			  char *outFile)
{
  char* outDir1 = (char*) malloc(strlen(outDir)+2); // extra byte for a possible '/' character
  if(!outDir1)
    {
      printErr("ERROR: failed to allocated memory");
      exit(2);
    }
  memcpy(outDir1,outDir,strlen(outDir)+1);
  
  // Make sure that the output directory ends wiht '/'
  int l = (int)strlen(outDir1);
  if(outDir1[l-1]!='/')
    {
      outDir1[l] = '/';
      outDir1[l+1] = '\0';
    }
  
  // Find where the directories end
  // and the input file name begins
  const char* ch = strrchr (inFile, '/');


  // When input file is in the same directory as the program
  if (ch==NULL) ch = inFile - 1;


  l = strlen (inFile) - (ch + 1 - inFile);
  if (l == 0)
    {
      printErr("%s: bad input file name, need more characters",inFile);
      return 0;
    }

  
  memcpy (outFile, outDir1, strlen (outDir1));
  memcpy (outFile + strlen (outDir1), ch + 1, l);
  *(outFile + strlen (outDir1) + l) = '\0';
  memcpy (outFile + strlen (outFile), outSuffix, strlen (outSuffix) + 1);

  free(outDir1);
  return 1;
}


// This is for the case when we either expected a variety of input suffixes
// Replace the input suffix with the output suffix
// and obtain the full path for the output file
int SDIO::makeOutFileName(const char *inFile, const char *outDir, const char *outSuffix, 
			  char *outFile)
{
  char inSuffix[0x400];
  if (getDSTsuffix(inFile, inSuffix))
    return GetOutFileName(inFile, outDir, inSuffix, outSuffix, outFile);
  printErr("'%s': Bad DST suffix: use either '.dst' or '.dst.gz' or '.dst.bz2'",inFile);
  return 0;
}

int SDIO::check_dst_suffix(const char *dstname)
{
  char dst_suffix[0x400];
  if(!getDSTsuffix(dstname,dst_suffix))
    {
      printErr("'%s': Bad DST suffix: use either '.dst' or '.dst.gz' or '.dst.bz2'",dstname);
      return 0;
    }
  return 1;
}

// Substitutes the 1st found pattern1 in str1 with pattern2 and gives the result in str2.  
int SDIO::patternSubst(const char *str1, const char *pattern1, const char *pattern2, char *str2)
{
  char *ch;
  char *dummy;
  dummy = new char[strlen(str1)+strlen(pattern2)+1];
  memcpy(dummy,str1,strlen(str1)+1);
  if((ch=strstr(dummy,pattern1))==NULL)
    {
      delete [] dummy;  
      return 0;
    }
  memcpy(str2,dummy,(ch-dummy));
  memcpy(str2+(ch-dummy),pattern2,strlen(pattern2)+1);
  memcpy(str2+strlen(str2),ch+strlen(pattern1),
	 (strlen(dummy)-(ch-dummy)+1));
  delete [] dummy;
  return 1;
}

int SDIO::have_stdin(int poll_timeout_ms)
{
  pollfd fds;
  fds.fd     = 0;
  fds.events = POLLIN;
  return (poll(&fds, 1, poll_timeout_ms) == 1 ? 1: 0);
}

void SDIO::vprintMessage(FILE *fp, const char *who, const char *what, va_list args)
{
  va_list args_args;
  // older C++ compilers may not have va_copy properly defined
#ifndef va_copy
#define va_copy(dest, src) __builtin_va_copy(dest, src)
#endif
  va_copy(args_args, args);
  // assume that most strings are shorter than this; if the string
  // is larger then the routine will resize the buffer and redo the printing
  vector<char> buf(1024);
  size_t len = vsnprintf(&buf.front(), buf.size(), what, args);
  // if the buffer size wasn't enough to print the string and the
  // null terminating character, then resize the buffer and re-print the string
  if(len >= buf.size())
    {
      buf.resize(len + 1);
      vsnprintf(&buf.front(), buf.size(), what, args_args);
    }
  va_end(args_args);
  // print the result
  if(who)
    fprintf(fp, "%s: %s\n", who, &buf.front());
  else
    fprintf(fp, "%s\n", &buf.front());
  fflush(fp);
}


void SDIO::printMessage(FILE *fp, const char *who, const char *what, ...)
{
  va_list args;
  va_start(args, what);
  vprintMessage(fp, who, what, args);
  va_end(args);
  fflush(fp);
}

string SDIO::strprintf(const char* what, ...)
{
  // assume that most strings are shorter than this; if the string
  // is larger then the routine will resize the buffer and redo the printing
  vector<char>buf(1024);
  va_list args;
  va_start(args, what);
  size_t len = vsnprintf(&buf.front(), buf.size(), what, args);
  va_end(args);
  // if the string fits into the buffer then return the answer
  if(len < buf.size())
    return string(&buf.front());
  // if the string did not fit into this size then resize the buffer 
  // to a correct string length and re-print
  va_start(args, what);
  buf.resize(len + 1);
  vsnprintf(&buf.front(), len + 1, what, args);
  va_end(args);
  return string(&buf.front());
}


void SDIO::printErr(const char *form, ...)
{
  va_list args;
  va_start(args, form);
  vprintMessage(stderr,"sdio",form,args);
  va_end(args);
}


