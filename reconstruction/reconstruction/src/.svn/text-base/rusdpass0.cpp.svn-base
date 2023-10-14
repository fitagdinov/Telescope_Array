/* 
 * Program for parsing raw TASD data:
 * Reads the raw SD data, time-matches events and monitoring cycles,
 * calibrates events, and writes out DST files: one with events and 
 * one with monitoring cycles.
 * 
 * 
 * 
 * 
 * Dmitri Ivanov <ivanov@physics.rutgers.edu>
 * Created: Aug 10, 2009
 * Last Modified: Aug 10, 2009
 */

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rusdpass0io.h"
#include "tower_parser.h"
#include "sdindexing.h"
#include "parsingmanager.h"
#include "event.h"

int main(int argc, char **argv)
{
  rusdpass0io *p0io = 0; // I/O handler
  parsingManager *parser; // Parsing managing class
  
  // Get the command line arguments
  listOfOpt opt;
  if (!opt.getFromCmdLine(argc, argv))
    exit(2);
    
  // Print program run options
  opt.printOpts();

  // Initialize I/O handler
  p0io = new rusdpass0io(opt);

  // Initialize the parser
  parser = new parsingManager(opt,p0io);

  // Start parsing the data
  parser->Start();

  // Print out stats for the day
  parser->printStats();

  // Finish I/O (close DST files, etc) and clean up
  delete p0io;
  delete parser;

  fprintf(stdout,"\n\nDone\n");
  fflush(stdout);
  return 0;
}
